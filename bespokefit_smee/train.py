"""Apply OpenFF parameters to molecule, cluster conformers by RMSD and train"""

import copy
import datetime
import functools
import logging
from contextlib import redirect_stderr

import datasets
import datasets.combine
import descent
import descent.optim
import loguru
import openff.toolkit
import smee
import tensorboardX
import torch
from tqdm import tqdm

from .data_maker import get_data_MLMD, get_data_MMMD
from .loss_functions import get_loss_closure_fn, prediction_loss
from .parameterizer import build_parameters, convert_to_smirnoff
from .settings import DEFAULT_CONFIG_PATH, TrainingConfig
from .writers import (
    get_potential_comparison,
    open_writer,
    report,
    write_metrics,
    write_scatter,
)

logging.basicConfig(
    level=logging.INFO,  # or logging.DEBUG for more detail
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logging.getLogger("descent").setLevel(logging.DEBUG)

logger = loguru.logger


def _iterate_training_levenberg_marquardt(
    trainable_parameters: torch.Tensor,
    trainable: descent.train.Trainable,
    topology: smee.TensorTopology,
    dataset: datasets.Dataset,
    dataset_test: datasets.Dataset,
    config: TrainingConfig,
    iteration: int,
) -> tuple[torch.Tensor, descent.train.Trainable]:
    """
    Iterate the training process using the Levenberg-Marquardt algorithm.

    Parameters
    ----------
        trainable_parameters: torch.Tensor
            The parameters to be optimized.
        trainable: descent.train.Trainable
            The trainable object containing the parameters.
        topology: smee.TensorTopology
            The topology of the system.
        dataset: datasets.Dataset
            The dataset to be used for training.
        dataset_test: datasets.Dataset
            The dataset to be used for testing.
        config: TrainingConfig
            The configuration object containing training parameters.
        iteration: int
            The current iteration number.

    Returns
    -------
        tuple[torch.Tensor, float, float, float, float]
            The updated parameters and the mean and standard deviation of energy and forces.
    """
    # Some book-keeping
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = config.output_dir / timestamp

    # Run the training with the LM optimiser
    lm_config = descent.optim.LevenbergMarquardtConfig(
        mode="adaptive", n_convergence_criteria=2, max_steps=100
    )

    closure_fn = get_loss_closure_fn(
        trainable,
        topology,
        dataset,
    )

    correct_fn = trainable.clamp

    report_fn = functools.partial(
        report,
        trainable=trainable,
        topology=topology,
        dataset_test=dataset_test,
        metrics_file=config.output_dir / f"training-{iteration}.data",
        experiment_dir=experiment_dir,
    )

    trainable_parameters = descent.optim.levenberg_marquardt(
        trainable_parameters, lm_config, closure_fn, correct_fn, report_fn
    )
    trainable_parameters.requires_grad_(True)

    return trainable_parameters, trainable


def _iterate_training_adam(
    trainable_parameters: torch.Tensor,
    trainable: descent.train.Trainable,
    topology: smee.TensorTopology,
    dataset: datasets.Dataset,
    dataset_test: datasets.Dataset,
    config: TrainingConfig,
    iteration: int,
) -> tuple[torch.Tensor, descent.train.Trainable]:
    """
    Iterate the training process using the Adam optimizer.

    Parameters
    ----------
        trainable_parameters: torch.Tensor
            The parameters to be optimized.
        trainable: descent.train.Trainable
            The trainable object containing the parameters.
        topology: smee.TensorTopology
            The topology of the system.
        dataset: datasets.Dataset
            The dataset to be used for training.
        dataset_test: datasets.Dataset
            The dataset to be used for testing.
        config: TrainingConfig
            The configuration object containing training parameters.
        iteration: int
            The current iteration number.

    Returns
    -------
        tuple[torch.Tensor, float, float, float, float]
            The updated parameters and the mean and standard deviation of energy and forces.
    """
    # some book-keeping
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = config.output_dir / timestamp

    # run the ML training
    with open(config.output_dir / f"training-{iteration}.data", "w") as metrics_file:
        with open_writer(experiment_dir) as writer:
            optimizer = torch.optim.Adam(
                [trainable_parameters], lr=config.learning_rate, amsgrad=True
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=config.learning_rate_decay
            )
            for v in tensorboardX.writer.hparams(
                {"optimizer": "Adam", "lr": config.learning_rate}, {}
            ):
                writer.file_writer.add_summary(v)
            for i in tqdm(
                range(config.n_epochs),
                leave=False,
                colour="blue",
                desc="Running ML Optimization",
            ):
                loss_trn = prediction_loss(
                    dataset,
                    trainable.to_force_field(trainable_parameters),
                    topology,
                    config.loss_force_weight,
                    config.device_type,
                )
                if i % 10 == 0:
                    loss_tst = prediction_loss(
                        dataset_test,
                        trainable.to_force_field(trainable_parameters),
                        topology,
                        config.loss_force_weight,
                        config.device_type,
                    )
                    write_metrics(i, loss_trn, loss_tst, writer, metrics_file)
                loss_trn.backward(retain_graph=True)  # type: ignore[no-untyped-call]
                # trainable.freeze_grad()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                trainable.clamp(trainable_parameters)
                if i % config.learning_rate_decay_step == 0:
                    scheduler.step()
        # some book-keeping and outputting
        loss_tst = prediction_loss(
            dataset_test,
            trainable.to_force_field(trainable_parameters),
            topology,
            config.loss_force_weight,
            config.device_type,
        )
        write_metrics(config.n_epochs, loss_trn, loss_tst, writer, metrics_file)

        return trainable_parameters, trainable


def train(config: TrainingConfig, write_config: bool = True) -> None:
    """
    Train a bespoke force field according to the provided configuration.

    Parameters:
        config  : TrainingConfig
            Configuration object containing all the necessary parameters for training.
        write_config: bool
            Whether to write the configuration to a YAML file in the output directory. Useful
            for reproducibility when the config has been read from command line arguments.
    """
    # Summarise input parameters
    logger.info(f"Training settings:\n{config.pretty_string}")

    # Create the output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save config to YAML file
    if write_config:
        config.to_yaml(yaml_path=config.output_dir / DEFAULT_CONFIG_PATH)

    timestep_ps = config.timestep / 1000  # Convert to ps

    # Parameterize the molecule and output the forcefield to a file
    mol = openff.toolkit.Molecule.from_smiles(
        config.smiles, allow_undefined_stereo=True, hydrogens_are_explicit=False
    )
    VdW_forcefield = openff.toolkit.ForceField(config.initial_force_field)
    old_force_field, trainable, topology = build_parameters(
        mol,
        VdW_forcefield,
        config.ml_potential,
        config.linear_harmonics,
        config.linear_torsions,
        config.use_modified_seminario,
        config.modified_seminario_finite_step / 10,  # Convert to nm
        config.modified_seminario_vib_scaling,
        config.modified_seminario_tolerance,
        device_type=config.device_type,
    )

    # Move to the requested device (not strictly necessary if on CPU)
    trainable_parameters = trainable.to_values().to((config.device_type))
    topology = topology.to(config.device_type)

    # Convert topology assignment tensors to dense - needed for some reason for hessian calc
    for param in topology.parameters.values():
        param.assignment_matrix = param.assignment_matrix.to_dense()

    off_force_field = convert_to_smirnoff(
        trainable.to_force_field(trainable_parameters), base=VdW_forcefield
    )
    off_force_field.to_file(config.output_dir / "trained-0.offxml")

    # get the inital training data

    if config.method == "data":
        dataset = datasets.Dataset.load_from_disk(config.data)
    else:
        dataset = config.run_md_fn(
            mol,
            off_force_field,
            config.ml_potential,
            config.temperature,
            timestep_ps,
            config.n_train_snapshots_per_conformer,
            config.n_conformers,
            config.snapshot_interval,
            config.n_equilibration_steps,
            config.energy_upper_cutoff,
            config.energy_lower_cutoff,
            config.cluster_tolerance,
            config.cluster_parallel,
            config.output_dir / "trajectory.pdb",
            config.output_dir / "bias_output",
        )

    with open("/dev/null", "w") as f:
        with redirect_stderr(f):
            dataset.save_to_disk(config.output_dir / "data_it_0")

    # Generate the test set
    if config.test_data_path is not None:
        logger.info(
            f"Loading test set from {config.test_data_path} instead of generating it"
        )
        dataset_test = datasets.Dataset.load_from_disk(config.test_data_path)
    else:
        logger.info("Generating Test Set with MLMD")
        dataset_test = get_data_MLMD(
            mol,
            off_force_field,
            config.ml_potential,
            config.temperature,
            timestep_ps,
            config.n_test_snapshots_per_conformer,
            config.n_conformers,
            config.snapshot_interval / 100,  # TODO: remove this hack
            config.n_equilibration_steps,
            config.energy_upper_cutoff,
            config.energy_lower_cutoff,
        )

        with open("/dev/null", "w") as f:
            with redirect_stderr(f):
                dataset_test.save_to_disk(config.output_dir / "data_test")

    # Generate the Energy Scatter Plot
    energy_mean, energy_SD, forces_mean, forces_SD = write_scatter(
        dataset_test,
        trainable.to_force_field(trainable_parameters),
        topology,
        config.device_type,
        config.output_dir / "trained-0.scat",
    )

    for iteration in tqdm(
        range(1, config.n_iterations + 1),  # Start from 1 (0 is untrained)
        leave=False,
        colour="magenta",
        desc="Iterating the Fit",
    ):
        iterate_training_fns = {
            "lm": _iterate_training_levenberg_marquardt,
            "adam": _iterate_training_adam,
        }
        trainable_parameters, trainable = iterate_training_fns[config.optimiser](
            trainable_parameters,
            trainable,
            topology,
            dataset,
            dataset_test,
            config,
            iteration,
        )

        summary_output = f"Summary for Iteration {iteration}".center(88, "=")
        summary_output += "\n"
        summary_output += "Parameterization".center(88, "=")
        summary_output += "\n"
        summary_output += "\n"
        for potential_type in trainable._param_types:
            summary_output += get_potential_comparison(
                old_force_field.potentials_by_type[potential_type],
                trainable.to_force_field(trainable_parameters).potentials_by_type[
                    potential_type
                ],
            )
            old_force_field.potentials_by_type[potential_type].parameters = copy.copy(
                trainable.to_force_field(trainable_parameters)
                .potentials_by_type[potential_type]
                .parameters
            )
        # Output the forcefield to a file
        off_force_field = convert_to_smirnoff(
            trainable.to_force_field(trainable_parameters), base=VdW_forcefield
        )
        off_force_field.to_file(config.output_dir / f"trained-{iteration}.offxml")
        # Generate the Energy Scatter Plot and summarize convergence
        energy_mean_new, energy_SD_new, forces_mean_new, forces_SD_new = write_scatter(
            dataset_test,
            trainable.to_force_field(trainable_parameters),
            topology,
            config.device_type,
            config.output_dir / f"trained-{iteration}.scat",
        )
        logger.info("")
        logger.info("")
        logger.info("Convergence".center(88, "="))
        logger.info("")
        logger.info(
            f"    Energy Error (Mean): {energy_mean:10.3e}->{energy_mean_new:10.3e} : Change = {energy_mean_new - energy_mean:10.3e}"
        )
        logger.info(
            f"                 (SD):   {energy_SD:10.3e}->{energy_SD_new:10.3e} : Change = {energy_SD_new - energy_SD:10.3e}"
        )
        logger.info(
            f"    Forces Error (Mean): {forces_mean:10.3e}->{forces_mean_new:10.3e} : Change = {forces_mean_new - forces_mean:10.3e}"
        )
        logger.info(
            f"                 (SD):   {forces_SD:10.3e}->{forces_SD_new:10.3e} : Change = {forces_SD_new - forces_SD:10.3e}"
        )
        logger.info("")
        energy_mean, energy_SD = energy_mean_new, energy_SD_new
        forces_mean, forces_SD = forces_mean_new, forces_SD_new

        # Get the next iteration of training data, unless the loop is finished
        if iteration < config.n_iterations:
            get_data_fn = config.run_md_fn if config.method != "data" else get_data_MMMD

            new_dataset = get_data_fn(
                mol,
                off_force_field,
                config.ml_potential,
                config.temperature,
                timestep_ps,
                config.n_train_snapshots_per_conformer,
                config.n_conformers,
                config.snapshot_interval,
                config.n_equilibration_steps,
                config.energy_upper_cutoff,
                config.energy_lower_cutoff,
                config.cluster_tolerance,
                config.cluster_parallel,
            )

            if config.memory:
                dataset = datasets.combine.concatenate_datasets([dataset, new_dataset])
            else:
                dataset = new_dataset

            with open("/dev/null", "w") as f:
                with redirect_stderr(f):
                    dataset.save_to_disk(config.output_dir / f"data_it_{iteration}")
