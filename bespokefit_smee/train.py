"""Apply OpenFF parameters to molecule, cluster conformers by RMSD and train"""

import copy
import datetime
import pathlib
from contextlib import redirect_stderr

import datasets
import datasets.combine
import loguru
import openff.toolkit
import tensorboardX
import torch
from tqdm import tqdm

from .data_maker import get_data_cMMMD, get_data_MLMD, get_data_MMMD
from .loss_functions import prediction_loss
from .parameterizer import build_parameters, convert_to_smirnoff
from .settings import DEFAULT_CONFIG_PATH, TrainingConfig
from .writers import (
    get_potential_comparison,
    open_writer,
    write_metrics,
    write_scatter,
)

logger = loguru.logger


def train(config: TrainingConfig) -> None:
    """
    Train a bespoke force field according to the provided configuration.

    Parameters:
        config  : TrainingConfig
            Configuration object containing all the necessary parameters for training.
    """
    # Summarise input parameters
    logger.info(f"Training settings:\n{config.pretty_string}")

    # Save config to YAML file
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
        config.use_modified_seminaro,
        config.modified_seminario_finite_step / 10,  # Convert to nm
        config.modified_seminario_vib_scaling,
        config.modified_seminario_tolerance,
        device_type=config.device_type,
    )

    # Move to the requested device (not strictly necessary if on CPU)
    trainable_parameters = trainable.to_values().to((config.device_type))
    topology = topology.to(config.device_type)

    off_force_field = convert_to_smirnoff(
        trainable.to_force_field(trainable_parameters), base=VdW_forcefield
    )
    off_force_field.to_file("default.offxml")

    # get the inital training data
    if config.method == "DATA":
        dataset = datasets.Dataset.load_from_disk(config.data)
    elif config.method == "MMMD":
        dataset = get_data_MMMD(
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
        )
    elif config.method == "cMMMD":
        dataset = get_data_cMMMD(
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
    elif config.method == "MLMD":
        dataset = get_data_MLMD(
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
        )
    else:
        raise ValueError(f"Chosen method is not supported (method = {config.method})")
    with open("/dev/null", "w") as f:
        with redirect_stderr(f):
            dataset.save_to_disk("data_it_0")
    # Generate the test set
    logger.info("Generating Test Set with MLMD")
    dataset_test = get_data_MLMD(
        mol,
        off_force_field,
        config.ml_potential,
        config.temperature,
        timestep_ps,
        config.n_test_snapshots_per_conformer,
        config.n_conformers,
        config.snapshot_interval,
        config.n_equilibration_steps,
        config.energy_upper_cutoff,
        config.energy_lower_cutoff,
    )

    # Generate the Energy Scatter Plot
    energy_mean, energy_SD, forces_mean, forces_SD = write_scatter(
        dataset_test,
        trainable.to_force_field(trainable_parameters),
        topology,
        config.device_type,
        "default.scat",
    )
    for iteration in tqdm(
        range(config.n_iterations),
        leave=False,
        colour="magenta",
        desc="Iterating the Fit",
    ):
        # some book-keeping
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = pathlib.Path(f"{timestamp}")

        # run the ML training
        with open("training-" + str(iteration) + ".data", "w") as metrics_file:
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
            # some book-keeping and outputing
            loss_tst = prediction_loss(
                dataset_test,
                trainable.to_force_field(trainable_parameters),
                topology,
                config.loss_force_weight,
                config.device_type,
            )
            write_metrics(config.n_epochs, loss_trn, loss_tst, writer, metrics_file)
        summary_output = f"Summary for Iteration {iteration + 1}".center(88, "=")
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
        off_force_field.to_file("trained-" + str(iteration) + ".offxml")
        # Generate the Energy Scatter Plot and summarize convergence
        energy_mean_new, energy_SD_new, forces_mean_new, forces_SD_new = write_scatter(
            dataset_test,
            trainable.to_force_field(trainable_parameters),
            topology,
            config.device_type,
            "trained-" + str(iteration) + ".scat",
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
        # get the next iteration of training data, unless the loop is finished
        if iteration + 1 < config.n_iterations:
            if config.memory:
                if config.method == "cMMMD":
                    dataset = datasets.combine.concatenate_datasets(
                        [
                            dataset,
                            get_data_cMMMD(
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
                            ),
                        ]
                    )
                else:
                    dataset = datasets.combine.concatenate_datasets(
                        [
                            dataset,
                            get_data_MMMD(
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
                            ),
                        ]
                    )
            else:
                if config.method == "cMMMD":
                    dataset = get_data_cMMMD(
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
                else:
                    dataset = get_data_MMMD(
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
                    )
            with open("/dev/null", "w") as f:
                with redirect_stderr(f):
                    dataset.save_to_disk("data_it_" + str(iteration))
