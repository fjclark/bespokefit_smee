"""Implements the overall workflow for fitting a bespoke force field."""

import copy
import pathlib

import datasets
import loguru
from descent.train import Trainable
from openff.toolkit import ForceField
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from presto.convert import convert_to_smirnoff

from .analyse import analyse_workflow
from .convert import parameterise
from .data_utils import filter_dataset_outliers
from .outputs import OutputStage, OutputType, StageKind
from .sampling_coordinator import sample_ligands
from .settings import WorkflowSettings
from .train import _TRAINING_FNS_REGISTRY
from .writers import write_scatter

logger = loguru.logger
console = Console()


def get_bespoke_force_field(
    settings: WorkflowSettings, write_settings: bool = True
) -> ForceField:
    """Fit a bespoke force field.

    This involves:

    - Parameterising a base force field for the target molecule and generating
      specific tagged SMARTS parameters
    - Generating training data (e.g. from high-temperature MD simulations)
    - Optimising the parameters of the force field to reproduce the training data
    - Validating the fitted force field against test data

    Parameters
    ----------
    settings : WorkflowSettings
        The workflow settings to use for fitting the force field.

    write_settings : bool, optional
        Whether to write the settings to a YAML file in the output directory, by default True.

    Returns:
    -------
    ForceField
        The fitted bespoke force field.
    """
    path_manager = settings.get_path_manager()
    # Refuse to run if generated output from an earlier fit is still present.
    path_manager.require_clean()
    stage = OutputStage(StageKind.BASE)
    path_manager.mk_stage_dir(stage)

    if write_settings:
        settings_output_path = path_manager.get_output_path(
            stage, OutputType.WORKFLOW_SETTINGS
        )
        logger.info(f"Writing workflow settings to {settings_output_path}.")
        # Copy the settings and change the output directory to be "." as we save
        # to the output directory already
        output_settings = copy.deepcopy(settings)
        output_settings.output_dir = pathlib.Path(".")
        output_settings.to_yaml(settings_output_path)

    # Parameterise the base force field for all molecules
    off_mols, initial_off_ff, tensor_tops, tensor_ff = parameterise(
        settings.param_settings, device=settings.device_type
    )

    pruned_parameter_configs = {
        p_type: p_config
        for p_type, p_config in settings.training_settings.parameter_configs.items()
        if p_type in tensor_ff.potentials_by_type
    }

    trainable = Trainable(
        tensor_ff,
        pruned_parameter_configs,
        settings.training_settings.attribute_configs,
    )

    trainable_parameters = trainable.to_values().to(settings.device)

    # Get a copy of the initial trainable parameters for regularisation
    initial_parameters = trainable_parameters.clone().detach()

    # Persist the initial force field before sampling so spawned workers
    # reconstruct their own OpenFF/OpenMM state from disk.
    initial_stage = OutputStage(StageKind.INITIAL_STATISTICS)
    path_manager.mk_stage_dir(initial_stage)
    initial_offxml_path = path_manager.get_output_path(initial_stage, OutputType.OFFXML)
    initial_off_ff.to_file(str(initial_offxml_path))

    # Generate the test data for all molecules
    stage = OutputStage(StageKind.TESTING)
    path_manager.mk_stage_dir(stage)
    logger.info("Generating test data")
    test_output_paths = {
        output_type: path_manager.get_output_path(stage, output_type)
        for output_type in settings.testing_sampling_settings.output_types
    }
    datasets_test = sample_ligands(
        mols=off_mols,
        offxml_path=initial_offxml_path,
        device_type=settings.device_type,
        sampling_settings=settings.testing_sampling_settings,
        output_paths=test_output_paths,
        dataset_output_paths=[
            path_manager.get_output_path_for_mol(
                stage, OutputType.ENERGIES_AND_FORCES, i
            )
            for i in range(len(off_mols))
        ],
        n_processes=settings.n_sampling_processes,
    )

    # Write out statistics on the initial force field
    stage = OutputStage(StageKind.INITIAL_STATISTICS)
    path_manager.mk_stage_dir(stage)

    # Write scatter plots for each molecule
    for mol_idx, (dataset_test, tensor_top) in enumerate(
        zip(datasets_test, tensor_tops, strict=True)
    ):
        scatter_path_mol = path_manager.get_output_path_for_mol(
            stage, OutputType.SCATTER, mol_idx
        )
        energy_mean, energy_sd, forces_mean, forces_sd = write_scatter(
            dataset_test,
            tensor_ff,
            tensor_top,
            settings.device,
            str(scatter_path_mol),
        )
        logger.info(
            f"Molecule {mol_idx} initial force field statistics: Energy (Mean/SD): {energy_mean:.3e}/{energy_sd:.3e} kcal/mol, Forces (Mean/SD): {forces_mean:.3e}/{forces_sd:.3e} kcal/mol/Å"
        )

    off_ff = convert_to_smirnoff(
        trainable.to_force_field(trainable_parameters), base=initial_off_ff
    )
    off_ff.to_file(str(path_manager.get_output_path(stage, OutputType.OFFXML)))

    train_fn = _TRAINING_FNS_REGISTRY[settings.training_settings.optimiser]

    # Train the force field
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )

    previous_datasets = None  # Only None for the first iteration

    def process_training_dataset(
        mol_idx: int, dataset: datasets.Dataset
    ) -> datasets.Dataset:
        if settings.outlier_filter_settings is not None:
            logger.info(
                f"Applying outlier filtering to molecule {mol_idx} training data"
            )
            dataset = filter_dataset_outliers(
                dataset=dataset,
                force_field=tensor_ff,
                topology=tensor_tops[mol_idx],
                settings=settings.outlier_filter_settings,
                device=settings.device,
            )
        if settings.memory and previous_datasets is not None:
            dataset = datasets.combine.concatenate_datasets(
                [previous_datasets[mol_idx], dataset]
            )
        return dataset

    with progress:
        for iteration in progress.track(
            range(1, settings.n_iterations + 1),  # Start from 1 (0 is untrained)
            description="Iterating the Fit",
        ):
            stage = OutputStage(StageKind.TRAINING, iteration)
            path_manager.mk_stage_dir(stage)

            sampling_output_paths = {
                output_type: path_manager.get_output_path(stage, output_type)
                for output_type in settings.training_sampling_settings.output_types
            }

            # Workers load the force field produced by the previous iteration.
            sampling_offxml_path = (
                initial_offxml_path
                if iteration == 1
                else path_manager.get_output_path(
                    OutputStage(StageKind.TRAINING, iteration - 1), OutputType.OFFXML
                )
            )
            datasets_train = sample_ligands(
                mols=off_mols,
                offxml_path=sampling_offxml_path,
                device_type=settings.device_type,
                sampling_settings=settings.training_sampling_settings,
                output_paths=sampling_output_paths,
                dataset_output_paths=[
                    path_manager.get_output_path_for_mol(
                        stage, OutputType.ENERGIES_AND_FORCES, i
                    )
                    for i in range(len(off_mols))
                ],
                n_processes=settings.n_sampling_processes,
                process_dataset=process_training_dataset,
            )

            train_output_paths = {
                output_type: path_manager.get_output_path(stage, output_type)
                for output_type in settings.training_settings.output_types
            }

            trainable_parameters, trainable = train_fn(
                trainable_parameters=trainable_parameters,
                initial_parameters=initial_parameters,
                trainable=trainable,
                topologies=tensor_tops,
                datasets=datasets_train,
                datasets_test=datasets_test,
                settings=settings.training_settings,
                output_paths=train_output_paths,
                device=settings.device,
            )

            for potential_type in trainable._param_types:
                tensor_ff.potentials_by_type[potential_type].parameters = copy.copy(
                    trainable.to_force_field(trainable_parameters)
                    .potentials_by_type[potential_type]
                    .parameters
                )

            off_ff = convert_to_smirnoff(
                trainable.to_force_field(trainable_parameters), base=initial_off_ff
            )
            off_ff.to_file(str(path_manager.get_output_path(stage, OutputType.OFFXML)))

            # Write scatter plots for each molecule
            for mol_idx, (dataset_test, tensor_top) in enumerate(
                zip(datasets_test, tensor_tops, strict=True)
            ):
                scatter_path_mol = path_manager.get_output_path_for_mol(
                    stage, OutputType.SCATTER, mol_idx
                )
                energy_mean_new, energy_sd_new, forces_mean_new, forces_sd_new = (
                    write_scatter(
                        dataset_test,
                        tensor_ff,
                        tensor_top,
                        settings.device,
                        str(scatter_path_mol),
                    )
                )
                logger.info(
                    f"Iteration {iteration} Molecule {mol_idx} force field statistics: Energy (Mean/SD): {energy_mean_new:.3e}/{energy_sd_new:.3e} kcal/mol, Forces (Mean/SD): {forces_mean_new:.3e}/{forces_sd_new:.3e} kcal/mol/Å"
                )

            previous_datasets = datasets_train  # Carry into the next iteration

    # Plot
    analyse_workflow(settings)

    return off_ff
