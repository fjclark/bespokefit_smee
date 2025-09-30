"""Implements the overall workflow for fitting a bespoke force field."""

import copy
import pathlib

import datasets
import loguru
from openff.toolkit import ForceField
from tqdm import tqdm

from bespokefit_smee.parameterise import convert_to_smirnoff

from .analyse import analyse_workflow
from .outputs import OutputStage, OutputType, StageKind
from .parameterise import parameterise
from .sample import _SAMPLING_FNS_REGISTRY, SampleFn
from .settings import WorkflowSettings
from .train import _TRAINING_FNS_REGISTRY
from .utils._suppress_output import suppress_unwanted_output
from .writers import write_scatter

logger = loguru.logger

suppress_unwanted_output()


def get_bespoke_force_field(
    settings: WorkflowSettings, write_settings: bool = True
) -> ForceField:
    """
    Fit a bespoke force field. This involves:

    - Parameterising a base force field for the target molecule
    - Generating training data (e.g. from high-temperature MD simulations)
    - Optimising the parameters of the force field to reproduce the training data
    - Validating the fitted force field against test data

    Parameters
    ----------
    settings : WorkflowSettings
        The workflow settings to use for fitting the force field.

    write_settings : bool, optional
        Whether to write the settings to a YAML file in the output directory, by default True.

    Returns
    -------
    ForceField
        The fitted bespoke force field.
    """
    suppress_unwanted_output()

    path_manager = settings.get_path_manager()
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

    # Parameterise the base force field
    # TODO: break this down and make the getting the trainable the responsibility of the
    # train module. Also process everything at the OFF FF level before converting to the
    # tensor FF (will be a bit of a pain to update to do this though...)
    off_mol, initial_off_ff, tensor_top, tensor_ff, trainable = parameterise(
        settings.parameterisation_settings, device=settings.device_type
    )
    trainable_parameters = trainable.to_values().to((settings.device))

    # Generate the test data
    stage = OutputStage(StageKind.TESTING)
    path_manager.mk_stage_dir(stage)
    test_sample_fn: SampleFn = _SAMPLING_FNS_REGISTRY[
        type(settings.testing_sampling_settings)
    ]
    logger.info("Generating test data")
    dataset_test = test_sample_fn(
        mol=off_mol,
        off_ff=initial_off_ff,
        device=settings.device,
        settings=settings.testing_sampling_settings,
        output_paths={
            output_type: path_manager.get_output_path(stage, output_type)
            for output_type in settings.testing_sampling_settings.output_types
        },
    )
    dataset_test.save_to_disk(
        path_manager.get_output_path(stage, OutputType.ENERGIES_AND_FORCES)
    )

    # Write out statistics on the initial force field
    stage = OutputStage(StageKind.INITIAL_STATISTICS)
    path_manager.mk_stage_dir(stage)
    energy_mean, energy_sd, forces_mean, forces_sd = write_scatter(
        dataset_test,
        tensor_ff,
        tensor_top,
        str(settings.device),
        path_manager.get_output_path(stage, OutputType.SCATTER),
    )
    logger.info(
        f"Initial force field statistics: Energy (Mean/SD): {energy_mean:.3e}/{energy_sd:.3e}, Forces (Mean/SD): {forces_mean:.3e}/{forces_sd:.3e}"
    )
    off_ff = convert_to_smirnoff(
        trainable.to_force_field(trainable_parameters), base=initial_off_ff
    )
    off_ff.to_file(path_manager.get_output_path(stage, OutputType.OFFXML))

    train_sample_fn = _SAMPLING_FNS_REGISTRY[type(settings.training_sampling_settings)]

    train_fn = _TRAINING_FNS_REGISTRY[settings.training_settings.optimiser]

    # Train the force field
    for iteration in tqdm(
        range(1, settings.n_iterations + 1),  # Start from 1 (0 is untrained)
        leave=False,
        colour="magenta",
        desc="Iterating the Fit",
    ):
        stage = OutputStage(StageKind.TRAINING, iteration)
        path_manager.mk_stage_dir(stage)
        dataset_train = None  # Only None for the first iteration

        dataset_train_new = train_sample_fn(
            mol=off_mol,
            off_ff=initial_off_ff,
            device=settings.device,
            settings=settings.training_sampling_settings,
            output_paths={
                output_type: path_manager.get_output_path(stage, output_type)
                for output_type in settings.training_sampling_settings.output_types
            },
        )

        # Update training dataset: concatenate if memory is enabled and not the first iteration
        should_concatenate = settings.memory and dataset_train is not None
        dataset_train = (
            datasets.combine.concatenate_datasets([dataset_train, dataset_train_new])
            if should_concatenate
            else dataset_train_new
        )
        dataset_train.save_to_disk(
            path_manager.get_output_path(stage, OutputType.ENERGIES_AND_FORCES)
        )

        train_output_paths = {
            output_type: path_manager.get_output_path(stage, output_type)
            for output_type in settings.training_settings.output_types
        }

        trainable_parameters, trainable = train_fn(
            trainable_parameters=trainable_parameters,
            trainable=trainable,
            topology=tensor_top,
            dataset=dataset_train,
            dataset_test=dataset_test,
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
        off_ff.to_file(path_manager.get_output_path(stage, OutputType.OFFXML))

        energy_mean_new, energy_sd_new, forces_mean_new, forces_sd_new = write_scatter(
            dataset_test,
            tensor_ff,
            tensor_top,
            str(settings.device),
            path_manager.get_output_path(stage, OutputType.SCATTER),
        )
        logger.info(
            f"Iteration {iteration} force field statistics: Energy (Mean/SD): {energy_mean:.3e}/{energy_sd:.3e}, Forces (Mean/SD): {forces_mean:.3e}/{forces_sd:.3e}"
        )
        logger.info(
            f"    Energy Error (Mean): {energy_mean:10.3e}->{energy_mean_new:10.3e} : Change = {energy_mean_new - energy_mean:10.3e}"
        )
        logger.info(
            f"                 (SD):   {energy_sd:10.3e}->{energy_sd_new:10.3e} : Change = {energy_sd_new - energy_sd:10.3e}"
        )
        logger.info(
            f"    Forces Error (Mean): {forces_mean:10.3e}->{forces_mean_new:10.3e} : Change = {forces_mean_new - forces_mean:10.3e}"
        )
        logger.info(
            f"                 (SD):   {forces_sd:10.3e}->{forces_sd_new:10.3e} : Change = {forces_sd_new - forces_sd:10.3e}"
        )
        energy_mean, energy_sd = energy_mean_new, energy_sd_new
        forces_mean, forces_sd = forces_mean_new, forces_sd_new

    # Plot
    analyse_workflow(settings)

    return off_ff
