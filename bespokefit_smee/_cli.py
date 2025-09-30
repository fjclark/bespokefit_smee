"""A CLI for bespokefit_smee."""

from pathlib import Path

import loguru
from pydantic import BaseModel, Field
from pydantic_settings import CliApp, CliPositionalArg, CliSubCommand

from .analyse import analyse_workflow
from .outputs import OutputStage, OutputType, StageKind, WorkflowPathManager
from .settings import (
    _DEFAULT_SMILES_PLACEHOLDER,
    ParameterisationSettings,
    WorkflowSettings,
)
from .workflow import get_bespoke_force_field

logger = loguru.logger

_DEFAULT_WORKFLOW_SETTINGS_PATH = WorkflowPathManager(Path(".")).get_output_path(
    OutputStage(StageKind.BASE), OutputType.WORKFLOW_SETTINGS
)


class TrainFromCli(WorkflowSettings):
    """Run the training process with command line arguments."""

    def cli_cmd(self) -> None:
        get_bespoke_force_field(self, write_settings=True)


class TrainFromYAML(BaseModel):
    """Run the training process with arguments read from a YAML file."""

    settings_yaml: CliPositionalArg[Path] = Field(
        _DEFAULT_WORKFLOW_SETTINGS_PATH,
        description="Path to the YAML settings file for training",
    )

    def cli_cmd(self) -> None:
        logger.info(f"Running bespokefit_smee with settings from {self.settings_yaml}")
        settings = WorkflowSettings.from_yaml(Path(self.settings_yaml))
        if settings.parameterisation_settings.smiles == _DEFAULT_SMILES_PLACEHOLDER:
            raise ValueError(
                f"Please change the SMILES string in {self.settings_yaml} to a valid value."
            )
        # No need to write a settings file if we're reading from one already
        get_bespoke_force_field(settings, write_settings=False)


class WriteDefaultYAML(BaseModel):
    """Write a default YAML file for the training settings."""

    file_name: CliPositionalArg[Path] = Field(
        _DEFAULT_WORKFLOW_SETTINGS_PATH,
        description="The name of the default YAML settings file to write",
    )

    def cli_cmd(self) -> None:
        logger.info(f"Writing default YAML settings to {self.file_name}.")
        # Temporarily set a valid SMILES string to pass validation, then overwrite it
        # with a placeholder value before writing the file
        param_settings = ParameterisationSettings(smiles="O")
        # Bypass validation
        object.__setattr__(param_settings, "smiles", _DEFAULT_SMILES_PLACEHOLDER)
        WorkflowSettings(parameterisation_settings=param_settings).to_yaml(
            self.file_name
        )


class Clean(BaseModel):
    """Clean the output directory by removing generated files."""

    settings_yaml: CliPositionalArg[Path] = Field(
        _DEFAULT_WORKFLOW_SETTINGS_PATH,
        description="Path to the YAML settings used for training",
    )

    def cli_cmd(self) -> None:
        logger.info(
            f"Cleaning output directory with settings from {self.settings_yaml}"
        )
        settings = WorkflowSettings.from_yaml(Path(self.settings_yaml))
        settings.get_path_manager().clean()


class Analyse(BaseModel):
    """Analyse the training data and results."""

    settings_yaml: CliPositionalArg[Path] = Field(
        _DEFAULT_WORKFLOW_SETTINGS_PATH,
        description="Path to the YAML settings used for training",
    )

    def cli_cmd(self) -> None:
        logger.info(f"Analysing training data with settings from {self.settings_yaml}")
        training_settings = WorkflowSettings.from_yaml(Path(self.settings_yaml))
        analyse_workflow(training_settings)


class CLI(BaseModel):
    """Bespokefit_smee: parameterise a bespoke force field from high-temperature MD data."""

    train: CliSubCommand[TrainFromCli] = Field(
        description="Train a bespoke force field from high-temperature MD data",
    )

    train_from_yaml: CliSubCommand[TrainFromYAML] = Field(
        description="The same as 'train', but arguments are read from a YAML file",
    )

    write_default_yaml: CliSubCommand[WriteDefaultYAML] = Field(
        description="Write a default YAML settings file for the training process",
    )

    clean: CliSubCommand[Clean] = Field(
        description="Clean the output directory by removing generated files",
    )

    analyse: CliSubCommand[Analyse] = Field(
        description="Analyse the training data and results",
    )

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def run_cli() -> None:
    CliApp.run(
        CLI,
    )
