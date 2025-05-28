"""A CLI for bespokefit_smee."""

from pathlib import Path

import loguru
from pydantic import BaseModel, Field
from pydantic_settings import CliApp, CliPositionalArg, CliSubCommand

from .analysis import OutputData, plot_all
from .settings import DEFAULT_CONFIG_PATH, TrainingConfig
from .train import train

logger = loguru.logger

_DEFAULT_CONFIG_SMILES = "CHANGEME"


class TrainFromCli(TrainingConfig):
    """Run the training process with command line arguments."""

    def cli_cmd(self) -> None:
        train(self, write_config=True)


class TrainFromYAML(BaseModel):
    """Run the training process with arguments read from a YAML file."""

    config_yaml: CliPositionalArg[Path] = Field(
        DEFAULT_CONFIG_PATH,
        description="Path to the YAML configuration file for training",
    )

    def cli_cmd(self) -> None:
        logger.info(f"Running bespokefit_smee with settings from {self.config_yaml}")
        config = TrainingConfig.from_yaml(Path(self.config_yaml))
        if config.smiles == _DEFAULT_CONFIG_SMILES:
            raise ValueError(
                f"Please change the SMILES string in {self.config_yaml} to a valid value."
            )
        # No need to write a config file if we're reading from one already
        train(config, write_config=False)


class WriteDefaultYAML(BaseModel):
    """Write a default YAML file for the training configuration."""

    file_name: CliPositionalArg[Path] = Field(
        DEFAULT_CONFIG_PATH,
        description="The name of the default YAML configuration file to write",
    )

    def cli_cmd(self) -> None:
        logger.info(f"Writing default YAML configuration to {self.file_name}.")
        TrainingConfig(smiles=_DEFAULT_CONFIG_SMILES).to_yaml(self.file_name)


class Analyse(BaseModel):
    """Analyse the training data and results."""

    config_yaml: CliPositionalArg[Path] = Field(
        DEFAULT_CONFIG_PATH,
        description="Path to the YAML configuration used for training",
    )

    def cli_cmd(self) -> None:
        logger.info(f"Analyzing training data with settings from {self.config_yaml}")
        training_config = TrainingConfig.from_yaml(Path(self.config_yaml))
        output_data = OutputData(training_config)
        plot_all(output_data)


class CLI(BaseModel):
    """Bespokefit_smee: parameterise a bespoke force field from high-temperature MD data."""

    train: CliSubCommand[TrainFromCli] = Field(
        description="Train a bespoke force field from high-temperature MD data",
    )

    train_from_yaml: CliSubCommand[TrainFromYAML] = Field(
        description="The same as 'train', but arguments are read from a YAML file",
    )

    write_default_yaml: CliSubCommand[WriteDefaultYAML] = Field(
        description="Write a default YAML configuration file for the training process",
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
