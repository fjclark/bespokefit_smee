"""A CLI for bespokefit_smee."""

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import CliApp, CliPositionalArg, CliSubCommand

from .analysis import OutputData, plot_all
from .settings import DEFAULT_CONFIG_PATH, TrainingConfig
from .train import train

_DEFAULT_CONFIG_SMILES = "CHANGEME"


class TrainFromCli(TrainingConfig):
    """Run the training process with command line arguments."""

    def cli_cmd(self) -> None:
        print("Running bespokefit_smee CLI application")
        """Command to run the CLI application."""
        train(1, self)


class TrainFromYAML(BaseModel):
    """Run the training process with arguments read from a YAML file."""

    config_yaml: CliPositionalArg[Path] = Field(
        DEFAULT_CONFIG_PATH,
        description="Path to the YAML configuration file for training",
    )

    def cli_cmd(self) -> None:
        print(f"Running bespokefit_smee with settings from {self.config_yaml}")
        config = TrainingConfig.from_yaml(Path(self.config_yaml))
        if config.smiles == _DEFAULT_CONFIG_SMILES:
            raise ValueError(
                f"Please change the SMILES string in {self.config_yaml} to a valid value."
            )
        train(1, config)


class WriteDefaultYAML(BaseModel):
    """Write a default YAML file for the training configuration."""

    file_name: CliPositionalArg[Path] = Field(
        DEFAULT_CONFIG_PATH,
        description="The name of the default YAML configuration file to write",
    )

    def cli_cmd(self) -> None:
        print("Writing default YAML configuration file")
        TrainingConfig(smiles=_DEFAULT_CONFIG_SMILES).to_yaml(self.file_name)
        print(
            "Default YAML configuration file written to 'bespokefit_smee_default.yaml'"
        )


class Analyse(BaseModel):
    """Analyse the training data and results."""

    config_yaml: CliPositionalArg[Path] = Field(
        DEFAULT_CONFIG_PATH,
        description="Path to the YAML configuration used for training",
    )

    def cli_cmd(self) -> None:
        print(f"Analyzing training data with settings from {self.config_yaml}")
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
    print("Running app")
    CliApp.run(
        CLI,
    )
