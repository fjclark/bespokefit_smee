"""Pydantic models which control/validate the settings."""

import warnings
from pathlib import Path
from typing import Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import data_maker
from .utils.typing import PathLike, TorchDevice

DEFAULT_CONFIG_PATH = Path("training_config.yaml")

METHOD_TO_GET_DATA_FN = {
    "MMMD": data_maker.get_data_MMMD,
    "MLMD": data_maker.get_data_MLMD,
    "cMMMD": data_maker.get_data_cMMMD,
}


class TrainingConfig(BaseModel):
    """Configuration for the training process."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    smiles: str = Field(..., description="SMILES string")
    output_dir: Path = Field(
        Path("."),
        description="Directory where the output files will be saved",
    )
    device_type: TorchDevice = Field(
        "cuda", description="Device type for training, either 'cpu' or 'cuda'"
    )
    method: Literal["MMMD", "MLMD", "cMMMD", "data"] = Field(
        "MMMD", description="Method for generating data"
    )
    n_epochs: int = Field(1000, description="Number of epochs in the ML fit")
    learning_rate: float = Field(0.002, description="Learning Rate in the ML fit")
    learning_rate_decay: float = Field(
        1.00, description="Learning Rate Decay. 0.99 is 1%, and 1.0 is no decay."
    )
    learning_rate_decay_step: int = Field(10, description="Learning Rate Decay Step")
    loss_force_weight: float = Field(
        1e5, description="Scaling Factor for the Force loss term"
    )
    initial_force_field: str = Field(
        "openff-2.2.1.offxml",
        description="The force field from which to start. This can be any"
        " OpenFF force field, or your own .offxml file.",
    )
    ml_potential: str = Field(
        "mace-off23-small",
        description="The machine learning potential to use for calculating energies and forces,"
        " and test MD trajectory. If the method is 'MLMD', this will also be used to generate the"
        "training data.",
    )
    n_train_snapshots: int = Field(
        1000, description="Number of MD snapshots required for the training set"
    )
    n_test_snapshots: int = Field(
        1000, description="Number of MD snapshots required for the test set"
    )
    n_conformers: int = Field(
        10, description="Number of conformers to generate, from which MD is started"
    )
    n_iterations: int = Field(
        10,
        description="Number of iterations of running MD, then training the FF to run",
    )
    snapshot_interval: int = Field(
        10, description="Number of Time Steps Between MD Snapshots"
    )
    n_equilibration_steps: int = Field(
        100, description="Number of time steps ignored from beginning of MD runs"
    )
    temperature: int = Field(500, description="Temperature to run MD at, in Kelvin")
    timestep: float = Field(1.0, description="MD timestep, in femtoseconds")
    energy_lower_cutoff: float = Field(
        1.0, description="Lower bound for the energy cutoff function"
    )
    energy_upper_cutoff: float = Field(
        10.0, description="Upper bound for the energy cutoff function"
    )
    cluster_tolerance: float = Field(
        0.075, description="Tolerance used in the RMSD clustering"
    )
    cluster_parallel: int = Field(
        1, description="MPI nodes used in the RMSD clustering"
    )
    data: str | None = Field(
        None,
        description="Location of pre-calculated data set. Must be None unless method == 'data'",
    )
    memory: bool = Field(False, description="Retain data upon iteration (Default)")
    linear_harmonics: bool = Field(
        True,
        description="Linearize the Harmonic potentials in the Force Field (Default)",
    )
    linear_torsions: bool = Field(
        False,
        description="Linearize the Torsion potentials in the Force Field (Default)",
    )
    use_modified_seminaro: bool = Field(
        True,
        description="Use modified Seminario method to initialize the Force Field",
    )
    modified_seminario_finite_step: float = Field(
        0.005291772,
        description="Finite step to calculate Hessian (Angstrom) in the modified Seminario method",
    )
    modified_seminario_tolerance: float = Field(
        0.0001,
        description="Tolerance for the geometry optimizer in the modified Seminario method",
    )
    modified_seminario_vib_scaling: float = Field(
        0.957,
        description="Vibrational scaling factor for the modified Seminario method",
    )

    @field_validator("device_type")
    @classmethod
    def validate_device_type(cls, value: TorchDevice) -> TorchDevice:
        """Ensure that the requested device type is available."""
        if value == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this system.")

        if value == "cpu":
            warnings.warn(
                "Using CPU for training. This may be slow. Consider using CUDA if available.",
                UserWarning,
                stacklevel=2,
            )

        return value

    @model_validator(mode="after")
    def validate_snapshots(self) -> "TrainingConfig":
        """Ensure that the number of snapshots is divisible by the number of conformers."""
        if self.n_train_snapshots % self.n_conformers != 0:
            raise ValueError(
                f"Number of training snapshots ({self.n_train_snapshots}) must be divisible by the number of conformers ({self.n_conformers})."
            )
        if self.n_test_snapshots % self.n_conformers != 0:
            raise ValueError(
                f"Number of test snapshots ({self.n_test_snapshots}) must be divisible by the number of conformers ({self.n_conformers})."
            )
        return self

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_type)

    @property
    def n_test_snapshots_per_conformer(self) -> int:
        """Return the number of test snapshots per conformer."""
        return self.n_test_snapshots // self.n_conformers

    @property
    def n_train_snapshots_per_conformer(self) -> int:
        """Return the number of training snapshots per conformer."""

        return self.n_train_snapshots // self.n_conformers

    # Validate data - this must be None unless method == 'data', in which case
    # it must be a valid path
    @model_validator(mode="after")
    def validate_data(self) -> "TrainingConfig":
        """Validate the data field based on the method."""
        if self.method == "data" and self.data is None:
            raise ValueError(
                "If method is 'data', the data field must be a valid path."
            )
        elif self.method != "data" and self.data is not None:
            raise ValueError("If method is not 'data', the data field must be None.")

        return self

    @property
    def run_md_fn(self) -> data_maker.GetDataFnType:
        """Return the function to get data based on the selected method."""
        if self.method == "data":
            raise ValueError(
                "The method is 'data' - no need to run MD. Use the data field to load data."
            )
        return METHOD_TO_GET_DATA_FN[self.method]

    @property
    def pretty_string(self) -> str:
        """Return a pretty string representation of the configuration."""
        lines = ["TrainingConfig Parameters".center(60, "=")]
        lines.append("")
        for field, value in self.model_dump().items():
            lines.append(f"{'':5}{field:<24}: {value}")
        lines.append("")
        return "\n".join(lines)

    @classmethod
    def from_yaml(cls, yaml_path: PathLike = DEFAULT_CONFIG_PATH) -> "TrainingConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r") as file:
            config_data = yaml.safe_load(file)
        return cls(**config_data)

    def to_yaml(self, yaml_path: PathLike = DEFAULT_CONFIG_PATH) -> None:
        """Save configuration to a YAML file."""
        data = self.model_dump()
        # Convert Path objects to strings for YAML serialisation
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
        with open(yaml_path, "w") as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
