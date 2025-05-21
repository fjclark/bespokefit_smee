"""Pydantic models which control/validate the settings."""

import warnings
from pathlib import Path

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .utils.typing import PathLike, TorchDevice

DEFAULT_CONFIG_PATH = Path("training_config.yaml")


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
    method: str = Field("MMMD", description="Method for generating data")
    N_epochs: int = Field(1000, description="Number of epochs in the ML fit")
    learning_rate: float = Field(0.1, description="Learning Rate in the ML fit")
    learning_rate_decay: float = Field(0.99, description="Learning Rate Decay")
    learning_rate_decay_step: int = Field(10, description="Learning Rate Decay Step")
    loss_force_weight: float = Field(
        1e5, description="Scaling Factor for the Force loss term"
    )
    force_field_init: str = Field(
        "openff-2.2.0.offxml", description="Starting guess force field"
    )
    MLMD_potential: str = Field(
        "mace-off23-small", description="Name of the MD potential used"
    )
    N_train: int = Field(1000, description="Number of datapoints in training sets")
    N_test: int = Field(1000, description="Number of datapoints in testing sets")
    N_conformers: int = Field(10, description="Number of Starting Conformers")
    N_iterations: int = Field(5, description="Number of ML Iterations Performed")
    MD_stepsize: int = Field(
        10, description="Number of Time Steps Between MD Snapshots"
    )
    MD_startup: int = Field(100, description="Number of Time Steps Ignored")
    MD_temperature: int = Field(500, description="Temperature in Kelvin")
    MD_dt: float = Field(1.0, description="MD Stepsize in femtoseconds")
    MD_energy_lower_cutoff: float = Field(
        1.0, description="Lower bound for the energy cutoff function"
    )
    MD_energy_upper_cutoff: float = Field(
        10.0, description="Upper bound for the energy cutoff function"
    )
    Cluster_tolerance: float = Field(
        0.075, description="Tolerance used in the RMSD clustering"
    )
    Cluster_Parallel: int = Field(
        1, description="MPI nodes used in the RMSD clustering"
    )
    data: str = Field("train_data", description="Location of pre-calculated data set")
    modSem_finite_step: float = Field(
        0.005291772, description="Finite Step to Calculate Hessian in Ang"
    )
    modSem_vib_scaling: float = Field(
        0.957, description="Vibrational Scaling Parameter"
    )
    modSem_tolerance: float = Field(
        0.0001, description="Tolerance for the geometry optimizer"
    )
    memory: bool = Field(False, description="Retain data upon iteration (Default)")
    linear_harmonics: bool = Field(
        True,
        description="Linearize the Harmonic potentials in the Force Field (Default)",
    )
    linear_torsions: bool = Field(
        True,
        description="Linearize the Torsion potentials in the Force Field (Default)",
    )
    modSem: bool = Field(
        True,
        description="Use mod-Seminario method to initialize the Force Field (Default)",
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
            yaml.dump(data, file, default_flow_style=False)
