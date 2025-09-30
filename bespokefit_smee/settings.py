"""Pydantic models which control/validate the settings."""

import warnings
from abc import ABC
from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch
import yaml
from openmm import unit
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic_units import OpenMMQuantity
from rdkit import Chem
from typing_extensions import Self

from . import mlp
from ._exceptions import InvalidSettingsError
from .outputs import OutputType, WorkflowPathManager
from .utils.typing import OptimiserName, PathLike, TorchDevice

_DEFAULT_SMILES_PLACEHOLDER = "CHANGEME"


class _DefaultSettings(BaseModel, ABC):
    """Default configuration for all models."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    def to_yaml(self, yaml_path: PathLike) -> None:
        """Save the settings to a YAML file"""
        data = self.model_dump(mode="json")
        with open(yaml_path, "w") as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_path: PathLike) -> Self:
        """Load settings from a YAML file"""
        with open(yaml_path, "r") as file:
            settings_data = yaml.safe_load(file)
        return cls(**settings_data)

    @property
    def output_types(self) -> set[OutputType]:
        """Return a set of expected output types for the function which
        implements this settings object. Subclasses should override this method."""
        return set()

    # @property
    # def output_types(self) -> set[OutputType]:
    #     """Return a set of expected output types for this settings object
    #     and all _DefaultSettings it contains."""
    #     outputs = set(self._output_types)
    #     for name, field in self.model_fields.items():
    #         if issubclass(field.annotation, _DefaultSettings):
    #             nested_settings = getattr(self, name)
    #             outputs.update(nested_settings.output_types)
    #     return outputs


class _SamplingSettingsBase(_DefaultSettings, ABC):
    """Settings for sampling (usually molecular dynamics)."""

    sampling_protocol: str = Field(
        ...,
        description="Type of sampling protocol. Each sampling settings subclass "
        "should set this to a unique value. This is used as a discriminator when "
        "loading from YAML.",
    )

    ml_potential: Literal[mlp.AvailableModels] = Field(
        "egret-1",
        description="The machine learning potential to use for calculating energies and forces of "
        " the snapshots. Note that this is not generally the potential used for sampling.",
    )

    timestep: OpenMMQuantity[unit.femtoseconds] = Field(  # type: ignore[type-arg]
        default=1 * unit.femtoseconds,
        description="MD timestep",
    )

    temperature: OpenMMQuantity[unit.kelvin] = Field(  # type: ignore[type-arg]
        default=500 * unit.kelvin,
        description="Temperature to run MD at",
    )

    snapshot_interval: OpenMMQuantity[unit.femtoseconds] = Field(  # type: ignore[type-arg]
        default=100 * unit.femtoseconds,
        description="Interval between saving snapshots during production sampling",
    )

    n_conformers: int = Field(
        10,
        description="The number of conformers to generate, from which sampling is started",
    )

    equilibration_sampling_time_per_conformer: OpenMMQuantity[unit.picoseconds] = Field(  # type: ignore[type-arg]
        default=0.1 * unit.picoseconds,
        description="Equilibration sampling time per conformer. No snapshots are saved during "
        "equilibration sampling. The total sampling time per conformer will be this plus "
        "the production_sampling_time_per_conformer.",
    )

    production_sampling_time_per_conformer: OpenMMQuantity[unit.picoseconds] = Field(  # type: ignore[type-arg]
        default=10 * unit.picoseconds,
        description="Production sampling time per conformer. The total sampling time per conformer "
        "will be this plus the equilibration_sampling_time_per_conformer.",
    )

    @property
    def equilibration_n_steps_per_conformer(self) -> int:
        return int(self.equilibration_sampling_time_per_conformer / self.timestep)

    @property
    def production_n_snapshots_per_conformer(self) -> int:
        return int(self.production_sampling_time_per_conformer / self.snapshot_interval)

    @property
    def production_n_steps_per_snapshot_per_conformer(self) -> int:
        return int(self.snapshot_interval / self.timestep)

    @property
    def output_types(self) -> set[OutputType]:
        return {OutputType.PDB_TRAJECTORY}

    @model_validator(mode="after")
    def validate_sampling_times(self) -> Self:
        """Ensure that the sampling times divide exactly by the timestep and (for production) the snapshot interval."""
        for time, name in [
            (
                self.equilibration_sampling_time_per_conformer,
                "equilibration_sampling_time_per_conformer",
            ),
            (
                self.production_sampling_time_per_conformer,
                "production_sampling_time_per_conformer",
            ),
        ]:
            n_steps = time / self.timestep
            if not n_steps.is_integer():
                raise InvalidSettingsError(
                    f"{name} ({time}) must be divisible by the timestep ({self.timestep})."
                )

        # Additionally check that production sampling time divides by snapshot interval
        time = self.production_sampling_time_per_conformer / self.snapshot_interval
        if not n_steps.is_integer():
            raise InvalidSettingsError(
                f"production_sampling_time_per_conformer ({time}) must be divisible by the snapshot_interval ({self.snapshot_interval})."
            )

        return self


class MMMDSamplingSettings(_SamplingSettingsBase):
    """Settings for molecular dynamics sampling using a molecular mechanics
    force field. This is initally the force field supplined in the parameterisation
    settings, but is updated as the bespoke force field is trained."""

    sampling_protocol: Literal["mm_md"] = Field(
        "mm_md", description="Sampling protocol to use."
    )


class MLMDSamplingSettings(_SamplingSettingsBase):
    """Settings for molecular dynamics sampling using a machine learning
    potential. This protocol uses the ML reference potential for sampling as
    well as for energy and force calculations."""

    sampling_protocol: Literal["ml_md"] = Field(
        "ml_md", description="Sampling protocol to use."
    )


class MMMDMetadynamicsSamplingSettings(_SamplingSettingsBase):
    """Settings for molecular dynamics sampling using a molecular mechanics
    force field with metadynamics. This is initally the force field supplined in the parameterisation
    settings, but is updated as the bespoke force field is trained."""

    sampling_protocol: Literal["mm_md_metadynamics"] = Field(
        "mm_md_metadynamics", description="Sampling protocol to use."
    )

    metadynamics_bias_factor: float = Field(
        10.0, description="Bias factor for well-tempered metadynamics"
    )

    bias_width: float = Field(np.pi / 10, description="Width of the bias (in radians)")

    bias_factor: float = Field(
        10.0,
        description="Bias factor for well-tempered metadynamics. Typical range: 5-20",
    )

    bias_height: OpenMMQuantity[unit.kilojoules_per_mole] = Field(  # type: ignore[type-arg]
        2.0 * unit.kilojoules_per_mole,
        description="Initial height of the bias",
    )

    bias_frequency: OpenMMQuantity[unit.picoseconds] = Field(  # type: ignore[type-arg]
        0.5 * unit.picoseconds,
        description="Frequency at which to add bias",
    )

    bias_save_frequency: OpenMMQuantity[unit.picoseconds] = Field(  # type: ignore[type-arg]
        1.0 * unit.picoseconds,
        description="Frequency at which to save the bias",
    )

    # Make sure that the frequency and save_frequency are multiples of the timestep
    @model_validator(mode="after")
    def validate_frequencies(self) -> Self:
        for freq, name in [
            (self.bias_frequency, "frequency"),
            (self.bias_save_frequency, "save_frequency"),
        ]:
            n_steps = freq / self.timestep
            if not n_steps.is_integer():
                raise InvalidSettingsError(
                    f"{name} ({freq}) must be divisible by the timestep ({self.timestep})."
                )

            # Make sure that the sampling time per conformer is a multiple of the save frequency
            n_saves = self.production_sampling_time_per_conformer / freq
            if not n_saves.is_integer():
                raise InvalidSettingsError(
                    f"production_sampling_time_per_conformer ({self.production_sampling_time_per_conformer}) must be divisible by the {name} ({freq})."
                )
        return self

    @property
    def n_steps_per_bias(self) -> int:
        return int(self.bias_frequency / self.timestep)

    @property
    def n_steps_per_bias_save(self) -> int:
        return int(self.bias_save_frequency / self.timestep)

    @property
    def output_types(self) -> set[OutputType]:
        return {OutputType.METADYNAMICS_BIAS, OutputType.PDB_TRAJECTORY}


SamplingSettings = Union[
    MMMDSamplingSettings,
    MLMDSamplingSettings,
    MMMDMetadynamicsSamplingSettings,
]


class TrainingSettings(_DefaultSettings):
    """Settings for the training process."""

    optimiser: OptimiserName = Field(
        "adam",
        description="Optimiser to use for the training. 'adam' is Adam, 'lm' is Levenberg-Marquardt",
    )
    test_data_path: Path | None = Field(
        None,
        description="Path to the test data. If None, the data will be generated using the ML potential.",
    )
    data: str | None = Field(
        None,
        description="Location of pre-calculated data set. Must be None unless method == 'data'",
    )
    n_epochs: int = Field(1000, description="Number of epochs in the ML fit")
    learning_rate: float = Field(0.01, description="Learning Rate in the ML fit")
    learning_rate_decay: float = Field(
        1.00, description="Learning Rate Decay. 0.99 is 1%, and 1.0 is no decay."
    )
    learning_rate_decay_step: int = Field(10, description="Learning Rate Decay Step")
    loss_force_weight: float = Field(
        0.1, description="Scaling Factor for the Force loss term"
    )
    energy_upper_cutoff: float = Field(
        10.0, description="Upper bound for the energy cutoff function"
    )

    @property
    def output_types(self) -> set[OutputType]:
        return {
            OutputType.TENSORBOARD,
            OutputType.TRAINING_METRICS,
        }


class MSMSettings(_DefaultSettings):
    """Settings for the modified Seminario method."""

    ml_potential: Literal[mlp.AvailableModels] = Field(
        "egret-1",
        description="The machine learning potential to use for calculating the Hessian matrix",
    )

    finite_step: OpenMMQuantity[unit.nanometers] = Field(  # type: ignore[type-arg]
        default=0.0005291772 * unit.nanometers,
        description="Finite step to calculate Hessian (Angstrom)",
    )

    tolerance: OpenMMQuantity[unit.kilocalories_per_mole / unit.angstrom] = Field(  # type: ignore[type-arg, valid-type]
        default=0.005291772 * unit.kilocalories_per_mole / unit.angstrom,
        description="Tolerance for the geometry optimizer",
    )
    vib_scaling: float = Field(
        0.957,
        description="Vibrational scaling factor",
    )


class ParameterisationSettings(_DefaultSettings):
    """Settings for the starting parameterisation."""

    smiles: str = Field(..., description="SMILES string")

    initial_force_field: str = Field(
        "openff_unconstrained-2.2.1.offxml",
        description="The force field from which to start. This can be any"
        " OpenFF force field, or your own .offxml file.",
    )

    linear_harmonics: bool = Field(
        True,
        description="Linearise the harmonic potentials in the Force Field (Default)",
    )
    linear_torsions: bool = Field(
        False,
        description="Linearise the torsion potentials in the Force Field (Default)",
    )
    msm_settings: MSMSettings | None = Field(
        default=None,
        description="Settings for the modified Seminario method. If None, the modified Seminario method "
        "will not be used to derive bonded parameters.",
    )

    expand_torsions: bool = Field(
        True,
        description="Whether to expand the torsion periodicities up to 4.",
    )

    # Make sure that the smiles isn't set to the placeholder value (as done in the CLI)
    @field_validator("smiles")
    def validate_smiles(cls, value: str) -> str:
        if Chem.MolFromSmiles(value) is None:
            raise ValueError(f"Invalid SMILES string: {value}")
        return value


class WorkflowSettings(_DefaultSettings):
    """Overall settings for the full fitting workflow."""

    output_dir: Path = Field(
        Path("."),
        description="Directory where the output files will be saved",
    )

    device_type: TorchDevice = Field(
        "cuda", description="Device type for training, either 'cpu' or 'cuda'"
    )

    n_iterations: int = Field(
        5,
        description="Number of iterations of sampling, then training the FF to run",
    )

    memory: bool = Field(
        False,
        description="Whether to append new training data to training data from the previous iterations,"
        " or overwrite it (False).",
    )

    parameterisation_settings: ParameterisationSettings = Field(
        description="Settings for the starting parameterisation",
    )

    training_sampling_settings: SamplingSettings = Field(
        default_factory=lambda: MMMDSamplingSettings(),
        description="Settings for sampling for generating the training data (usually molecular dynamics)",
        discriminator="sampling_protocol",
    )

    testing_sampling_settings: SamplingSettings = Field(
        default_factory=lambda: MLMDSamplingSettings(
            production_sampling_time_per_conformer=1 * unit.picoseconds
        ),
        description="Settings for sampling for generating the testing data (usually molecular dynamics)",
        discriminator="sampling_protocol",
    )

    training_settings: TrainingSettings = Field(
        default_factory=lambda: TrainingSettings(),
        description="Settings for the training process",
    )

    @field_validator("device_type")
    @classmethod
    def validate_device_type(cls, value: TorchDevice) -> TorchDevice:
        """Ensure that the requested device type is available."""
        if value == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this system.")

        if value == "cpu":
            warnings.warn(
                "Using CPU for training and sampling. This may be slow. Consider using CUDA if available.",
                UserWarning,
                stacklevel=2,
            )

        return value

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_type)

    def get_path_manager(self) -> WorkflowPathManager:
        """Get the output paths manager for this workflow settings object."""
        return WorkflowPathManager(
            output_dir=self.output_dir,
            n_iterations=self.n_iterations,
            training_settings=self.training_settings,
            training_sampling_settings=self.training_sampling_settings,
            testing_sampling_settings=self.testing_sampling_settings,
        )
