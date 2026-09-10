"""Pydantic models which control/validate the settings."""

import warnings
from abc import ABC
from pathlib import Path
from typing import Any, Literal, Self, TypeVar

import numpy as np
import torch
import yaml
from descent.train import AttributeConfig, ParameterConfig
from loguru import logger
from openff.toolkit import Molecule
from openmm import unit
from packaging.version import Version
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_units import OpenMMQuantity

from . import __version__
from ._exceptions import InvalidSettingsError
from .find_torsions import (
    DEFAULT_TORSIONS_TO_EXCLUDE_SMARTS,
    DEFAULT_TORSIONS_TO_INCLUDE_SMARTS,
)
from .load_molecules import (
    MOLECULE_LOADERS,
    PROBLEMATIC_FUNCTIONAL_GROUP_WARNINGS,
    MoleculeInputType,
    find_problematic_functional_groups,
    load_conformers_for_molecule,
)
from .outputs import OutputType, WorkflowPathManager
from .utils.dicts import deep_update
from .utils.typing import (
    AllowedAttributeType,
    NonLinearValenceType,
    OptimiserName,
    PathLike,
    TorchDevice,
    ValenceType,
)

_DEFAULT_INPUT_PLACEHOLDER = "CHANGEME"
_RUNTIME_OBJECT_PLACEHOLDER = "__PRESTO_RUNTIME_OBJECT_PLACEHOLDER__"


def _replace_non_serializable(obj: dict[str, Any]) -> dict[str, Any]:
    """Recursively replace non-JSON-serializable values with the runtime placeholder."""
    return {k: _replace_value(v) for k, v in obj.items()}


def _replace_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _replace_non_serializable(value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return _RUNTIME_OBJECT_PLACEHOLDER


def _find_placeholder_paths(obj: Any, prefix: str = "") -> list[str]:
    """Return bracket-notation paths of all placeholder values in a (possibly nested) dict."""
    paths: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            paths.extend(_find_placeholder_paths(v, f"{prefix}[{k}]"))
    elif obj == _RUNTIME_OBJECT_PLACEHOLDER:
        paths.append(prefix)
    return paths


_DEFAULT_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    validate_assignment=True,
    arbitrary_types_allowed=True,
)


def _model_to_yaml(
    model: BaseModel, yaml_path: PathLike, overwrite: dict[str, Any] | None = None
) -> None:
    """Save the settings to a YAML file."""
    data = model.model_dump(mode="json")
    if overwrite:
        data = deep_update(data, overwrite)
    with open(yaml_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False, indent=4)


_T = TypeVar("_T", bound=BaseModel)


def _model_from_yaml(
    cls: type[_T], yaml_path: PathLike, overwrite: dict[str, Any] | None = None
) -> _T:
    """Load settings from a YAML file."""
    with open(yaml_path) as file:
        settings_data = yaml.safe_load(file) or {}
    if overwrite:
        settings_data = deep_update(settings_data, overwrite)
    return cls(**settings_data)


class _DefaultSettings(BaseModel, ABC):
    """Default configuration for all models."""

    model_config = _DEFAULT_MODEL_CONFIG

    def to_yaml(
        self, yaml_path: PathLike, overwrite: dict[str, Any] | None = None
    ) -> None:
        """Save the settings to a YAML file."""
        _model_to_yaml(self, yaml_path, overwrite=overwrite)

    @classmethod
    def from_yaml(
        cls, yaml_path: PathLike, overwrite: dict[str, Any] | None = None
    ) -> Self:
        """Load settings from a YAML file."""
        return _model_from_yaml(cls, yaml_path, overwrite=overwrite)

    @property
    def output_types(self) -> set[OutputType]:
        """Return the expected output types for the function implementing this settings object.

        Subclasses should override this method.
        """
        return set()


class MLPSettings(_DefaultSettings):
    """Settings for selecting and configuring the reference ML potential."""

    ml_potential: str = Field(
        "aimnet2",
        description="The OpenMM-ML potential identifier (for example `aimnet2`, "
        "`aceff-2.0`, or `ase`). Any model supported by your OpenMM-ML installation "
        "can be used.",
    )

    ml_potential_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments passed to openmmml.MLPotential(...) when "
        "creating the reference potential.",
    )

    ml_system_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments passed to MLPotential.createSystem(...). "
        "For ASE-backed runs, this can include keys such as `calculator`, `aseAtoms`, "
        "and `info`.",
    )

    @field_serializer("ml_system_kwargs")
    def serialize_ml_system_kwargs(self, value: dict[str, Any]) -> dict[str, Any]:
        """Replace non-serializable runtime objects (e.g. ASE calculators) with placeholders."""
        return _replace_non_serializable(value)

    @model_validator(mode="after")
    def validate_no_runtime_placeholders(self) -> Self:
        """Raise if ml_system_kwargs contains runtime placeholder values (i.e. was loaded from YAML without overwrite)."""
        placeholder_paths = _find_placeholder_paths(self.ml_system_kwargs)
        if placeholder_paths:
            raise InvalidSettingsError(
                f"ml_system_kwargs contains runtime-only placeholder values at keys: "
                f"{placeholder_paths}. Supply the actual objects via from_yaml(..., overwrite=...) "
                "before validation."
            )
        return self


def _validate_starting_conformers_path(value: Path | None) -> Path | None:
    """Validate an optional starting-conformers SDF path.

    Only checks the obvious, molecule-independent problems (missing file, wrong
    suffix). Whether the file actually contains conformers for the molecules being
    fitted can only be checked once the molecules are known — see
    ``WorkflowSettings._check_starting_conformers_match_molecules``.
    """
    if value is None:
        return value
    if value.suffix.lower() != ".sdf":
        raise InvalidSettingsError(
            f"starting_conformers must be an SDF file ending in .sdf: {value}"
        )
    if not value.exists():
        raise InvalidSettingsError(
            f"starting_conformers SDF file does not exist: {value}"
        )
    return value


class _SamplingSettingsBase(_DefaultSettings, ABC):
    """Settings for sampling (usually molecular dynamics)."""

    sampling_protocol: str = Field(
        ...,
        description="Type of sampling protocol. Each sampling settings subclass "
        "should set this to a unique value. This is used as a discriminator when "
        "loading from YAML.",
    )

    mlp_settings: MLPSettings = Field(
        default_factory=MLPSettings,
        description="Settings controlling the OpenMM-ML reference potential used for "
        "energy and force calculations.",
    )

    timestep: OpenMMQuantity[unit.femtoseconds] = Field(  # type: ignore[type-arg]
        default=1 * unit.femtoseconds,
        description="MD timestep (femtoseconds). Must divide evenly into the "
        "equilibration and production sampling times.",
    )

    temperature: OpenMMQuantity[unit.kelvin] = Field(  # type: ignore[type-arg]
        default=500 * unit.kelvin,
        description="Temperature to run MD at (kelvin). Defaults to 500 K to broaden "
        "the sampled conformer distribution beyond room temperature.",
    )

    snapshot_interval: OpenMMQuantity[unit.femtoseconds] = Field(  # type: ignore[type-arg]
        default=0.5 * unit.picoseconds,
        description="Interval between saving snapshots during production sampling",
    )

    n_conformers: int = Field(
        10,
        description="The number of conformers to generate, from which sampling is started. "
        "Ignored when `starting_conformers` is set.",
    )

    starting_conformers: Path | None = Field(
        None,
        description="Optional path to an SDF of starting conformers for this sampling "
        "stage. If set, sampling starts from every conformer in the file that matches "
        "the molecule (matched by graph, atom order aligned automatically) and "
        "`n_conformers` is ignored for this stage. If None (default), conformers are "
        "generated with ETKDG.",
    )

    @field_validator("starting_conformers")
    @classmethod
    def _validate_starting_conformers(cls, value: Path | None) -> Path | None:
        """Validate that any supplied starting-conformers path is an existing SDF."""
        return _validate_starting_conformers_path(value)

    equilibration_sampling_time_per_conformer: OpenMMQuantity[unit.picoseconds] = Field(  # type: ignore[type-arg]
        default=0.0 * unit.picoseconds,
        description="Equilibration sampling time per conformer. No snapshots are saved during "
        "equilibration sampling. The total sampling time per conformer will be this plus "
        "the production_sampling_time_per_conformer.",
    )

    production_sampling_time_per_conformer: OpenMMQuantity[unit.picoseconds] = Field(  # type: ignore[type-arg]
        default=100 * unit.picoseconds,
        description="Production sampling time per conformer. The total sampling time per conformer "
        "will be this plus the equilibration_sampling_time_per_conformer.",
    )

    loss_energy_weight: float = Field(
        1000.0,
        description="Scaling factor for the energy loss term (energies are in "
        "kcal/mol). The default (1000) is much larger than `loss_force_weight` (0.1) "
        "to balance the different units of energy and force contributions to the loss.",
    )

    loss_force_weight: float = Field(
        0.1,
        description="Scaling factor for the force loss term (forces are in "
        "kcal/mol/Å). See `loss_energy_weight` for context on the default ratio.",
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
        if not time.is_integer():
            raise InvalidSettingsError(
                f"production_sampling_time_per_conformer ({time}) must be divisible by the snapshot_interval ({self.snapshot_interval})."
            )

        return self


class MMMDSamplingSettings(_SamplingSettingsBase):
    """Settings for molecular dynamics sampling using a molecular mechanics force field.

    The force field is initially taken from the parameterisation settings, but is
    updated as the bespoke force field is trained.
    """

    sampling_protocol: Literal["mm_md"] = Field(
        "mm_md", description="Sampling protocol to use."
    )


class MLMDSamplingSettings(_SamplingSettingsBase):
    """Settings for molecular dynamics sampling using a machine learning potential.

    This protocol uses the ML reference potential for both sampling and
    energy/force calculations.
    """

    sampling_protocol: Literal["ml_md"] = Field(
        "ml_md", description="Sampling protocol to use."
    )


class MMMDMetadynamicsSamplingSettings(_SamplingSettingsBase):
    """Settings for molecular dynamics sampling using a molecular mechanics force field with metadynamics.

    The force field is initially taken from the parameterisation settings, but is
    updated as the bespoke force field is trained.
    """

    sampling_protocol: Literal["mm_md_metadynamics"] = Field(
        "mm_md_metadynamics", description="Sampling protocol to use."
    )

    bias_width: float = Field(np.pi / 10, description="Width of the bias (in radians)")

    bias_factor: float = Field(
        20.0,
        description="Bias factor for well-tempered metadynamics. Typical range: 5-20",
    )

    bias_height: OpenMMQuantity[unit.kilojoules_per_mole] = Field(  # type: ignore[type-arg]
        1.0 * unit.kilojoules_per_mole,
        description="Initial height of the Gaussian bias (kJ/mol). In well-tempered "
        "metadynamics this is scaled down over time according to `bias_factor`.",
    )

    bias_frequency: OpenMMQuantity[unit.picoseconds] = Field(  # type: ignore[type-arg]
        0.1 * unit.picoseconds,
        description="How often to add a Gaussian to the bias (picoseconds). Must "
        "divide evenly into the timestep.",
    )

    bias_save_frequency: OpenMMQuantity[unit.picoseconds] = Field(  # type: ignore[type-arg]
        10 * unit.picoseconds,
        description="How often to save the accumulated bias to disk (picoseconds).",
    )

    torsions_to_include_smarts: list[str] = Field(
        default_factory=lambda: DEFAULT_TORSIONS_TO_INCLUDE_SMARTS.copy(),
        description="SMARTS patterns for torsions to include in metadynamics biasing. "
        "Note that the RDKit default aromaticity model is used rather than OpenFF's default MDL model, as the "
        "RDKIT default gives more sane aromaticty perception. These should match the "
        "entire torsion (4 atoms), not just the rotatable bond. ",
    )

    torsions_to_exclude_smarts: list[str] = Field(
        default_factory=lambda: DEFAULT_TORSIONS_TO_EXCLUDE_SMARTS.copy(),
        description="SMARTS patterns for bonds to exclude from metadynamics biasing. Note that "
        "the RDKit default aromaticity model is used rather than OpenFF's default MDL model, as the "
        "RDKIT default gives more sane aromaticty perception. Matches are removed from the list of "
        "torsions matched by the include patterns. These should match only the rotatable bond "
        "(2 atoms), not the full torsion.",
    )

    # Make sure that the frequency and save_frequency are multiples of the timestep
    @model_validator(mode="after")
    def validate_frequencies(self) -> Self:
        """Validate that bias frequencies and save frequencies divide evenly into the sampling time."""
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
        """Number of simulation steps between each bias addition."""
        return int(self.bias_frequency / self.timestep)

    @property
    def n_steps_per_bias_save(self) -> int:
        """Number of simulation steps between each bias save."""
        return int(self.bias_save_frequency / self.timestep)

    @property
    def output_types(self) -> set[OutputType]:
        """Return the expected output types for this sampling protocol."""
        return {OutputType.METADYNAMICS_BIAS, OutputType.PDB_TRAJECTORY}


class MMMDMetadynamicsTorsionMinimisationSamplingSettings(
    MMMDMetadynamicsSamplingSettings
):
    """Settings for MM MD metadynamics sampling with additional torsion-restrained minimisation structures.

    Extends MMMDMetadynamicsSamplingSettings by generating additional training data
    from torsion-restrained minimisations.
    """

    sampling_protocol: Literal["mm_md_metadynamics_torsion_minimisation"] = Field(  # type: ignore[assignment]
        "mm_md_metadynamics_torsion_minimisation",
        description="Sampling protocol to use.",
    )

    # Settings for torsion-restrained minimisation
    ml_minimisation_steps: int = Field(
        10,
        description="Number of MLP minimisation steps with restrained torsions.",
    )

    mm_minimisation_steps: int = Field(
        10,
        description="Number of MM minimisation steps with restrained torsions.",
    )

    torsion_restraint_force_constant: OpenMMQuantity[  # type: ignore[type-arg, valid-type]
        unit.kilojoules_per_mole / unit.radian**2
    ] = Field(
        0.0 * unit.kilojoules_per_mole / unit.radian**2,
        description="Force constant for torsion restraints.",
    )

    # Loss weights for the torsion-minimised samples
    map_ml_coords_energy_to_mm_coords_energy: bool = Field(
        False,
        description="Whether to substitute the MLP energy for the MM-minimised coordinates with the "
        "MLP energy for the corresponding MLP-minimised coordinates.",
    )

    loss_energy_weight_mm_torsion_min: float = Field(
        1000.0,
        description="Scaling factor for the energy loss term for torsion-minimised samples, using "
        "MM minimisation. Note that the weights for the MMMD samples are controlled by the "
        "loss_energy_weight field.",
    )

    loss_force_weight_mm_torsion_min: float = Field(
        0.1,
        description="Scaling factor for the force loss term for torsion-minimised samples, using "
        "MM minimisation. Note that the weights for the MMMD samples are controlled by the "
        "loss_force_weight field.",
    )

    loss_energy_weight_ml_torsion_min: float = Field(
        1000.0,
        description="Scaling factor for the energy loss term for torsion-minimised samples, using "
        "MLP minimisation. Note that the weights for the MMMD samples are controlled by the "
        "loss_energy_weight field.",
    )

    loss_force_weight_ml_torsion_min: float = Field(
        0.1,
        description="Scaling factor for the force loss term for torsion-minimised samples, using "
        "MLP minimisation. Note that the weights for the MMMD samples are controlled by the "
        "loss_force_weight field.",
    )

    @property
    def output_types(self) -> set[OutputType]:
        """Return the expected output types for this sampling protocol."""
        return {
            OutputType.METADYNAMICS_BIAS,
            OutputType.PDB_TRAJECTORY,
            OutputType.ML_MINIMISED_PDB,
            OutputType.MM_MINIMISED_PDB,
        }


class PreComputedDatasetSettings(_DefaultSettings):
    """Settings for loading pre-computed datasets from disk.

    For single-molecule fits, provide a single Path.
    For multi-molecule fits, provide a list of Paths (one per molecule).
    """

    sampling_protocol: Literal["pre_computed"] = Field(
        "pre_computed", description="Sampling protocol identifier."
    )

    dataset_paths: list[Path] = Field(
        ...,
        description="Path(s) to pre-computed dataset(s) saved with dataset.save_to_disk(). "
        "For single-molecule fits, provide a single Path. "
        "For multi-molecule fits, provide a list of Paths (one per molecule in order).",
    )

    @field_validator("dataset_paths", mode="before")
    @classmethod
    def normalize_dataset_paths(cls, value: Path | list[Path]) -> list[Path]:
        """Normalize dataset_paths to always be a list internally."""
        if isinstance(value, (str, Path)):
            return [Path(value)]
        return [Path(p) for p in value]

    @property
    def output_types(self) -> set[OutputType]:
        """Pre-computed datasets don't produce any output files."""
        return set()


SamplingSettings = (
    MMMDSamplingSettings
    | MLMDSamplingSettings
    | MMMDMetadynamicsSamplingSettings
    | MMMDMetadynamicsTorsionMinimisationSamplingSettings
    | PreComputedDatasetSettings
)
"""Union type for all sampling settings. See the associated `sampling_protocol` field
in each class for the string identifier which should be supplied to
`training_sampling_settings` and `testing_sampling_settings` fields in
`WorkflowSettings`."""


def _get_default_regularised_parameters() -> dict[ValenceType, list[str]]:
    return {
        "ProperTorsions": ["k"],
        "ImproperTorsions": ["k"],
    }


class TrainingSettings(_DefaultSettings):
    """Settings for the training process."""

    optimiser: OptimiserName = Field(
        "adam",
        description="Optimiser to use for the training. 'adam' is Adam, 'lm' is Levenberg-Marquardt",
    )
    # Use AttributeConfigs to prevent the user passing exclude or include keys,
    # which should be set in the parameterisation settings because they decide
    # which tagged SMARTS are generated
    parameter_configs: dict[ValenceType, ParameterConfig] = Field(
        default_factory=lambda: {  # type: ignore[arg-type]
            "LinearBonds": ParameterConfig(
                cols=["k1", "k2"],
                scales={"k1": 0.0028, "k2": 0.0028},
                limits={"k1": (1e-8, None), "k2": (1e-8, None)},
                include=None,
                exclude=None,
            ),
            "LinearAngles": ParameterConfig(
                cols=["k1", "k2"],
                scales={"k1": 0.012, "k2": 0.011},
                limits={"k1": (1e-8, None), "k2": (1e-8, None)},
                include=None,
                exclude=None,
            ),
            "ProperTorsions": ParameterConfig(
                cols=["k"],
                scales={"k": 1.3},
                limits={"k": (None, None)},
                regularize={"k": 1.0},
                include=None,
                # Exclude linear torsions to avoid non-zero force constants which can
                # cause instabilities. Taken from https://github.com/openforcefield/openff-forcefields/blob/05f7ad0daad1ccdefdf931846fd13df863ab5c7d/openforcefields/offxml/openff-2.2.1.offxml#L326-L328
                exclude=[
                    {
                        "id": "[*:1]-[*:2]#[*:3]-[*:4]",
                        "mult": 0,
                        "associated_handler": "ProperTorsions",
                    },
                    {
                        "id": "[*:1]~[*:2]-[*:3]#[*:4]",
                        "mult": 0,
                        "associated_handler": "ProperTorsions",
                    },
                    {
                        "id": "[*:1]~[*:2]=[#6,#7,#16,#15;X2:3]=[*:4]",
                        "mult": 0,
                        "associated_handler": "ProperTorsions",
                    },
                ],
            ),
            "ImproperTorsions": ParameterConfig(
                cols=["k"],
                scales={"k": 0.12},
                limits={"k": (0, None)},
                regularize={"k": 1.0},
                include=None,
                exclude=None,
            ),
        },
        description="Configuration for the force field parameters to be trained.",
    )

    attribute_configs: dict[AllowedAttributeType, AttributeConfig] = Field(
        {},
        description="Configuration for the force field attributes to be trained. "
        "This allows 1-4 scaling for 'vdW' and 'Electrostatics' to be trained.",
    )

    n_epochs: int = Field(
        1000,
        description="Number of training epochs. The default (1000) is comfortably "
        "above the typical convergence point for the Adam optimiser on small "
        "molecules; reduce for quick iteration and raise if the loss has not "
        "flattened.",
    )
    learning_rate: float = Field(0.01, description="Learning Rate in the ML fit")
    learning_rate_decay: float = Field(
        1.00, description="Learning Rate Decay. 0.99 is 1%, and 1.0 is no decay."
    )
    learning_rate_decay_step: int = Field(10, description="Learning Rate Decay Step")
    regularisation_target: Literal["initial", "zero"] = Field(
        "initial",
        description="Target value to regularise parameters towards. 'initial' is the initial parameter value, "
        "'zero' is zero.",
    )

    @property
    def output_types(self) -> set[OutputType]:
        """Return the expected output types for the training protocol."""
        return {
            OutputType.TENSORBOARD,
            OutputType.TRAINING_METRICS,
        }


class OutlierFilterSettings(_DefaultSettings):
    """Settings for filtering outliers from datasets based on MM vs MLP differences.

    Outliers are identified by comparing MM and reference (typically MLP) energies
    and forces. Conformations where the absolute difference exceeds a threshold
    are removed.
    """

    energy_outlier_threshold: float | None = Field(
        2.0,
        description="Absolute threshold in kcal/mol/atom for energy outlier detection. "
        "Conformations where |energy_mm - energy_ref| / n_atoms (energies relative to median) "
        "exceeds this threshold will be removed. Set to None to disable energy-based filtering.",
    )

    force_outlier_threshold: float | None = Field(
        500.0,
        description="Absolute threshold in kcal/mol/Å for force outlier detection. "
        "Conformations where max |force_mm - force_ref| exceeds this threshold "
        "will be removed. Set to None to disable force-based filtering.",
    )

    min_conformations: int = Field(
        1,
        ge=1,
        description="Minimum number of conformations to keep per molecule. "
        "If filtering would remove too many conformations, an error is raised.",
    )


class TypeGenerationSettings(_DefaultSettings):
    """Settings for generating tagged SMARTS types for a given potential type."""

    max_extend_distance: int = Field(
        -1,
        description="Maximum number of bonds to extend from the atoms to which the potential is applied "
        "when generating tagged SMARTS patterns. A value of -1 means no limit.",
    )
    include: list[str] = Field(
        [],
        description="List of SMARTS present in the initial force field for which to generate new SMARTS "
        " patterns. This allows you to split specific types for reparameterisation. This is mutually exclusive "
        "with the exclude field.",
    )

    exclude: list[str] = Field(
        [],
        description="List of SMARTS patterns to exclude when generating tagged SMARTS types. If present, "
        " these patterns will remain the same as in the initial force field. This is mutually exclusive "
        "with the include field.",
    )

    @model_validator(mode="after")
    def validate_include_exclude(self) -> Self:
        """Ensure that only one of include or exclude is set."""
        if self.include and self.exclude:
            raise InvalidSettingsError(
                "Only one of include or exclude can be set in TypeGenerationSettings."
            )
        return self


class MSMSettings(_DefaultSettings):
    """Settings for the modified Seminario method (MSM).

    The MSM derives bond and angle force constants and equilibrium values from
    the molecular Hessian — here computed using the reference MLP. See
    https://doi.org/10.1021/acs.jctc.7b00785 for the algorithm.
    """

    mlp_settings: MLPSettings = Field(
        default_factory=MLPSettings,
        description="Settings controlling the OpenMM-ML reference potential used for "
        "Hessian calculations.",
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
        1.0, description="Vibrational scaling factor. Set as appropriate for your MLP."
    )

    n_conformers: int = Field(
        1,
        description="Number of conformers to generate and calculate MSM parameters for. "
        "The resulting bond and angle parameters will be averaged over all conformers. "
        "Ignored when `starting_conformers` is set.",
    )

    starting_conformers: Path | None = Field(
        None,
        description="Optional path to an SDF of starting conformers for the MSM Hessian "
        "calculation. If set, MSM parameters are calculated for every conformer in the "
        "file that matches the molecule (matched by graph, atom order aligned "
        "automatically) and averaged, and `n_conformers` is ignored. If None (default), "
        "conformers are generated with ETKDG.",
    )

    @field_validator("starting_conformers")
    @classmethod
    def _validate_starting_conformers(cls, value: Path | None) -> Path | None:
        """Validate that any supplied starting-conformers path is an existing SDF."""
        return _validate_starting_conformers_path(value)


class ParamSettings(_DefaultSettings):
    """Settings controlling the initial parameterisation."""

    molecule_input_type: MoleculeInputType = Field(
        "smiles",
        description="Input type for molecule loading.",
    )

    molecules: list[str] = Field(
        ...,
        description="Molecule input(s). Meaning depends on molecule_input_type.",
    )

    initial_force_field: str = Field(
        "openff_unconstrained-2.3.0.offxml",
        description="The force field from which to start. This can be any"
        " OpenFF force field, or your own .offxml file.",
    )

    expand_torsions: bool = Field(
        True,
        description="Whether to expand the torsion periodicities up to 4.",
    )

    linearise_harmonics: bool = Field(
        True,
        description="Linearise the harmonic potentials in the Force Field (Default)",
    )

    msm_settings: MSMSettings | None = Field(
        default_factory=lambda: MSMSettings(),
        description="Settings for the modified Seminario method to initialise force field parameters.",
    )

    type_generation_settings: dict[NonLinearValenceType, TypeGenerationSettings] = (
        Field(
            default_factory=lambda: {  # type: ignore[arg-type]
                "Bonds": TypeGenerationSettings(max_extend_distance=-1, exclude=[]),
                "Angles": TypeGenerationSettings(max_extend_distance=-1, exclude=[]),
                "ProperTorsions": TypeGenerationSettings(
                    max_extend_distance=-1,
                    exclude=[
                        "[*:1]-[*:2]#[*:3]-[*:4]",  # Linear torsions should be kept linear
                        "[*:1]~[*:2]-[*:3]#[*:4]",  # Linear torsions should be kept linear
                        "[*:1]~[*:2]=[#6,#7,#16,#15;X2:3]=[*:4]",  # Linear torsions should be kept linear
                    ],
                ),
                "ImproperTorsions": TypeGenerationSettings(
                    max_extend_distance=-1, exclude=[]
                ),
            },
            description="Settings for generating tagged SMARTS types for each valence type.",
        )
    )

    @field_validator("molecules", mode="before")
    @classmethod
    def normalize_input(cls, value: Any) -> list[str]:
        """Normalize molecule input to a unique, non-empty list of strings."""
        if isinstance(value, (str, Path)):
            normalized = [str(value)]
        elif isinstance(value, list):
            normalized = [str(v) for v in value]
        else:
            raise ValueError(
                "input must be a string/path or a list of string/path values"
            )

        if not normalized:
            raise ValueError("input list cannot be empty")

        if len(normalized) != len(set(normalized)):
            duplicates = [item for item in normalized if normalized.count(item) > 1]
            unique_duplicates = sorted(set(duplicates))
            raise ValueError(f"Duplicate inputs found: {unique_duplicates}")

        return normalized

    def _load_molecules(self) -> list[Molecule]:
        """Load and validate molecules from input on every instantiation/update."""
        if self.molecule_input_type not in MOLECULE_LOADERS:
            raise ValueError(f"Unsupported input_type: {self.molecule_input_type}")
        loader = MOLECULE_LOADERS[self.molecule_input_type]
        return [
            molecule
            for input_value in self.molecules
            for molecule in loader(input_value)
        ]

    @model_validator(mode="after")
    def _check_molecule_loading(self) -> Self:
        """Check that molecules can be loaded."""
        # It's a waste reloading every time, but this is pretty cheap,
        # and avoids issues with appending to `molecules` not-causing re-validation
        # if caching. Setting `molecules` to a tuple messes with the CLI.
        molecules = self._load_molecules()

        for smarts, descriptions in find_problematic_functional_groups(
            molecules
        ).items():
            molecule_lines = "\n".join(f"  - {item}" for item in descriptions)
            warnings.warn(
                f"Molecules matching known problematic SMARTS `{smarts}` were "
                f"found:\n{molecule_lines}\n"
                f"{PROBLEMATIC_FUNCTIONAL_GROUP_WARNINGS[smarts]}",
                UserWarning,
                stacklevel=2,
            )

        return self

    @property
    def openff_molecules(self) -> list[Molecule]:
        """Return the loaded OpenFF Molecule objects."""
        return self._load_molecules()


class WorkflowSettings(_DefaultSettings):
    """Overall settings for the full fitting workflow."""

    version: str = Field(
        __version__,
        description="Version of presto used to create these settings",
    )

    output_dir: Path = Field(
        Path("."),
        description="Directory where the output files will be saved",
    )

    device_type: TorchDevice = Field(
        "cuda",
        description="Device type for training and sampling, either 'cpu' or 'cuda'. "
        "Using 'cuda' requires an NVIDIA driver compatible with CUDA >= 12.9 "
        "(required by OpenMM 8.5's PythonForce). 'cpu' is supported but very slow.",
    )

    n_iterations: int = Field(
        2,
        ge=1,
        description="Number of (sample, train) iterations to run. Iteration 1 samples "
        "with the initial force field; later iterations sample with the bespoke force "
        "field produced by the previous iteration, which usually improves test loss.",
    )

    memory: bool = Field(
        False,
        description="If True, each iteration appends its newly sampled training data "
        "to the data from previous iterations (growing dataset). If False (default), "
        "each iteration replaces the previous training dataset. Enabling memory "
        "increases peak GPU memory usage with each iteration.",
    )

    n_sampling_processes: int = Field(
        1,
        ge=1,
        description="Number of spawned worker processes used to sample independent "
        "ligands.",
    )

    param_settings: ParamSettings = Field(
        description="Settings controlling the initial parameterisation",
    )

    training_sampling_settings: SamplingSettings = Field(
        default_factory=lambda: MMMDMetadynamicsTorsionMinimisationSamplingSettings(),
        description="Settings for sampling for generating the training data (usually molecular dynamics)",
        discriminator="sampling_protocol",
    )

    testing_sampling_settings: SamplingSettings = Field(
        default_factory=lambda: MLMDSamplingSettings(
            temperature=298 * unit.kelvin,
            snapshot_interval=20 * unit.femtoseconds,
            production_sampling_time_per_conformer=2 * unit.picoseconds,
        ),
        description="Settings for sampling for generating the testing data (usually molecular dynamics)",
        discriminator="sampling_protocol",
    )

    training_settings: TrainingSettings = Field(
        default_factory=lambda: TrainingSettings(),
        description="Settings for the training process",
    )

    outlier_filter_settings: OutlierFilterSettings | None = Field(
        default_factory=lambda: OutlierFilterSettings(),
        description="Settings for filtering outliers from training data. "
        "Set to None to disable outlier filtering.",
    )

    # Raise an error if the major and minor versions do not match
    # (don't care about patch version)
    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        """Validate version format and check compatibility."""
        try:
            parsed = Version(value)
        except Exception as e:
            raise ValueError(f"Invalid version format: {value}") from e

        actual_version = Version(__version__)

        # Warn the user if major or minor versions do not match
        if parsed.major != actual_version.major or parsed.minor != actual_version.minor:
            logger.warning(
                f"Version mismatch: settings version {value} may not be compatible with current version {__version__}."
            )

        return value

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

    # Validate that linearise_harmonics argument in parameterisation settings is consistent with the valence types
    # in the training settings
    @model_validator(mode="after")
    def validate_parameterisation_training_consistency(self) -> Self:
        """Validate that linearise_harmonics in parameterisation settings is consistent with the valence types in the training settings."""
        harmonics_linearised = self.param_settings.linearise_harmonics
        excluded_valence_types = (
            ("Bonds", "Angles")
            if harmonics_linearised
            else ("LinearBonds", "LinearAngles")
        )
        if any(
            valence_type in self.training_settings.parameter_configs
            for valence_type in excluded_valence_types
        ):
            raise InvalidSettingsError(
                f"ParamSettings.linearise_harmonics is {harmonics_linearised}, but TrainingSettings.parameter_configs "
                f"contains valence types that are inconsistent with this setting: {excluded_valence_types}. "
            )

        return self

    @model_validator(mode="after")
    def validate_starting_conformers_match_molecules(self) -> Self:
        """Fail fast if a configured starting-conformers SDF lacks a molecule being fitted.

        This runs before the (slow) parameterisation stage so a mismatch between the
        supplied conformers and the fitted molecules surfaces immediately rather than
        mid-run.
        """
        msm_settings = self.param_settings.msm_settings
        stages: list[tuple[str, Path | None]] = [
            (
                "training_sampling_settings",
                getattr(self.training_sampling_settings, "starting_conformers", None),
            ),
            (
                "testing_sampling_settings",
                getattr(self.testing_sampling_settings, "starting_conformers", None),
            ),
            (
                "param_settings.msm_settings",
                None if msm_settings is None else msm_settings.starting_conformers,
            ),
        ]

        configured = [(name, path) for name, path in stages if path is not None]
        if not configured:
            return self

        molecules = self.param_settings.openff_molecules
        for name, path in configured:
            for molecule in molecules:
                try:
                    load_conformers_for_molecule(molecule, path)
                except ValueError as exc:
                    raise InvalidSettingsError(
                        f"{name}.starting_conformers ({path}) is invalid: {exc}"
                    ) from exc

        return self

    @property
    def device(self) -> torch.device:
        """Return a torch.device corresponding to the configured device_type."""
        return torch.device(self.device_type)

    def get_path_manager(self) -> WorkflowPathManager:
        """Get the output paths manager for this workflow settings object."""
        # Get the number of molecules from the validated molecule list
        n_mols = len(self.param_settings.openff_molecules)
        return WorkflowPathManager(
            output_dir=self.output_dir,
            n_iterations=self.n_iterations,
            n_mols=n_mols,
            training_settings=self.training_settings,
            training_sampling_settings=self.training_sampling_settings,
            testing_sampling_settings=self.testing_sampling_settings,
        )
