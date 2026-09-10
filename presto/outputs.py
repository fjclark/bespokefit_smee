"""Functionality for handling the outputs of a workflow."""

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .settings import SamplingSettings, TrainingSettings


class OutputType(Enum):
    """An enumeration of the different types of outputs produced by bespoke fitting functions.

    Every run lays out files in the following stages under ``output_dir``:

    - ``workflow_settings.yaml`` at the top, recording the inputs.
    - ``test_data/`` containing the validation dataset.
    - ``initial_statistics/`` with the offxml and scatter data for the starting force field.
    - ``training_iteration_<n>/`` for each iteration, containing the bespoke offxml,
      sampled trajectories, energies/forces, and TensorBoard metrics.
    - ``plots/`` aggregating loss and parameter diagnostic plots across iterations.

    The string value of each member is the filename or directory name on disk.
    """

    WORKFLOW_SETTINGS = "workflow_settings.yaml"
    """The settings yaml file which is written if the user runs using `presto train`
    (rather than `presto train-from-yaml <settings file>`). This provides a record of the
    settings used and allows easy re-running of the workflow later."""

    ENERGIES_AND_FORCES = "energy_and_force_data"
    """Directory containing energies and forces data files in HDF5 format."""

    TENSORBOARD = "tensorboard"
    """Directory containing TensorBoard logs."""

    TRAINING_METRICS = "metrics.txt"
    """File containing training metrics."""

    OFFXML = "bespoke_ff.offxml"
    """The output OpenFF ForceField file containing the bespoke parameters.
    One bespoke FF file is produced per training iteration."""

    SCATTER = "energies_and_forces.hdf5"
    """HDF5 file containing scatter data for energies and forces."""

    PDB_TRAJECTORY = "trajectory.pdb"
    """PDB trajectory file containing structures sampled during sampling."""

    METADYNAMICS_BIAS = "metadynamics_bias"
    """Directory containing metadynamics bias files."""

    LOSS_PLOT = "loss.png"
    """Plot of training and validation loss over training epochs."""

    ERROR_PLOT = "error_distributions.png"
    """Plot of error distributions for energies and forces. These are with
    respect to the 'test' data."""

    CORRELATION_PLOT = "correlation.png"
    """Plot of predicted vs reference energies and forces. These are with respect
    to the 'test' data."""

    FORCE_ERROR_BY_ATOM_INDEX_PLOT = "force_error_by_atom_index.png"
    """Plot of force errors broken down by atom index. These are with respect to
    the 'test' data."""

    PARAMETER_VALUES_PLOT = "parameter_values.png"
    """Plot of parameter values before and after fitting. Note that the 'before' force field
    is the one used for the initial sampling, after the MSM step."""

    PARAMETER_DIFFERENCES_PLOT = "parameter_differences.png"
    """Plot of parameter differences (fitted - initial) after fitting. Note that the 'initial' force field
    is the one used for the initial sampling, after the MSM step."""

    # PDB outputs for torsion-minimised structures
    ML_MINIMISED_PDB = "ml_minimised.pdb"
    """PDB file containing structures minimised using the machine-learned potential."""

    MM_MINIMISED_PDB = "mm_minimised.pdb"
    """PDB file containing structures minimised using the molecular mechanics potential."""

    TORSION_SAMPLING_PLOT = "torsion_sampling.png"
    """Plot of dihedral angles for all rotatable torsions during trajectories."""


# Output types that are generated per-molecule (one file/directory per molecule)
PER_MOLECULE_OUTPUT_TYPES: set[OutputType] = {
    OutputType.ENERGIES_AND_FORCES,
    OutputType.SCATTER,
    OutputType.PDB_TRAJECTORY,
    OutputType.METADYNAMICS_BIAS,
    OutputType.ERROR_PLOT,
    OutputType.CORRELATION_PLOT,
    OutputType.FORCE_ERROR_BY_ATOM_INDEX_PLOT,
    OutputType.PARAMETER_VALUES_PLOT,
    OutputType.PARAMETER_DIFFERENCES_PLOT,
    OutputType.ML_MINIMISED_PDB,
    OutputType.MM_MINIMISED_PDB,
    OutputType.TORSION_SAMPLING_PLOT,
}


class StageKind(StrEnum):
    """Enumeration of output directory names for each stage of the workflow."""

    BASE = ""
    INITIAL_STATISTICS = "initial_statistics"
    TESTING = "test_data"
    TRAINING = "training_iteration"
    PLOTS = "plots"


class WorkflowStatus(StrEnum):
    """The state of generated output for a workflow fit."""

    CLEAN = "clean"
    PARTIAL = "partial"
    COMPLETE = "complete"


@dataclass(frozen=True)
class OutputStage:
    """A specific stage in the workflow output directory structure."""

    kind: StageKind
    index: int | None = None

    def __str__(self) -> str:
        """String representation."""
        return (
            f"{self.kind.value}_{self.index}"
            if self.index is not None
            else self.kind.value
        )


@dataclass
class WorkflowPathManager:
    """Manages paths for workflow outputs based on WorkflowSettings."""

    output_dir: Path
    n_iterations: int = 1
    n_mols: int = 1
    training_settings: "TrainingSettings | None" = None
    training_sampling_settings: "SamplingSettings | None" = None
    testing_sampling_settings: "SamplingSettings | None" = None

    def _generated_stages(self) -> list[OutputStage]:
        """Return every stage these settings predict whose directory exists."""
        return [
            stage
            for stage in self.outputs_by_stage
            if stage.kind is not StageKind.BASE
            and (
                (stage_path := self.get_stage_path(stage)).is_dir()
                or stage_path.is_symlink()
            )
        ]

    def _foreign_children(self, stage: OutputStage) -> list[Path]:
        """Return entries in a stage directory which these settings do not predict."""
        predicted = {
            path
            for paths in self.get_all_output_paths(only_if_exists=False)
            .get(stage, {})
            .values()
            for path in (paths if isinstance(paths, list) else [paths])
        }
        return [
            child
            for child in self.get_stage_path(stage).iterdir()
            if child not in predicted
        ]

    @property
    def final_force_field(self) -> Path:
        """The bespoke force field written by the last training iteration."""
        return self.get_output_path(
            OutputStage(StageKind.TRAINING, self.n_iterations), OutputType.OFFXML
        )

    @property
    def status(self) -> WorkflowStatus:
        """Return whether generated fit output is clean, partial, or complete."""
        # is_symlink covers a broken link, which exists() reports as missing.
        if self.final_force_field.exists() or self.final_force_field.is_symlink():
            return WorkflowStatus.COMPLETE
        if self._generated_stages():
            return WorkflowStatus.PARTIAL
        return WorkflowStatus.CLEAN

    def require_clean(self) -> None:
        """Raise if generated output must be cleaned before starting a fit."""
        if (status := self.status) is not WorkflowStatus.CLEAN:
            raise RuntimeError(
                f"The output directory {self.output_dir} contains a {status.value} "
                "Presto fit. Run `presto clean` or use a new output directory before "
                "starting another fit."
            )

    @property
    def outputs_by_stage(self) -> dict[OutputStage, set[OutputType]]:
        """Return a dictionary mapping each stage to expected output types."""
        outputs_by_stage: dict[OutputStage, set[OutputType]] = {
            OutputStage(StageKind.BASE): {OutputType.WORKFLOW_SETTINGS},
            OutputStage(StageKind.TESTING): (
                self.testing_sampling_settings.output_types
                if self.testing_sampling_settings
                else set()
            )
            | {OutputType.ENERGIES_AND_FORCES},
            OutputStage(StageKind.INITIAL_STATISTICS): {
                OutputType.OFFXML,
                OutputType.SCATTER,
            },
            **{
                OutputStage(StageKind.TRAINING, i): (
                    self.training_settings.output_types
                    if self.training_settings
                    else set()
                )
                | (
                    self.training_sampling_settings.output_types
                    if self.training_sampling_settings
                    else set()
                )
                | {
                    OutputType.ENERGIES_AND_FORCES,
                    OutputType.OFFXML,
                    OutputType.SCATTER,
                }
                for i in range(1, self.n_iterations + 1)
            },
            OutputStage(StageKind.PLOTS): {
                OutputType.LOSS_PLOT,
                OutputType.ERROR_PLOT,
                OutputType.CORRELATION_PLOT,
                OutputType.FORCE_ERROR_BY_ATOM_INDEX_PLOT,
                OutputType.PARAMETER_VALUES_PLOT,
                OutputType.PARAMETER_DIFFERENCES_PLOT,
                OutputType.TORSION_SAMPLING_PLOT,
            },
        }
        return outputs_by_stage

    def get_stage_path(self, stage: OutputStage) -> Path:
        """Get the directory path for a workflow stage."""
        if stage not in self.outputs_by_stage:
            raise ValueError(f"Unknown stage: {stage}")
        return self.output_dir / str(stage)

    def mk_stage_dir(self, stage: OutputStage) -> None:
        """Create the directory for a workflow stage."""
        path = self.get_stage_path(stage)
        path.mkdir(parents=True, exist_ok=True)

    def get_output_path(self, stage: OutputStage, output_type: OutputType) -> Path:
        """Get the path for an output type in a stage.

        Note: For per-molecule output types (those in PER_MOLECULE_OUTPUT_TYPES),
        use get_output_path_for_mol instead.
        """
        if stage not in self.outputs_by_stage:
            raise ValueError(f"Unknown stage: {stage}")
        if output_type not in self.outputs_by_stage.get(stage, set()):
            raise ValueError(f"Output type {output_type} not expected in stage {stage}")

        return self.get_stage_path(stage) / output_type.value

    def get_output_path_for_mol(
        self, stage: OutputStage, output_type: OutputType, mol_idx: int
    ) -> Path:
        """Get the path for a per-molecule output type in a stage.

        Parameters
        ----------
        stage : OutputStage
            The workflow stage.
        output_type : OutputType
            The type of output (must be a per-molecule output type).
        mol_idx : int
            The molecule index.

        Returns:
        -------
        Path
            The path for the per-molecule output.

        Raises:
        ------
        ValueError
            If the output type is not a per-molecule output type, or if mol_idx
            is out of range.
        """
        if output_type not in PER_MOLECULE_OUTPUT_TYPES:
            raise ValueError(
                f"Output type {output_type} is not a per-molecule output type. "
                f"Use get_output_path instead."
            )
        if mol_idx < 0 or mol_idx >= self.n_mols:
            raise ValueError(
                f"mol_idx {mol_idx} is out of range for n_mols={self.n_mols}"
            )

        base_path = self.get_output_path(stage, output_type)

        # For directory-type outputs (no suffix), just append _mol{mol_idx}
        return get_mol_path(base_path, mol_idx)

    def get_all_output_paths(
        self, only_if_exists: bool = True
    ) -> dict[OutputStage, dict[OutputType, Path | list[Path]]]:
        """Get all expected output paths organized by stage.

        For per-molecule output types, returns a list of paths (one per molecule).
        For other output types, returns a single path.
        """
        all_paths: dict[OutputStage, dict[OutputType, Path | list[Path]]] = {}

        for stage in self.outputs_by_stage:
            paths_for_stage: dict[OutputType, Path | list[Path]] = {}
            for output_type in self.outputs_by_stage.get(stage, set()):
                if output_type in PER_MOLECULE_OUTPUT_TYPES:
                    # Collect paths for all molecules
                    mol_paths = []
                    for mol_idx in range(self.n_mols):
                        path = self.get_output_path_for_mol(stage, output_type, mol_idx)
                        if not only_if_exists or path.exists():
                            mol_paths.append(path)
                    if mol_paths:
                        paths_for_stage[output_type] = mol_paths
                else:
                    path = self.get_output_path(stage, output_type)
                    if not only_if_exists or path.exists():
                        paths_for_stage[output_type] = path

            if paths_for_stage:
                all_paths[stage] = paths_for_stage

        return all_paths

    def get_all_output_paths_by_output_type(
        self, only_if_exists: bool = True
    ) -> dict[OutputType, list[Path]]:
        """Get all expected output paths organized by output type.

        Note: For per-molecule output types, paths from all molecules are flattened
        into a single list. Use get_all_output_paths_by_output_type_by_molecule
        for per-molecule organization.
        """
        all_paths = self.get_all_output_paths(only_if_exists=only_if_exists)
        paths_by_output_type: dict[OutputType, list[Path]] = defaultdict(list)

        for _, paths in all_paths.items():
            for output_type, path_or_paths in paths.items():
                if isinstance(path_or_paths, list):
                    paths_by_output_type[output_type].extend(path_or_paths)
                else:
                    paths_by_output_type[output_type].append(path_or_paths)

        return paths_by_output_type

    def get_all_output_paths_by_output_type_by_molecule(
        self, only_if_exists: bool = True
    ) -> dict[OutputType, dict[int, list[Path]] | list[Path]]:
        """Get all expected output paths organized by output type and molecule.

        For per-molecule output types, returns a dict mapping mol_idx to list of paths.
        For non-per-molecule output types, returns a flat list of paths.

        Parameters
        ----------
        only_if_exists : bool, optional
            If True, only return paths that exist on disk, by default True.

        Returns:
        -------
        dict[OutputType, dict[int, list[Path]] | list[Path]]
            A dictionary mapping output types to either:
            - For per-molecule types: dict mapping mol_idx -> list of paths (one per stage)
            - For other types: list of paths (one per stage)
        """
        all_paths = self.get_all_output_paths(only_if_exists=only_if_exists)
        result: dict[OutputType, dict[int, list[Path]] | list[Path]] = {}

        for _stage, paths in all_paths.items():
            for output_type, path_or_paths in paths.items():
                if output_type in PER_MOLECULE_OUTPUT_TYPES:
                    # Per-molecule type: organize by molecule index
                    if output_type not in result:
                        result[output_type] = {}
                    mol_dict = result[output_type]
                    assert isinstance(mol_dict, dict)

                    # path_or_paths is a list of paths for each molecule
                    if isinstance(path_or_paths, list):
                        for path in path_or_paths:
                            # Extract mol_idx from path
                            mol_idx = self._extract_mol_idx_from_path(path)
                            if mol_idx not in mol_dict:
                                mol_dict[mol_idx] = []
                            mol_dict[mol_idx].append(path)
                else:
                    # Non-per-molecule type: flat list
                    if output_type not in result:
                        result[output_type] = []
                    path_list = result[output_type]
                    assert isinstance(path_list, list)
                    if isinstance(path_or_paths, list):
                        path_list.extend(path_or_paths)
                    else:
                        path_list.append(path_or_paths)

        return result

    def _extract_mol_idx_from_path(self, path: Path) -> int:
        """Extract molecule index from a per-molecule path.

        Parameters
        ----------
        path : Path
            A path with the _mol{idx} naming convention.

        Returns:
        -------
        int
            The molecule index.
        """
        stem = path.stem if path.suffix else path.name
        if "_mol" in stem:
            mol_idx_str = stem.split("_mol")[-1]
            return int(mol_idx_str)
        return 0  # Default for backward compatibility

    def clean(self) -> None:
        """Remove every stage directory these settings predict.

        Stages beyond the predicted ones, such as an iteration past
        ``n_iterations``, are left alone, as is anything outside a stage directory.

        Raises:
        ------
        RuntimeError
            If a predicted stage directory holds a path these settings do not
            predict. Nothing is deleted in that case.
        """
        stages = self._generated_stages()
        for stage in stages:
            stage_path = self.get_stage_path(stage)
            if stage_path.is_symlink() or not (
                foreign := self._foreign_children(stage)
            ):
                continue
            raise RuntimeError(
                f"{stage_path} contains files presto did not generate "
                f"({', '.join(child.name for child in foreign[:3])}). Delete it "
                "yourself or use a different output_dir."
            )
        for stage in stages:
            delete_path(self.get_stage_path(stage), recursive=True)


def delete_path(path: Path, recursive: bool = False) -> None:
    """Delete an output file or directory if it exists.

    Deletes the entire contents of a directory.

    Parameters
    ----------
    path : Path
        The path to delete.

    recursive : bool, optional
        Whether to delete directories recursively, by default False. If False, only
        empty directories will be deleted.
    """
    if not path.exists() and not path.is_symlink():
        return

    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        if recursive:
            for child in path.iterdir():
                delete_path(child, recursive=True)
        path.rmdir()  # Will only remove if empty
    else:
        path.unlink()


def get_mol_path(base_path: Path, mol_idx: int) -> Path:
    """Get a molecule-specific path from a base path.

    This function applies the standard naming convention for per-molecule outputs,
    where the molecule index is appended to the filename/directory name.

    Parameters
    ----------
    base_path : Path
        The base path (e.g., from get_output_path or output_paths dict).
    mol_idx : int
        The molecule index.

    Returns:
    -------
    Path
        The path with the molecule index appended.

    Examples:
    --------
    >>> get_mol_path(Path("output/scatter.hdf5"), 0)
    PosixPath('output/scatter_mol0.hdf5')
    >>> get_mol_path(Path("output/energy_data"), 1)
    PosixPath('output/energy_data_mol1')
    """
    if not base_path.suffix:
        # Directory-type output
        return base_path.parent / f"{base_path.name}_mol{mol_idx}"
    else:
        # File-type output
        return base_path.parent / f"{base_path.stem}_mol{mol_idx}{base_path.suffix}"
