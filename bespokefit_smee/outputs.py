"""Functionality for handling the outputs of a workflow."""

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .settings import SamplingSettings, TrainingSettings


class OutputType(Enum):
    """An enumeration of the different types of outputs produced by bespoke fitting functions"""

    WORKFLOW_SETTINGS = "workflow_settings.yaml"
    ENERGIES_AND_FORCES = "energy_and_force_data"
    TENSORBOARD = "tensorboard"
    TRAINING_METRICS = "metrics.txt"
    OFFXML = "bespoke_ff.offxml"
    SCATTER = "scatter.scat"
    PDB_TRAJECTORY = "trajectory.pdb"
    METADYNAMICS_BIAS = "metadynamics_bias"
    LOSS_PLOT = "loss.png"
    ERROR_PLOT = "error_distributions.png"


class StageKind(str, Enum):
    BASE = ""
    INITIAL_STATISTICS = "initial_statistics"
    TESTING = "test_data"
    TRAINING = "training_iteration"
    PLOTS = "plots"


@dataclass(frozen=True)
class OutputStage:
    kind: StageKind
    index: int | None = None

    def __str__(self) -> str:
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
    training_settings: "TrainingSettings | None" = None
    training_sampling_settings: "SamplingSettings | None" = None
    testing_sampling_settings: "SamplingSettings | None" = None

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
            OutputStage(StageKind.PLOTS): {OutputType.LOSS_PLOT, OutputType.ERROR_PLOT},
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
        """Get the path for an output type in a stage."""
        if stage not in self.outputs_by_stage:
            raise ValueError(f"Unknown stage: {stage}")
        if output_type not in self.outputs_by_stage.get(stage, set()):
            raise ValueError(f"Output type {output_type} not expected in stage {stage}")

        return self.get_stage_path(stage) / output_type.value

    def get_all_output_paths(
        self, only_if_exists: bool = True
    ) -> dict[OutputStage, dict[OutputType, Path]]:
        """Get all expected output paths organized by stage."""
        all_paths = {}

        for stage in self.outputs_by_stage:
            paths_for_stage = {}
            for output_type in self.outputs_by_stage.get(stage, set()):
                path = self.get_output_path(stage, output_type)
                if not only_if_exists or path.exists():
                    paths_for_stage[output_type] = path

            if paths_for_stage:
                all_paths[stage] = paths_for_stage

        return all_paths

    def get_all_output_paths_by_output_type(
        self, only_if_exists: bool = True
    ) -> dict[OutputType, list[Path]]:
        """Get all expected output paths organized by output type."""
        all_paths = self.get_all_output_paths(only_if_exists=only_if_exists)
        paths_by_output_type: dict[OutputType, list[Path]] = defaultdict(list)

        for _, paths in all_paths.items():
            for output_type, path in paths.items():
                paths_by_output_type[output_type].append(path)

        return paths_by_output_type

    def clean(self) -> None:
        """Remove all output files and empty stage directories."""

        # Delete all output files
        all_paths = self.get_all_output_paths(only_if_exists=True)

        for paths in all_paths.values():
            for output_type, path in paths.items():
                if output_type == OutputType.WORKFLOW_SETTINGS:
                    continue  # Don't delete workflow settings
                delete_path(path, recursive=True)

        # Remove empty stage directories
        for stage in self.outputs_by_stage.keys():
            if stage.kind == StageKind.BASE:
                continue
            delete_path(self.get_stage_path(stage), recursive=False)


def delete_path(path: Path, recursive: bool = False) -> None:
    """Delete an output file or directory if it exists. Deletes the entire contents of
    a directory.

    Parameters
    ----------
    path : Path
        The path to delete.

    recursive : bool, optional
        Whether to delete directories recursively, by default False. If False, only
        empty directories will be deleted.
    """
    if not path.exists():
        return

    if path.is_dir():
        if recursive:
            for child in path.iterdir():
                delete_path(child, recursive=True)
        path.rmdir()  # Will only remove if empty
    else:
        path.unlink()
