"""Unit tests for outputs module."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from presto.outputs import (
    PER_MOLECULE_OUTPUT_TYPES,
    OutputStage,
    OutputType,
    StageKind,
    WorkflowPathManager,
    WorkflowStatus,
    delete_path,
    get_mol_path,
)
from presto.settings import (
    MMMDMetadynamicsTorsionMinimisationSamplingSettings,
    MMMDSamplingSettings,
    TrainingSettings,
)


class TestGetMolPath:
    """Tests for the get_mol_path utility function."""

    def test_file_with_extension(self):
        """Test get_mol_path with a file that has an extension."""
        base_path = Path("/output/scatter.hdf5")
        result = get_mol_path(base_path, 0)
        assert result == Path("/output/scatter_mol0.hdf5")

        result = get_mol_path(base_path, 5)
        assert result == Path("/output/scatter_mol5.hdf5")

    def test_file_without_extension(self):
        """Test get_mol_path with a directory (no extension)."""
        base_path = Path("/output/energy_data")
        result = get_mol_path(base_path, 0)
        assert result == Path("/output/energy_data_mol0")

        result = get_mol_path(base_path, 3)
        assert result == Path("/output/energy_data_mol3")

    def test_nested_path(self):
        """Test get_mol_path with nested directory structure."""
        base_path = Path("/a/b/c/file.png")
        result = get_mol_path(base_path, 2)
        assert result == Path("/a/b/c/file_mol2.png")

    def test_relative_path(self):
        """Test get_mol_path with relative path."""
        base_path = Path("output/data.txt")
        result = get_mol_path(base_path, 1)
        assert result == Path("output/data_mol1.txt")


class TestOutputType:
    """Tests for OutputType enum."""

    def test_all_output_types_have_values(self):
        """Test that all output types have string values."""
        for output_type in OutputType:
            assert isinstance(output_type.value, str)
            assert len(output_type.value) > 0

    def test_workflow_settings_type(self):
        """Test workflow settings output type."""
        assert OutputType.WORKFLOW_SETTINGS.value == "workflow_settings.yaml"

    def test_offxml_type(self):
        """Test OFFXML output type."""
        assert OutputType.OFFXML.value == "bespoke_ff.offxml"


class TestStageKind:
    """Tests for StageKind enum."""

    def test_stage_kind_values(self):
        """Test that stage kinds have correct values."""
        assert StageKind.BASE.value == ""
        assert StageKind.TESTING.value == "test_data"
        assert StageKind.TRAINING.value == "training_iteration"
        assert StageKind.INITIAL_STATISTICS.value == "initial_statistics"
        assert StageKind.PLOTS.value == "plots"


class TestOutputStage:
    """Tests for OutputStage dataclass."""

    def test_stage_without_index(self):
        """Test stage without index."""
        stage = OutputStage(StageKind.BASE)
        assert stage.kind == StageKind.BASE
        assert stage.index is None
        assert str(stage) == ""

    def test_stage_with_index(self):
        """Test stage with index."""
        stage = OutputStage(StageKind.TRAINING, 1)
        assert stage.kind == StageKind.TRAINING
        assert stage.index == 1
        assert str(stage) == "training_iteration_1"

    def test_testing_stage(self):
        """Test testing stage."""
        stage = OutputStage(StageKind.TESTING)
        assert str(stage) == "test_data"

    def test_initial_statistics_stage(self):
        """Test initial statistics stage."""
        stage = OutputStage(StageKind.INITIAL_STATISTICS)
        assert str(stage) == "initial_statistics"

    def test_frozen_dataclass(self):
        """Test that OutputStage is frozen."""
        stage = OutputStage(StageKind.BASE)
        with pytest.raises(FrozenInstanceError):
            stage.kind = StageKind.TRAINING


class TestWorkflowPathManager:
    """Tests for WorkflowPathManager."""

    @pytest.fixture
    def path_manager(self, tmp_path):
        """Create a path manager for testing."""
        return WorkflowPathManager(
            output_dir=tmp_path,
            n_iterations=2,
            n_mols=2,
            training_settings=TrainingSettings(),
            training_sampling_settings=MMMDSamplingSettings(),
            testing_sampling_settings=MMMDSamplingSettings(),
        )

    def test_init(self, tmp_path):
        """Test initialization."""
        pm = WorkflowPathManager(output_dir=tmp_path)
        assert pm.output_dir == tmp_path
        assert pm.n_iterations == 1

    def test_status_clean_for_settings_only(self, tmp_path):
        """The preserved workflow settings file is not generated fit output."""
        path_manager = WorkflowPathManager(output_dir=tmp_path, n_iterations=2)
        assert path_manager.status == WorkflowStatus.CLEAN

        (tmp_path / OutputType.WORKFLOW_SETTINGS.value).write_text("settings")
        assert path_manager.status == WorkflowStatus.CLEAN

    @pytest.mark.parametrize(
        "stage_name",
        [
            "initial_statistics",
            "test_data",
            "plots",
            "training_iteration_1",
            "training_iteration_2",
        ],
    )
    def test_status_partial_for_any_predicted_stage(self, tmp_path, stage_name):
        """Every stage the settings predict marks a partial fit, even when empty."""
        path_manager = WorkflowPathManager(output_dir=tmp_path, n_iterations=2)
        (tmp_path / stage_name).mkdir()

        assert path_manager.status == WorkflowStatus.PARTIAL
        with pytest.raises(RuntimeError, match=r"partial.*presto clean"):
            path_manager.require_clean()

    def test_unpredicted_iterations_are_not_presto_output(self, tmp_path):
        """Iterations past `n_iterations` are neither owned nor deleted."""
        path_manager = WorkflowPathManager(output_dir=tmp_path, n_iterations=1)
        stale_iteration = tmp_path / "training_iteration_9"
        stale_iteration.mkdir()
        (stale_iteration / "trajectory_mol0.pdb").write_text("trajectory")

        assert path_manager.status == WorkflowStatus.CLEAN

        path_manager.clean()

        assert (stale_iteration / "trajectory_mol0.pdb").read_text() == "trajectory"

    @pytest.mark.parametrize(
        "file_name",
        [
            "initial_statistics",
            "test_data",
            "plots",
            "training_iteration_7",
            "training_iteration_notes.txt",
        ],
    )
    def test_status_and_clean_preserve_root_files(self, tmp_path, file_name):
        """Stage-like regular files are unrelated root files, not generated stages."""
        path_manager = WorkflowPathManager(output_dir=tmp_path, n_iterations=2)
        unrelated_file = tmp_path / file_name
        unrelated_file.write_text("keep me")

        assert path_manager.status == WorkflowStatus.CLEAN

        path_manager.clean()

        assert unrelated_file.read_text() == "keep me"

    @pytest.mark.parametrize("file_name", ["figure.png", "trajectory_mol0.pdb"])
    def test_clean_refuses_stage_directories_holding_foreign_files(
        self, tmp_path, file_name
    ):
        """A stage keeps only the outputs predicted for that stage, not any output.

        `trajectory_mol0.pdb` is a real output name, but sampling writes it to a
        training or testing stage, never to `plots/`, so there it is the user's.
        """
        path_manager = WorkflowPathManager(output_dir=tmp_path, n_iterations=2)
        user_file = tmp_path / "plots" / file_name
        user_file.parent.mkdir()
        user_file.write_text("keep me")
        stale_stage = tmp_path / "test_data"
        stale_stage.mkdir()

        with pytest.raises(RuntimeError, match=rf"{file_name}.*different output_dir"):
            path_manager.clean()

        assert user_file.read_text() == "keep me"
        assert stale_stage.exists()

    def test_clean_unlinks_stage_symlinks_without_deleting_targets(self, tmp_path):
        """Live and broken stage symlinks are owned, but their targets are not."""
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        external_stage = tmp_path / "external-stage"
        external_stage.mkdir()
        sentinel = external_stage / "sentinel.txt"
        sentinel.write_text("keep me")
        live_link = output_dir / "training_iteration_1"
        live_link.symlink_to(external_stage, target_is_directory=True)
        broken_link = output_dir / "plots"
        broken_link.symlink_to(tmp_path / "missing-stage", target_is_directory=True)
        path_manager = WorkflowPathManager(output_dir=output_dir)

        assert path_manager.status == WorkflowStatus.PARTIAL

        path_manager.clean()

        assert not live_link.is_symlink()
        assert not broken_link.is_symlink()
        assert sentinel.read_text() == "keep me"

    def test_status_complete_for_final_force_field(self, tmp_path):
        """The expected final force field marks fitting as complete."""
        path_manager = WorkflowPathManager(output_dir=tmp_path, n_iterations=2)
        final_force_field = tmp_path / "training_iteration_2" / "bespoke_ff.offxml"
        final_force_field.parent.mkdir()
        final_force_field.write_text("force field")

        assert path_manager.status == WorkflowStatus.COMPLETE
        with pytest.raises(RuntimeError, match=r"complete.*presto clean"):
            path_manager.require_clean()

    def test_outputs_by_stage_structure(self, path_manager):
        """Test that outputs_by_stage has correct structure."""
        outputs = path_manager.outputs_by_stage
        assert isinstance(outputs, dict)
        assert OutputStage(StageKind.BASE) in outputs
        assert OutputStage(StageKind.TESTING) in outputs
        assert OutputStage(StageKind.INITIAL_STATISTICS) in outputs
        assert OutputStage(StageKind.TRAINING, 1) in outputs
        assert OutputStage(StageKind.TRAINING, 2) in outputs
        assert OutputStage(StageKind.PLOTS) in outputs

    def test_outputs_by_stage_content(self, path_manager):
        """Test that outputs_by_stage has correct content."""
        outputs = path_manager.outputs_by_stage
        base_outputs = outputs[OutputStage(StageKind.BASE)]
        assert OutputType.WORKFLOW_SETTINGS in base_outputs

        testing_outputs = outputs[OutputStage(StageKind.TESTING)]
        assert OutputType.ENERGIES_AND_FORCES in testing_outputs
        assert OutputType.PDB_TRAJECTORY in testing_outputs

        training_outputs = outputs[OutputStage(StageKind.TRAINING, 1)]
        assert OutputType.OFFXML in training_outputs
        assert OutputType.SCATTER in training_outputs

    def test_get_stage_path(self, path_manager):
        """Test getting stage path."""
        base_path = path_manager.get_stage_path(OutputStage(StageKind.BASE))
        assert base_path == path_manager.output_dir / ""

        training_path = path_manager.get_stage_path(OutputStage(StageKind.TRAINING, 1))
        assert training_path == path_manager.output_dir / "training_iteration_1"

    def test_get_stage_path_unknown_stage(self, path_manager):
        """Test that unknown stage raises error."""
        # Create a stage that's not in outputs_by_stage
        invalid_stage = OutputStage(StageKind.TRAINING, 999)
        with pytest.raises(ValueError, match="Unknown stage"):
            path_manager.get_stage_path(invalid_stage)

    def test_mk_stage_dir(self, path_manager):
        """Test creating stage directory."""
        stage = OutputStage(StageKind.TRAINING, 1)
        path_manager.mk_stage_dir(stage)
        expected_path = path_manager.output_dir / "training_iteration_1"
        assert expected_path.exists()
        assert expected_path.is_dir()

    def test_get_output_path(self, path_manager):
        """Test getting output path."""
        stage = OutputStage(StageKind.TRAINING, 1)
        output_path = path_manager.get_output_path(stage, OutputType.OFFXML)
        expected = (
            path_manager.output_dir / "training_iteration_1" / "bespoke_ff.offxml"
        )
        assert output_path == expected

    def test_get_output_path_unknown_stage(self, path_manager):
        """Test that unknown stage raises error."""
        invalid_stage = OutputStage(StageKind.TRAINING, 999)
        with pytest.raises(ValueError, match="Unknown stage"):
            path_manager.get_output_path(invalid_stage, OutputType.OFFXML)

    def test_get_output_path_unexpected_output_type(self, path_manager):
        """Test that unexpected output type raises error."""
        stage = OutputStage(StageKind.BASE)
        with pytest.raises(ValueError, match="not expected in stage"):
            path_manager.get_output_path(stage, OutputType.OFFXML)

    def test_get_all_output_paths(self, path_manager):
        """Test getting all output paths."""
        all_paths = path_manager.get_all_output_paths(only_if_exists=False)
        assert isinstance(all_paths, dict)
        assert len(all_paths) > 0

        # Check structure
        for stage, outputs in all_paths.items():
            assert isinstance(stage, OutputStage)
            assert isinstance(outputs, dict)
            for output_type, path_or_paths in outputs.items():
                assert isinstance(output_type, OutputType)
                # Per-molecule types return lists, others return single Path
                if output_type in PER_MOLECULE_OUTPUT_TYPES:
                    assert isinstance(path_or_paths, list)
                    for path in path_or_paths:
                        assert isinstance(path, Path)
                else:
                    assert isinstance(path_or_paths, Path)

    def test_get_all_output_paths_only_existing(self, path_manager, tmp_path):
        """Test getting only existing output paths."""
        # Create some output files
        stage = OutputStage(StageKind.BASE)
        path_manager.mk_stage_dir(stage)
        output_path = path_manager.get_output_path(stage, OutputType.WORKFLOW_SETTINGS)
        output_path.write_text("test")

        all_paths = path_manager.get_all_output_paths(only_if_exists=True)
        assert OutputStage(StageKind.BASE) in all_paths
        assert OutputType.WORKFLOW_SETTINGS in all_paths[OutputStage(StageKind.BASE)]

    def test_get_all_output_paths_by_output_type(self, path_manager):
        """Test getting all output paths organized by output type."""
        paths_by_type = path_manager.get_all_output_paths_by_output_type(
            only_if_exists=False
        )
        assert isinstance(paths_by_type, dict)
        assert OutputType.OFFXML in paths_by_type
        assert isinstance(paths_by_type[OutputType.OFFXML], list)

        # Should have one OFFXML for initial stats + 2 for training iterations
        assert len(paths_by_type[OutputType.OFFXML]) == 3

    def test_clean_removes_files(self, path_manager):
        """Test that clean removes output files."""
        # Create some output files
        stage = OutputStage(StageKind.TRAINING, 1)
        path_manager.mk_stage_dir(stage)

        offxml_path = path_manager.get_output_path(stage, OutputType.OFFXML)
        offxml_path.write_text("test")

        # Create per-molecule scatter files
        scatter_paths = [
            path_manager.get_output_path_for_mol(stage, OutputType.SCATTER, mol_idx)
            for mol_idx in range(path_manager.n_mols)
        ]
        for scatter_path in scatter_paths:
            scatter_path.write_text("test")

        assert offxml_path.exists()
        for scatter_path in scatter_paths:
            assert scatter_path.exists()

        path_manager.clean()

        assert not offxml_path.exists()
        for scatter_path in scatter_paths:
            assert not scatter_path.exists()

    def test_clean_preserves_workflow_settings(self, path_manager):
        """Test that clean preserves workflow settings."""
        stage = OutputStage(StageKind.BASE)
        path_manager.mk_stage_dir(stage)

        settings_path = path_manager.get_output_path(
            stage, OutputType.WORKFLOW_SETTINGS
        )
        settings_path.write_text("test")

        path_manager.clean()

        assert settings_path.exists()

    def test_clean_removes_empty_directories(self, path_manager):
        """Test that clean removes empty stage directories."""
        stage = OutputStage(StageKind.TRAINING, 1)
        path_manager.mk_stage_dir(stage)
        stage_path = path_manager.get_stage_path(stage)

        assert stage_path.exists()

        path_manager.clean()

        assert not stage_path.exists()

    def test_clean_removes_every_sampling_output(self, tmp_path):
        """Every side output the sampling protocol produces is predicted, so removed."""
        sampling_settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings()
        path_manager = WorkflowPathManager(
            output_dir=tmp_path,
            n_mols=1,
            training_sampling_settings=sampling_settings,
        )
        stage = OutputStage(StageKind.TRAINING, 1)
        path_manager.mk_stage_dir(stage)
        stage_path = path_manager.get_stage_path(stage)

        paths = [
            path_manager.get_output_path_for_mol(stage, output_type, 0)
            for output_type in sampling_settings.output_types
        ]
        paths.append(
            path_manager.get_output_path_for_mol(
                stage, OutputType.ENERGIES_AND_FORCES, 0
            )
        )
        for path in paths:
            if path.suffix:
                path.write_text("sampling output")
            else:
                path.mkdir()

        path_manager.clean()

        assert all(not path.exists() for path in paths)
        assert not stage_path.exists()

    def test_clean_keeps_everything_outside_a_predicted_stage(self, tmp_path):
        """The settings yaml and unrelated root files survive a clean."""
        path_manager = WorkflowPathManager(output_dir=tmp_path, n_iterations=1)
        settings_path = tmp_path / OutputType.WORKFLOW_SETTINGS.value
        notes_path = tmp_path / "notes.txt"
        settings_path.write_text("settings")
        notes_path.write_text("keep me")
        path_manager.mk_stage_dir(OutputStage(StageKind.TRAINING, 1))

        path_manager.clean()

        assert path_manager.status == WorkflowStatus.CLEAN
        assert settings_path.exists()
        assert notes_path.read_text() == "keep me"


class TestDeletePath:
    """Tests for delete_path function."""

    def test_delete_file(self, tmp_path):
        """Test deleting a file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test")
        assert file_path.exists()

        delete_path(file_path)
        assert not file_path.exists()

    def test_delete_empty_directory_non_recursive(self, tmp_path):
        """Test deleting an empty directory non-recursively."""
        dir_path = tmp_path / "empty_dir"
        dir_path.mkdir()
        assert dir_path.exists()

        delete_path(dir_path, recursive=False)
        assert not dir_path.exists()

    def test_delete_non_empty_directory_non_recursive(self, tmp_path):
        """Test that non-empty directory is not deleted non-recursively."""
        dir_path = tmp_path / "non_empty_dir"
        dir_path.mkdir()
        (dir_path / "file.txt").write_text("test")

        with pytest.raises(OSError, match="Directory not empty"):
            delete_path(dir_path, recursive=False)

    def test_delete_directory_recursive(self, tmp_path):
        """Test deleting a directory recursively."""
        dir_path = tmp_path / "test_dir"
        dir_path.mkdir()
        (dir_path / "file1.txt").write_text("test")
        subdir = dir_path / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_text("test")

        delete_path(dir_path, recursive=True)
        assert not dir_path.exists()

    def test_delete_nonexistent_path(self, tmp_path):
        """Test that deleting nonexistent path doesn't raise error."""
        nonexistent = tmp_path / "nonexistent.txt"
        delete_path(nonexistent)  # Should not raise

    def test_delete_nested_structure(self, tmp_path):
        """Test deleting nested directory structure."""
        root = tmp_path / "root"
        root.mkdir()
        (root / "file1.txt").write_text("test")

        level1 = root / "level1"
        level1.mkdir()
        (level1 / "file2.txt").write_text("test")

        level2 = level1 / "level2"
        level2.mkdir()
        (level2 / "file3.txt").write_text("test")

        delete_path(root, recursive=True)
        assert not root.exists()


class TestWorkflowPathManagerMolPaths:
    """Tests for WorkflowPathManager per-molecule path methods."""

    @pytest.fixture
    def path_manager(self, tmp_path: Path) -> WorkflowPathManager:
        """Create a path manager for testing."""
        return WorkflowPathManager(
            output_dir=tmp_path,
            n_iterations=2,
            n_mols=3,
        )

    def test_get_output_path_for_mol_scatter(self, path_manager: WorkflowPathManager):
        """Test getting per-molecule scatter paths."""
        stage = OutputStage(StageKind.INITIAL_STATISTICS)

        for mol_idx in range(3):
            path = path_manager.get_output_path_for_mol(
                stage, OutputType.SCATTER, mol_idx
            )
            expected = (
                path_manager.output_dir
                / "initial_statistics"
                / f"energies_and_forces_mol{mol_idx}.hdf5"
            )
            assert path == expected

    def test_get_output_path_for_mol_directory_type(
        self, path_manager: WorkflowPathManager
    ):
        """Test getting per-molecule paths for directory-type outputs."""
        stage = OutputStage(StageKind.TESTING)

        for mol_idx in range(3):
            path = path_manager.get_output_path_for_mol(
                stage, OutputType.ENERGIES_AND_FORCES, mol_idx
            )
            expected = (
                path_manager.output_dir
                / "test_data"
                / f"energy_and_force_data_mol{mol_idx}"
            )
            assert path == expected

    def test_get_output_path_for_mol_invalid_output_type(
        self, path_manager: WorkflowPathManager
    ):
        """Test that non-per-molecule output types raise ValueError."""
        stage = OutputStage(StageKind.BASE)

        with pytest.raises(ValueError, match="not a per-molecule output type"):
            path_manager.get_output_path_for_mol(stage, OutputType.WORKFLOW_SETTINGS, 0)

    def test_get_output_path_for_mol_invalid_mol_idx(
        self, path_manager: WorkflowPathManager
    ):
        """Test that invalid mol_idx raises ValueError."""
        stage = OutputStage(StageKind.INITIAL_STATISTICS)

        with pytest.raises(ValueError, match=r"mol_idx .* is out of range"):
            path_manager.get_output_path_for_mol(stage, OutputType.SCATTER, 10)

        with pytest.raises(ValueError, match=r"mol_idx .* is out of range"):
            path_manager.get_output_path_for_mol(stage, OutputType.SCATTER, -1)

    def test_get_all_output_paths_includes_per_mol_paths(
        self, path_manager: WorkflowPathManager, tmp_path: Path
    ):
        """Test that get_all_output_paths includes per-molecule paths."""
        # Create some files to be found
        initial_stats_dir = tmp_path / "initial_statistics"
        initial_stats_dir.mkdir(parents=True, exist_ok=True)

        for mol_idx in range(3):
            (initial_stats_dir / f"energies_and_forces_mol{mol_idx}.hdf5").touch()

        all_paths = path_manager.get_all_output_paths(only_if_exists=True)

        stage = OutputStage(StageKind.INITIAL_STATISTICS)
        assert stage in all_paths
        assert OutputType.SCATTER in all_paths[stage]
        scatter_paths = all_paths[stage][OutputType.SCATTER]
        assert isinstance(scatter_paths, list)
        assert len(scatter_paths) == 3

    def test_clean_removes_per_mol_files(
        self, path_manager: WorkflowPathManager, tmp_path: Path
    ):
        """Test that clean removes per-molecule files."""
        # Create stage directory and some per-molecule files
        initial_stats_dir = tmp_path / "initial_statistics"
        initial_stats_dir.mkdir(parents=True, exist_ok=True)

        for mol_idx in range(3):
            (initial_stats_dir / f"energies_and_forces_mol{mol_idx}.hdf5").touch()

        # Verify files exist
        for mol_idx in range(3):
            assert (
                initial_stats_dir / f"energies_and_forces_mol{mol_idx}.hdf5"
            ).exists()

        # Clean
        path_manager.clean()

        # Verify files are deleted
        for mol_idx in range(3):
            assert not (
                initial_stats_dir / f"energies_and_forces_mol{mol_idx}.hdf5"
            ).exists()


class TestPerMoleculeOutputTypes:
    """Tests for PER_MOLECULE_OUTPUT_TYPES constant."""

    def test_per_molecule_types_defined(self):
        """Test that expected output types are marked as per-molecule."""
        expected_per_mol = {
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
        assert PER_MOLECULE_OUTPUT_TYPES == expected_per_mol

    def test_non_per_molecule_types(self):
        """Test that certain output types are NOT per-molecule."""
        non_per_mol = {
            OutputType.WORKFLOW_SETTINGS,
            OutputType.TENSORBOARD,
            OutputType.TRAINING_METRICS,
            OutputType.OFFXML,
            OutputType.LOSS_PLOT,
        }
        for output_type in non_per_mol:
            assert output_type not in PER_MOLECULE_OUTPUT_TYPES


class TestWorkflowPathManagerNMols:
    """Tests for n_mols parameter in WorkflowPathManager."""

    def test_n_mols_default(self, tmp_path: Path):
        """Test that n_mols defaults to 1."""
        pm = WorkflowPathManager(output_dir=tmp_path)
        assert pm.n_mols == 1

    def test_n_mols_custom(self, tmp_path: Path):
        """Test that n_mols can be set."""
        pm = WorkflowPathManager(output_dir=tmp_path, n_mols=5)
        assert pm.n_mols == 5


class TestGetAllOutputPathsByOutputTypeByMolecule:
    """Tests for get_all_output_paths_by_output_type_by_molecule method."""

    @pytest.fixture
    def path_manager(self, tmp_path: Path) -> WorkflowPathManager:
        """Create a path manager for testing."""
        return WorkflowPathManager(
            output_dir=tmp_path,
            n_iterations=2,
            n_mols=3,
        )

    def test_per_molecule_types_organized_by_mol_idx(
        self, path_manager: WorkflowPathManager, tmp_path: Path
    ):
        """Test that per-molecule output types are organized by molecule index."""
        # Create scatter files for multiple molecules and stages
        initial_stats_dir = tmp_path / "initial_statistics"
        initial_stats_dir.mkdir(parents=True, exist_ok=True)
        training_1_dir = tmp_path / "training_iteration_1"
        training_1_dir.mkdir(parents=True, exist_ok=True)
        training_2_dir = tmp_path / "training_iteration_2"
        training_2_dir.mkdir(parents=True, exist_ok=True)

        # Create scatter files for each molecule and stage
        for mol_idx in range(3):
            (initial_stats_dir / f"energies_and_forces_mol{mol_idx}.hdf5").touch()
            (training_1_dir / f"energies_and_forces_mol{mol_idx}.hdf5").touch()
            (training_2_dir / f"energies_and_forces_mol{mol_idx}.hdf5").touch()

        result = path_manager.get_all_output_paths_by_output_type_by_molecule(
            only_if_exists=True
        )

        # Check scatter paths are organized by molecule
        assert OutputType.SCATTER in result
        scatter_by_mol = result[OutputType.SCATTER]
        assert isinstance(scatter_by_mol, dict)

        # Each molecule should have 3 scatter files (initial + 2 training)
        for mol_idx in range(3):
            assert mol_idx in scatter_by_mol
            assert len(scatter_by_mol[mol_idx]) == 3

    def test_non_per_molecule_types_as_flat_list(
        self, path_manager: WorkflowPathManager, tmp_path: Path
    ):
        """Test that non-per-molecule types are returned as flat lists."""
        # Create OFFXML files (non-per-molecule type)
        initial_stats_dir = tmp_path / "initial_statistics"
        initial_stats_dir.mkdir(parents=True, exist_ok=True)
        training_1_dir = tmp_path / "training_iteration_1"
        training_1_dir.mkdir(parents=True, exist_ok=True)

        (initial_stats_dir / "bespoke_ff.offxml").touch()
        (training_1_dir / "bespoke_ff.offxml").touch()

        result = path_manager.get_all_output_paths_by_output_type_by_molecule(
            only_if_exists=True
        )

        # Check OFFXML paths are a flat list
        assert OutputType.OFFXML in result
        offxml_paths = result[OutputType.OFFXML]
        assert isinstance(offxml_paths, list)
        assert len(offxml_paths) == 2

    def test_extract_mol_idx_from_path(self, path_manager: WorkflowPathManager):
        """Test _extract_mol_idx_from_path helper method."""
        # File with extension
        path = Path("/output/scatter_mol2.hdf5")
        assert path_manager._extract_mol_idx_from_path(path) == 2

        # Directory (no extension)
        path = Path("/output/energy_data_mol5")
        assert path_manager._extract_mol_idx_from_path(path) == 5

        # Path without _mol (backward compatibility)
        path = Path("/output/scatter.hdf5")
        assert path_manager._extract_mol_idx_from_path(path) == 0

    def test_empty_result_when_no_files_exist(self, path_manager: WorkflowPathManager):
        """Test that empty dict is returned when no files exist."""
        result = path_manager.get_all_output_paths_by_output_type_by_molecule(
            only_if_exists=True
        )
        assert result == {}
