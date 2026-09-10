"""Unit tests for settings module."""

from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st
from openff.toolkit import Molecule
from openmm import unit as omm_unit
from pydantic import ValidationError
from rdkit import Chem

from presto import __version__
from presto.settings import (
    _DEFAULT_INPUT_PLACEHOLDER,
    _RUNTIME_OBJECT_PLACEHOLDER,
    MLMDSamplingSettings,
    MLPSettings,
    MMMDMetadynamicsSamplingSettings,
    MMMDMetadynamicsTorsionMinimisationSamplingSettings,
    MMMDSamplingSettings,
    MSMSettings,
    OutlierFilterSettings,
    ParamSettings,
    TrainingSettings,
    TypeGenerationSettings,
    WorkflowSettings,
)


class TestSamplingSettingsBase:
    """Tests for sampling settings base class."""

    @pytest.fixture
    def valid_mm_md_settings(self):
        """Test that default values are set correctly."""
        return MMMDSamplingSettings()

    def test_default_values(self, valid_mm_md_settings):
        """Test that default values are set correctly."""
        assert valid_mm_md_settings.mlp_settings.ml_potential == "aimnet2"
        assert valid_mm_md_settings.timestep.value_in_unit(omm_unit.femtoseconds) == 1.0
        assert valid_mm_md_settings.temperature.value_in_unit(omm_unit.kelvin) == 500.0
        assert valid_mm_md_settings.n_conformers == 10

    def test_equilibration_n_steps_per_conformer(self, valid_mm_md_settings):
        """Test calculation of equilibration steps."""
        expected = int(
            valid_mm_md_settings.equilibration_sampling_time_per_conformer
            / valid_mm_md_settings.timestep
        )
        assert valid_mm_md_settings.equilibration_n_steps_per_conformer == expected

    def test_production_n_snapshots_per_conformer(self, valid_mm_md_settings):
        """Test calculation of production snapshots."""
        expected = int(
            valid_mm_md_settings.production_sampling_time_per_conformer
            / valid_mm_md_settings.snapshot_interval
        )
        assert valid_mm_md_settings.production_n_snapshots_per_conformer == expected

    def test_production_n_steps_per_snapshot_per_conformer(self, valid_mm_md_settings):
        """Test calculation of production steps per snapshot."""
        expected = int(
            valid_mm_md_settings.snapshot_interval / valid_mm_md_settings.timestep
        )
        assert (
            valid_mm_md_settings.production_n_steps_per_snapshot_per_conformer
            == expected
        )

    def test_invalid_equilibration_time_not_divisible_by_timestep(self):
        """Test that invalid equilibration time raises error."""
        with pytest.raises(ValidationError, match="must be divisible by the timestep"):
            MMMDSamplingSettings(
                timestep=1.5 * omm_unit.femtoseconds,
                equilibration_sampling_time_per_conformer=1.0 * omm_unit.picoseconds,
            )

    def test_invalid_production_time_not_divisible_by_timestep(self):
        """Test that invalid production time raises error."""
        with pytest.raises(ValidationError, match="must be divisible by the timestep"):
            MMMDSamplingSettings(
                timestep=1.5 * omm_unit.femtoseconds,
                production_sampling_time_per_conformer=1.0 * omm_unit.picoseconds,
            )

    def test_output_types(self, valid_mm_md_settings):
        """Test that output types are defined."""
        from presto.outputs import OutputType

        assert OutputType.PDB_TRAJECTORY in valid_mm_md_settings.output_types

    def test_ase_does_not_require_special_fields(self):
        """Test ASE settings can be configured only through ml_system_kwargs."""
        settings = MLMDSamplingSettings(mlp_settings=MLPSettings(ml_potential="ase"))
        assert settings.mlp_settings.ml_potential == "ase"

    def test_runtime_object_yaml_placeholder_round_trip(self, tmp_path):
        """Test runtime objects are written as placeholders and must be overridden on load."""
        yaml_path = tmp_path / "ase_settings.yaml"
        settings = MLMDSamplingSettings(
            mlp_settings=MLPSettings(
                ml_potential="ase",
                ml_system_kwargs={"calculator": object()},
            )
        )
        settings.to_yaml(yaml_path)

        assert _RUNTIME_OBJECT_PLACEHOLDER in yaml_path.read_text()

        with pytest.raises(ValidationError, match="runtime-only placeholder values"):
            MLMDSamplingSettings.from_yaml(yaml_path)

        loaded = MLMDSamplingSettings.from_yaml(
            yaml_path,
            overwrite={"mlp_settings": {"ml_system_kwargs": {"calculator": object()}}},
        )
        assert loaded.mlp_settings.ml_potential == "ase"

    def test_runtime_placeholder_supports_nested_ml_system_kwargs(self, tmp_path):
        """Test nested ml_system_kwargs values are preserved unless runtime-only."""
        yaml_path = tmp_path / "runtime_placeholder_nested.yaml"
        settings = MLMDSamplingSettings(
            mlp_settings=MLPSettings(
                ml_potential="ase",
                ml_system_kwargs={"info": {"charge": 1, "calculator_obj": object()}},
            )
        )
        settings.to_yaml(yaml_path)

        text = yaml_path.read_text()
        assert _RUNTIME_OBJECT_PLACEHOLDER in text

        with pytest.raises(ValidationError, match=r"\[info\]\[calculator_obj\]"):
            MLMDSamplingSettings.from_yaml(yaml_path)

        loaded = MLMDSamplingSettings.from_yaml(
            yaml_path,
            overwrite={
                "mlp_settings": {
                    "ml_system_kwargs": {
                        "info": {"charge": 1, "calculator_obj": object()}
                    },
                }
            },
        )
        assert loaded.mlp_settings.ml_system_kwargs["info"]["charge"] == 1

    def test_yaml_round_trip_keeps_serializable_nested_ml_system_kwargs(self, tmp_path):
        """Test serializable nested ml_system_kwargs values survive YAML round-trip."""
        yaml_path = tmp_path / "runtime_placeholder_serializable.yaml"
        settings = MLMDSamplingSettings(
            mlp_settings=MLPSettings(
                ml_potential="ase",
                ml_system_kwargs={"info": {"charge": 1, "foo": "bar"}},
            )
        )
        settings.to_yaml(yaml_path)

        assert _RUNTIME_OBJECT_PLACEHOLDER not in yaml_path.read_text()
        loaded = MLMDSamplingSettings.from_yaml(yaml_path)
        assert loaded.mlp_settings.ml_system_kwargs["info"] == {
            "charge": 1,
            "foo": "bar",
        }


class TestMMMDSamplingSettings:
    """Tests for MM MD sampling settings."""

    def test_sampling_protocol_is_mm_md(self):
        """Test that sampling protocol is set correctly."""
        settings = MMMDSamplingSettings()
        assert settings.sampling_protocol == "mm_md"

    def test_to_yaml_and_from_yaml(self, tmp_path):
        """Test YAML serialization round-trip."""
        settings = MMMDSamplingSettings(
            n_conformers=5, temperature=600 * omm_unit.kelvin
        )
        yaml_path = tmp_path / "settings.yaml"
        settings.to_yaml(yaml_path)

        loaded = MMMDSamplingSettings.from_yaml(yaml_path)
        assert loaded.n_conformers == 5
        assert loaded.temperature.value_in_unit(omm_unit.kelvin) == 600.0


class TestMLMDSamplingSettings:
    """Tests for ML MD sampling settings."""

    def test_sampling_protocol_is_ml_md(self):
        """Test that sampling protocol is set correctly."""
        settings = MLMDSamplingSettings()
        assert settings.sampling_protocol == "ml_md"


class TestMMMDMetadynamicsSamplingSettings:
    """Tests for MM MD metadynamics sampling settings."""

    def test_sampling_protocol_is_mm_md_metadynamics(self):
        """Test that sampling protocol is set correctly."""
        settings = MMMDMetadynamicsSamplingSettings()
        assert settings.sampling_protocol == "mm_md_metadynamics"

    def test_default_metadynamics_parameters(self):
        """Test that default metadynamics parameters are set."""
        settings = MMMDMetadynamicsSamplingSettings()
        assert settings.bias_factor == 20.0
        assert settings.bias_width == np.pi / 10
        assert settings.bias_height.value_in_unit(omm_unit.kilojoules_per_mole) == 1.0

    def test_n_steps_per_bias(self):
        """Test calculation of steps per bias."""
        settings = MMMDMetadynamicsSamplingSettings(
            timestep=1.0 * omm_unit.femtoseconds,
            bias_frequency=0.5 * omm_unit.picoseconds,
        )
        assert settings.n_steps_per_bias == 500

    def test_n_steps_per_bias_save(self):
        """Test calculation of steps per bias save."""
        settings = MMMDMetadynamicsSamplingSettings(
            timestep=1.0 * omm_unit.femtoseconds,
            bias_save_frequency=1.0 * omm_unit.picoseconds,
        )
        assert settings.n_steps_per_bias_save == 1000

    def test_invalid_bias_frequency_not_divisible_by_timestep(self):
        """Test that invalid bias frequency raises error."""
        with pytest.raises(ValidationError, match="must be divisible by the timestep"):
            MMMDMetadynamicsSamplingSettings(
                timestep=1.5 * omm_unit.femtoseconds,
                bias_frequency=1.0 * omm_unit.picoseconds,
            )

    def test_invalid_bias_save_frequency_not_divisible_by_timestep(self):
        """Test that invalid bias save frequency raises error."""
        with pytest.raises(ValidationError, match="must be divisible by the timestep"):
            MMMDMetadynamicsSamplingSettings(
                timestep=1.5 * omm_unit.femtoseconds,
                bias_save_frequency=1.0 * omm_unit.picoseconds,
            )

    def test_production_time_not_divisible_by_bias_frequency(self):
        """Test that production time must be divisible by bias frequency."""
        with pytest.raises(ValidationError, match="must be divisible by"):
            MMMDMetadynamicsSamplingSettings(
                production_sampling_time_per_conformer=10.3 * omm_unit.picoseconds,
                bias_frequency=0.5 * omm_unit.picoseconds,
            )

    def test_output_types_includes_metadynamics_bias(self):
        """Test that output types include metadynamics bias."""
        from presto.outputs import OutputType

        settings = MMMDMetadynamicsSamplingSettings()
        assert OutputType.METADYNAMICS_BIAS in settings.output_types
        assert OutputType.PDB_TRAJECTORY in settings.output_types


class TestMMMDMetadynamicsTorsionMinimisationSamplingSettings:
    """Tests for MM MD metadynamics with torsion minimisation sampling settings."""

    def test_sampling_protocol(self):
        """Test that sampling protocol is set correctly."""
        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings()
        assert settings.sampling_protocol == "mm_md_metadynamics_torsion_minimisation"

    def test_inherits_from_metadynamics_settings(self):
        """Test that it inherits from MMMDMetadynamicsSamplingSettings."""
        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings()
        assert isinstance(settings, MMMDMetadynamicsSamplingSettings)

    def test_default_minimisation_steps(self):
        """Test default minimisation steps."""
        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings()
        assert settings.ml_minimisation_steps == 10
        assert settings.mm_minimisation_steps == 10

    def test_default_torsion_restraint_force_constant(self):
        """Test default torsion restraint force constant."""
        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings()
        assert (
            settings.torsion_restraint_force_constant.value_in_unit(
                omm_unit.kilojoules_per_mole / omm_unit.radian**2
            )
            == 0.0
        )

    def test_default_mmmd_loss_weights(self):
        """Test default loss weights for MMMD samples."""
        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings()
        assert settings.loss_energy_weight == 1000.0
        assert settings.loss_force_weight == 0.1

    def test_default_torsion_min_loss_weights(self):
        """Test default loss weights for torsion-minimised samples."""
        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings()
        assert settings.loss_energy_weight_mm_torsion_min == 1000.0
        assert settings.loss_force_weight_mm_torsion_min == 0.1
        assert settings.loss_energy_weight_ml_torsion_min == 1000.0
        assert settings.loss_force_weight_ml_torsion_min == 0.1

    def test_custom_minimisation_steps(self):
        """Test custom minimisation steps."""
        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings(
            ml_minimisation_steps=20,
            mm_minimisation_steps=15,
        )
        assert settings.ml_minimisation_steps == 20
        assert settings.mm_minimisation_steps == 15

    def test_custom_torsion_restraint_force_constant(self):
        """Test custom torsion restraint force constant."""
        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings(
            torsion_restraint_force_constant=500.0
            * omm_unit.kilojoules_per_mole
            / omm_unit.radian**2,
        )
        assert (
            settings.torsion_restraint_force_constant.value_in_unit(
                omm_unit.kilojoules_per_mole / omm_unit.radian**2
            )
            == 500.0
        )

    def test_custom_loss_weights(self):
        """Test custom loss weights."""
        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings(
            loss_energy_weight=500.0,
            loss_force_weight=0.5,
            loss_energy_weight_mm_torsion_min=200.0,
            loss_force_weight_mm_torsion_min=0.0,
            loss_energy_weight_ml_torsion_min=100.0,
            loss_force_weight_ml_torsion_min=0.1,
        )
        assert settings.loss_energy_weight == 500.0
        assert settings.loss_force_weight == 0.5
        assert settings.loss_energy_weight_mm_torsion_min == 200.0
        assert settings.loss_force_weight_mm_torsion_min == 0.0

    def test_yaml_round_trip(self, tmp_path):
        """Test YAML serialization round-trip."""
        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings(
            ml_minimisation_steps=25,
            mm_minimisation_steps=30,
            torsion_restraint_force_constant=750.0
            * omm_unit.kilojoules_per_mole
            / omm_unit.radian**2,
            loss_energy_weight=800.0,
            loss_force_weight_mm_torsion_min=0.05,
        )
        yaml_path = tmp_path / "settings.yaml"
        settings.to_yaml(yaml_path)

        loaded = MMMDMetadynamicsTorsionMinimisationSamplingSettings.from_yaml(
            yaml_path
        )
        assert loaded.ml_minimisation_steps == 25
        assert loaded.mm_minimisation_steps == 30
        assert (
            loaded.torsion_restraint_force_constant.value_in_unit(
                omm_unit.kilojoules_per_mole / omm_unit.radian**2
            )
            == 750.0
        )
        assert loaded.loss_energy_weight == 800.0
        assert loaded.loss_force_weight_mm_torsion_min == 0.05

    def test_inherits_metadynamics_parameters(self):
        """Test that metadynamics parameters are inherited."""
        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings(
            bias_factor=15.0,
        )
        assert settings.bias_factor == 15.0

    def test_output_types_includes_metadynamics_bias(self):
        """Test that output types include metadynamics bias."""
        from presto.outputs import OutputType

        settings = MMMDMetadynamicsTorsionMinimisationSamplingSettings()
        assert OutputType.METADYNAMICS_BIAS in settings.output_types
        assert OutputType.PDB_TRAJECTORY in settings.output_types


class TestMSMSettings:
    """Tests for MSMSettings class."""

    def test_default_settings(self):
        """Test default MSM settings."""
        settings = MSMSettings()
        assert settings.vib_scaling == 1.0
        assert settings.mlp_settings.ml_potential == "aimnet2"

    def test_custom_vib_scaling(self):
        """Test custom vibrational scaling."""
        settings = MSMSettings(vib_scaling=1.0)
        assert settings.vib_scaling == 1.0

    def test_custom_ml_potential(self):
        """Test custom ML potential."""
        settings = MSMSettings(
            mlp_settings=MLPSettings(ml_potential="mace-off23-medium")
        )
        assert settings.mlp_settings.ml_potential == "mace-off23-medium"


class TestTrainingSettings:
    """Tests for training settings."""

    def test_default_values(self):
        """Test default values."""
        settings = TrainingSettings()
        assert settings.optimiser == "adam"
        assert settings.n_epochs == 1000
        assert settings.learning_rate == 0.01
        assert settings.learning_rate_decay == 1.0
        # loss_energy_weight and loss_force_weight are now in sampling settings

    def test_optimiser_validation(self):
        """Test that only valid optimisers are accepted."""
        TrainingSettings(optimiser="adam")
        TrainingSettings(optimiser="lm")

        with pytest.raises(ValidationError):
            TrainingSettings(optimiser="invalid")

    def test_output_types(self):
        """Test that output types are defined."""
        from presto.outputs import OutputType

        settings = TrainingSettings()
        assert OutputType.TENSORBOARD in settings.output_types
        assert OutputType.TRAINING_METRICS in settings.output_types

    @given(
        n_epochs=st.integers(min_value=1, max_value=10000),
        learning_rate=st.floats(
            min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @hypothesis_settings(max_examples=10)
    def test_training_parameters_property(self, n_epochs, learning_rate):
        """Test that training parameters can be set."""
        settings = TrainingSettings(n_epochs=n_epochs, learning_rate=learning_rate)
        assert settings.n_epochs == n_epochs
        assert settings.learning_rate == learning_rate


class TestOutlierFilterSettings:
    """Tests for outlier filter settings."""

    def test_default_values(self):
        """Test default values."""
        settings = OutlierFilterSettings()
        assert settings.energy_outlier_threshold == 2.0
        assert settings.force_outlier_threshold == 500.0
        assert settings.min_conformations == 1

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        settings = OutlierFilterSettings(
            energy_outlier_threshold=5.0,
            force_outlier_threshold=25.0,
            min_conformations=5,
        )
        assert settings.energy_outlier_threshold == 5.0
        assert settings.force_outlier_threshold == 25.0
        assert settings.min_conformations == 5

    def test_disable_energy_filtering(self):
        """Test that energy filtering can be disabled."""
        settings = OutlierFilterSettings(energy_outlier_threshold=None)
        assert settings.energy_outlier_threshold is None
        assert settings.force_outlier_threshold == 500.0

    def test_disable_forces_filtering(self):
        """Test that forces filtering can be disabled."""
        settings = OutlierFilterSettings(force_outlier_threshold=None)
        assert settings.energy_outlier_threshold == 2.0
        assert settings.force_outlier_threshold is None

    def test_disable_all_filtering(self):
        """Test that all filtering can be disabled."""
        settings = OutlierFilterSettings(
            energy_outlier_threshold=None,
            force_outlier_threshold=None,
        )
        assert settings.energy_outlier_threshold is None
        assert settings.force_outlier_threshold is None

    def test_min_conformations_must_be_at_least_one(self):
        """Test that min_conformations cannot be less than 1."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            OutlierFilterSettings(min_conformations=0)

    def test_yaml_round_trip(self, tmp_path):
        """Test YAML serialization round-trip."""
        settings = OutlierFilterSettings(
            energy_outlier_threshold=7.5,
            force_outlier_threshold=40.0,
            min_conformations=3,
        )
        yaml_path = tmp_path / "outlier_settings.yaml"
        settings.to_yaml(yaml_path)

        loaded = OutlierFilterSettings.from_yaml(yaml_path)
        assert loaded.energy_outlier_threshold == 7.5
        assert loaded.force_outlier_threshold == 40.0
        assert loaded.min_conformations == 3


class TestTypeGenerationSettings:
    """Tests for type generation settings."""

    def test_default_values(self):
        """Test default values."""
        settings = TypeGenerationSettings()
        assert settings.max_extend_distance == -1
        assert settings.exclude == []

    def test_custom_max_extend_distance(self):
        """Test custom max_extend_distance."""
        settings = TypeGenerationSettings(max_extend_distance=3)
        assert settings.max_extend_distance == 3

    def test_custom_exclude_list(self):
        """Test custom exclude list."""
        exclude_list = ["[*:1]-[*:2]", "[*:1]~[*:2]"]
        settings = TypeGenerationSettings(exclude=exclude_list)
        assert settings.exclude == exclude_list

    def test_max_extend_distance_unlimited(self):
        """Test that -1 means unlimited extension."""
        settings = TypeGenerationSettings(max_extend_distance=-1)
        assert settings.max_extend_distance == -1

    def test_max_extend_distance_zero(self):
        """Test that 0 means no extension."""
        settings = TypeGenerationSettings(max_extend_distance=0)
        assert settings.max_extend_distance == 0

    def test_yaml_round_trip(self, tmp_path):
        """Test YAML serialization round-trip."""
        settings = TypeGenerationSettings(
            max_extend_distance=2, exclude=["[*:1]-[*:2]#[*:3]-[*:4]"]
        )
        yaml_path = tmp_path / "type_gen_settings.yaml"
        settings.to_yaml(yaml_path)

        loaded = TypeGenerationSettings.from_yaml(yaml_path)
        assert loaded.max_extend_distance == 2
        assert loaded.exclude == ["[*:1]-[*:2]#[*:3]-[*:4]"]


class TestParamSettings:
    """Tests for parameterisation settings."""

    def test_valid_smiles_input(self):
        """Test that valid SMILES input is accepted and converted to list."""
        settings = ParamSettings(molecule_input_type="smiles", molecules="CCO")
        assert settings.molecules == ["CCO"]
        assert len(settings.openff_molecules) == 1
        assert settings.openff_molecules[0].n_atoms == 9

    def test_invalid_smiles_raises_error(self):
        """Test that invalid SMILES raise error."""
        with pytest.raises(ValidationError, match="Invalid SMILES"):
            ParamSettings(molecule_input_type="smiles", molecules="invalid_smiles_123")

    def test_placeholder_input_raises_error(self):
        """Test that placeholder input raises error for smiles mode."""
        with pytest.raises(ValidationError, match="Invalid SMILES"):
            ParamSettings(
                molecule_input_type="smiles", molecules=_DEFAULT_INPUT_PLACEHOLDER
            )

    def test_phosphorus_warning_is_actionable_and_non_fatal(self):
        """Phosphorus warns about both minimisation-enabled code paths."""
        with pytest.warns(UserWarning) as caught:
            settings = ParamSettings(
                molecule_input_type="smiles", molecules="COP(=O)(O)O"
            )

        assert settings.openff_molecules
        message = str(caught[0].message)
        assert "molecule 0" in message
        assert "mm_md_metadynamics" in message
        assert "param_settings.msm_settings" in message
        assert "null" in message and "None" in message

    def test_every_matching_smiles_is_aggregated_in_one_warning(self):
        """A warning for one SMARTS lists all matching input molecules."""
        with pytest.warns(UserWarning) as caught:
            ParamSettings(
                molecule_input_type="smiles",
                molecules=["COP(=O)(O)O", "CP(=O)(O)O"],
            )

        assert len(caught) == 1
        message = str(caught[0].message)
        assert "molecule 0" in message and "COP(=O)(O)O" in message
        assert "molecule 1" in message and "CP(=O)(O)O" in message

    def test_different_warning_classes_are_distinct(self):
        """A molecule can independently trigger phosphorus and sulfonamide advice."""
        with pytest.warns(UserWarning) as caught:
            ParamSettings(
                molecule_input_type="smiles", molecules="OP(=O)(O)CS(=O)(=O)N"
            )

        assert len(caught) == 2
        assert any("[#15]" in str(item.message) for item in caught)
        assert any("[SX4]" in str(item.message) for item in caught)

    def test_sdf_warning_includes_record_name(self, tmp_path):
        """SDF inputs use their record name in a non-fatal warning."""
        sdf = tmp_path / "phosphorus.sdf"
        molecule = Chem.AddHs(Chem.MolFromSmiles("COP(=O)(O)O"))
        molecule.SetProp("_Name", "phosphorus-ligand")
        with Chem.SDWriter(str(sdf)) as writer:
            writer.write(molecule)

        with pytest.warns(UserWarning, match="phosphorus-ligand"):
            settings = ParamSettings(molecule_input_type="sdf", molecules=str(sdf))

        assert settings.openff_molecules[0].name == "phosphorus-ligand"

    def test_valid_single_molecule_sdf(self, tmp_path):
        """Test that valid SDF paths are accepted."""
        sdf = tmp_path / "ethanol.sdf"
        Molecule.from_smiles("CCO").to_file(str(sdf), "SDF")

        settings = ParamSettings(molecule_input_type="sdf", molecules=str(sdf))

        assert settings.molecules == [str(sdf)]
        assert len(settings.openff_molecules) == 1
        assert settings.openff_molecules[0].n_atoms == 9

    def test_missing_sdf_raises_error(self, tmp_path):
        """Test that missing SDF file raises error."""
        missing_path = tmp_path / "missing.sdf"
        with pytest.raises(ValidationError, match="SDF file does not exist"):
            ParamSettings(molecule_input_type="sdf", molecules=str(missing_path))

    def test_multi_molecule_sdf_accepts_unique_molecules(self, tmp_path):
        """Test that SDF with multiple distinct molecules is accepted."""
        sdf = tmp_path / "multi.sdf"
        with Chem.SDWriter(str(sdf)) as writer:
            writer.write(Chem.MolFromSmiles("CC"))
            writer.write(Chem.MolFromSmiles("CCC"))

        settings = ParamSettings(molecule_input_type="sdf", molecules=str(sdf))

        assert len(settings.openff_molecules) == 2
        assert (
            settings.openff_molecules[0].n_atoms != settings.openff_molecules[1].n_atoms
        )

    def test_append_to_molecules_changes_openff_molecules(self, tmp_path):
        """Test that appending to molecules updates openff_molecules."""
        settings = ParamSettings(molecule_input_type="smiles", molecules=["CC", "CCC"])
        assert len(settings.openff_molecules) == 2

        # Append another molecule
        settings.molecules.append("CCCO")

        assert len(settings.openff_molecules) == 3  # Now should have 4 molecules

    def test_duplicate_molecules_in_sdf_raises_error(self, tmp_path):
        """Test that duplicate molecules in a single SDF file are rejected."""
        sdf = tmp_path / "duplicate.sdf"
        with Chem.SDWriter(str(sdf)) as writer:
            writer.write(Chem.MolFromSmiles("CC"))
            writer.write(Chem.MolFromSmiles("CC"))

        with pytest.raises(ValidationError, match="duplicate molecule entries"):
            ParamSettings(molecule_input_type="sdf", molecules=str(sdf))

    def test_unsupported_input_type_raises_error(self):
        """Test that unsupported input_type is rejected."""
        with pytest.raises(ValidationError):
            ParamSettings(molecule_input_type="unsupported", molecules="CCO")

    def test_empty_input_list_raises_error(self):
        """Test that empty input list raises error."""
        with pytest.raises(ValidationError, match="input list cannot be empty"):
            ParamSettings(molecule_input_type="smiles", molecules=[])

    def test_duplicate_inputs_raise_error(self):
        """Test that duplicate inputs are rejected."""
        with pytest.raises(ValidationError, match="Duplicate inputs found"):
            ParamSettings(molecule_input_type="smiles", molecules=["CC", "CC"])

    def test_default_initial_force_field(self):
        """Test default initial force field."""
        settings = ParamSettings(molecule_input_type="smiles", molecules="CCO")
        assert settings.initial_force_field == "openff_unconstrained-2.3.0.offxml"

    def test_default_type_generation_settings(self):
        """Test that default type generation settings are set."""
        settings = ParamSettings(molecule_input_type="smiles", molecules="CCO")
        assert "ProperTorsions" in settings.type_generation_settings
        assert len(settings.type_generation_settings["ProperTorsions"].exclude) == 3
        assert (
            "[*:1]-[*:2]#[*:3]-[*:4]"
            in settings.type_generation_settings["ProperTorsions"].exclude
        )

    def test_linearise_harmonics(self):
        """Test linear harmonics setting."""
        settings = ParamSettings(molecule_input_type="smiles", molecules="CCO")
        assert settings.linearise_harmonics is True

    def test_expand_torsions_default(self):
        """Test that expand torsions is True by default."""
        settings = ParamSettings(molecule_input_type="smiles", molecules="CCO")
        assert settings.expand_torsions is True

    @given(smiles=st.sampled_from(["CCO", "CC", "C", "CCCC", "c1ccccc1"]))
    @hypothesis_settings(max_examples=5)
    def test_valid_simple_smiles(self, smiles):
        """Test that simple valid SMILES are accepted and converted to list."""
        settings = ParamSettings(molecule_input_type="smiles", molecules=smiles)
        assert settings.molecules == [smiles]


class TestWorkflowSettings:
    """Tests for workflow settings."""

    @pytest.fixture
    def valid_workflow_settings(self):
        """Test that default values are set correctly."""
        return WorkflowSettings(
            param_settings=ParamSettings(molecule_input_type="smiles", molecules="CCO"),
            device_type="cpu",
        )

    def test_default_values(self, valid_workflow_settings):
        """Test default values."""
        assert valid_workflow_settings.version == __version__
        assert valid_workflow_settings.output_dir == Path(".")
        assert valid_workflow_settings.n_iterations == 2
        assert valid_workflow_settings.memory is False

    def test_version_validation_compatible(self):
        """Test that compatible versions pass validation."""
        # Should accept same major.minor version
        settings = WorkflowSettings(
            version=__version__,
            param_settings=ParamSettings(molecule_input_type="smiles", molecules="CCO"),
        )
        assert settings.version == __version__

    def test_device_type_cpu(self):
        """Test that CPU device type is accepted."""
        settings = WorkflowSettings(
            param_settings=ParamSettings(molecule_input_type="smiles", molecules="CCO"),
            device_type="cpu",
        )
        assert settings.device_type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_type_cuda(self):
        """Test that CUDA device type is accepted when available."""
        settings = WorkflowSettings(
            param_settings=ParamSettings(molecule_input_type="smiles", molecules="CCO"),
            device_type="cuda",
        )
        assert settings.device_type == "cuda"

    def test_device_type_cuda_unavailable(self, monkeypatch):
        """Test that CUDA device type raises error when unavailable."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        with pytest.raises(ValueError, match="CUDA is not available"):
            WorkflowSettings(
                param_settings=ParamSettings(
                    molecule_input_type="smiles", molecules="CCO"
                ),
                device_type="cuda",
            )

    def test_device_property(self, valid_workflow_settings):
        """Test that device property returns torch.device."""
        device = valid_workflow_settings.device
        assert isinstance(device, torch.device)
        assert device.type == "cpu"

    def test_get_path_manager(self, valid_workflow_settings):
        """Test that path manager is created correctly."""
        path_manager = valid_workflow_settings.get_path_manager()
        assert path_manager.output_dir == valid_workflow_settings.output_dir
        assert path_manager.n_iterations == valid_workflow_settings.n_iterations

    def test_yaml_round_trip(self, tmp_path, valid_workflow_settings):
        """Test YAML serialization round-trip."""
        yaml_path = tmp_path / "workflow_settings.yaml"
        valid_workflow_settings.to_yaml(yaml_path)

        loaded = WorkflowSettings.from_yaml(yaml_path)
        assert loaded.n_iterations == valid_workflow_settings.n_iterations
        assert loaded.memory == valid_workflow_settings.memory
        assert (
            loaded.param_settings.molecules
            == valid_workflow_settings.param_settings.molecules
        )

    def test_workflow_yaml_with_runtime_object_requires_override(self, tmp_path):
        """Test workflow YAML with runtime placeholders requires overwrite before load."""
        settings = WorkflowSettings(
            param_settings=ParamSettings(molecule_input_type="smiles", molecules="CCO"),
            device_type="cpu",
            training_sampling_settings=MLMDSamplingSettings(
                mlp_settings=MLPSettings(
                    ml_potential="ase",
                    ml_system_kwargs={"calculator": object()},
                ),
            ),
        )
        yaml_path = tmp_path / "workflow_settings_ase.yaml"
        settings.to_yaml(yaml_path)

        with pytest.raises(ValidationError, match="runtime-only placeholder values"):
            WorkflowSettings.from_yaml(yaml_path)

        loaded = WorkflowSettings.from_yaml(
            yaml_path,
            overwrite={
                "training_sampling_settings": {
                    "mlp_settings": {"ml_system_kwargs": {"calculator": object()}}
                }
            },
        )
        assert loaded.training_sampling_settings.mlp_settings.ml_potential == "ase"

    def test_discriminated_union_for_sampling_settings(self):
        """Test that discriminated union works for sampling settings."""
        # MM MD
        settings_mm = WorkflowSettings(
            param_settings=ParamSettings(molecule_input_type="smiles", molecules="CCO"),
            training_sampling_settings=MMMDSamplingSettings(),
        )
        assert settings_mm.training_sampling_settings.sampling_protocol == "mm_md"

        # ML MD
        settings_ml = WorkflowSettings(
            param_settings=ParamSettings(molecule_input_type="smiles", molecules="CCO"),
            training_sampling_settings=MLMDSamplingSettings(),
        )
        assert settings_ml.training_sampling_settings.sampling_protocol == "ml_md"

        # Metadynamics
        settings_metad = WorkflowSettings(
            param_settings=ParamSettings(molecule_input_type="smiles", molecules="CCO"),
            training_sampling_settings=MMMDMetadynamicsSamplingSettings(),
        )
        assert (
            settings_metad.training_sampling_settings.sampling_protocol
            == "mm_md_metadynamics"
        )

        # Metadynamics with torsion minimisation
        settings_metad_torsion = WorkflowSettings(
            param_settings=ParamSettings(molecule_input_type="smiles", molecules="CCO"),
            training_sampling_settings=MMMDMetadynamicsTorsionMinimisationSamplingSettings(),
        )
        assert (
            settings_metad_torsion.training_sampling_settings.sampling_protocol
            == "mm_md_metadynamics_torsion_minimisation"
        )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            WorkflowSettings(
                param_settings=ParamSettings(
                    molecule_input_type="smiles", molecules="CCO"
                ),
                invalid_field="should_fail",
            )

    @given(
        n_iterations=st.integers(min_value=1, max_value=10),
        memory=st.booleans(),
    )
    @hypothesis_settings(max_examples=10)
    def test_workflow_configuration_property(self, n_iterations, memory):
        """Test that workflow configuration can be set."""
        settings = WorkflowSettings(
            param_settings=ParamSettings(molecule_input_type="smiles", molecules="CCO"),
            n_iterations=n_iterations,
            memory=memory,
        )
        assert settings.n_iterations == n_iterations
        assert settings.memory == memory

    def test_workflow_requires_at_least_one_iteration(self):
        """A workflow must have a final training iteration."""
        with pytest.raises(ValidationError, match="n_iterations"):
            WorkflowSettings(
                param_settings=ParamSettings(
                    molecule_input_type="smiles", molecules="CCO"
                ),
                n_iterations=0,
            )

    def test_outlier_filter_settings_has_default(self, valid_workflow_settings):
        """Test that outlier_filter_settings has default values."""
        assert valid_workflow_settings.outlier_filter_settings is not None
        assert (
            valid_workflow_settings.outlier_filter_settings.energy_outlier_threshold
            == 2.0
        )
        assert (
            valid_workflow_settings.outlier_filter_settings.force_outlier_threshold
            == 500.0
        )

    def test_outlier_filter_settings_can_be_set(self):
        """Test that outlier_filter_settings can be configured."""
        settings = WorkflowSettings(
            param_settings=ParamSettings(molecule_input_type="smiles", molecules="CCO"),
            device_type="cpu",
            outlier_filter_settings=OutlierFilterSettings(
                energy_outlier_threshold=5.0,  # kcal/mol/atom
                force_outlier_threshold=25.0,
            ),
        )
        assert settings.outlier_filter_settings is not None
        assert settings.outlier_filter_settings.energy_outlier_threshold == 5.0
        assert settings.outlier_filter_settings.force_outlier_threshold == 25.0

    def test_outlier_filter_settings_yaml_round_trip(self, tmp_path):
        """Test that outlier_filter_settings survives YAML round-trip."""
        settings = WorkflowSettings(
            param_settings=ParamSettings(molecule_input_type="smiles", molecules="CCO"),
            device_type="cpu",
            outlier_filter_settings=OutlierFilterSettings(
                energy_outlier_threshold=7.5,  # kcal/mol/atom
                force_outlier_threshold=30.0,
                min_conformations=2,
            ),
        )
        yaml_path = tmp_path / "workflow_with_outlier.yaml"
        settings.to_yaml(yaml_path)

        loaded = WorkflowSettings.from_yaml(yaml_path)
        assert loaded.outlier_filter_settings is not None
        assert loaded.outlier_filter_settings.energy_outlier_threshold == 7.5
        assert loaded.outlier_filter_settings.force_outlier_threshold == 30.0
        assert loaded.outlier_filter_settings.min_conformations == 2


def _write_single_conformer_sdf(smiles: str, path: Path) -> None:
    """Write a molecule with one conformer to an SDF file."""
    molecule = Molecule.from_smiles(smiles)
    molecule.generate_conformers(n_conformers=1)
    molecule.to_file(str(path), "SDF")


class TestStartingConformersField:
    """Tests for the optional starting_conformers field on sampling and MSM settings."""

    def test_default_is_none(self):
        """starting_conformers defaults to None (ETKDG) on every stage."""
        assert MMMDSamplingSettings().starting_conformers is None
        assert MLMDSamplingSettings().starting_conformers is None
        assert MSMSettings().starting_conformers is None

    def test_accepts_existing_sdf(self, tmp_path):
        """A valid SDF path is accepted."""
        sdf = tmp_path / "confs.sdf"
        _write_single_conformer_sdf("CCO", sdf)

        settings = MMMDSamplingSettings(starting_conformers=sdf)
        assert settings.starting_conformers == sdf

        msm = MSMSettings(starting_conformers=sdf)
        assert msm.starting_conformers == sdf

    def test_missing_file_rejected(self, tmp_path):
        """A missing SDF path is rejected at construction."""
        with pytest.raises(ValidationError, match="does not exist"):
            MMMDSamplingSettings(starting_conformers=tmp_path / "missing.sdf")

    def test_non_sdf_suffix_rejected(self, tmp_path):
        """A non-.sdf path is rejected at construction."""
        other = tmp_path / "confs.mol2"
        other.write_text("")
        with pytest.raises(ValidationError, match="must be an SDF file"):
            MMMDSamplingSettings(starting_conformers=other)

    def test_workflow_accepts_matching_conformers(self, tmp_path):
        """WorkflowSettings validates when the SDF contains the fitted molecule."""
        sdf = tmp_path / "confs.sdf"
        _write_single_conformer_sdf("CCO", sdf)

        settings = WorkflowSettings(
            param_settings=ParamSettings(molecule_input_type="smiles", molecules="CCO"),
            device_type="cpu",
            training_sampling_settings=MMMDSamplingSettings(starting_conformers=sdf),
        )
        assert settings.training_sampling_settings.starting_conformers == sdf

    def test_workflow_rejects_missing_molecule(self, tmp_path):
        """WorkflowSettings fails fast when a fitted molecule is absent from the SDF."""
        sdf = tmp_path / "confs.sdf"
        _write_single_conformer_sdf("CCO", sdf)

        with pytest.raises(ValidationError, match="starting_conformers"):
            WorkflowSettings(
                param_settings=ParamSettings(
                    molecule_input_type="smiles", molecules="c1ccccc1"
                ),
                device_type="cpu",
                training_sampling_settings=MMMDSamplingSettings(
                    starting_conformers=sdf
                ),
            )

    def test_workflow_validates_msm_conformers(self, tmp_path):
        """The MSM starting_conformers path is also cross-checked against molecules."""
        sdf = tmp_path / "confs.sdf"
        _write_single_conformer_sdf("CCO", sdf)

        with pytest.raises(ValidationError, match=r"msm_settings\.starting_conformers"):
            WorkflowSettings(
                param_settings=ParamSettings(
                    molecule_input_type="smiles",
                    molecules="c1ccccc1",
                    msm_settings=MSMSettings(starting_conformers=sdf),
                ),
                device_type="cpu",
            )
