"""Unit tests for convert.py."""

import math
from unittest import mock

import openff.interchange
import openff.toolkit
import pytest
import smee
import torch
from openff.units import unit as off_unit

from presto._exceptions import MoleculeParameterisationError
from presto.convert import (
    _add_angle_within_range,
    _compute_linear_harmonic_params,
    _expand_torsions,
    _linearize_angle_parameters,
    _linearize_bond_parameters,
    _reflect_angle,
    convert_to_smirnoff,
    linearise_harmonics_force_field,
    linearise_harmonics_topology,
    parameterise,
)
from presto.settings import MLPSettings, MSMSettings, ParamSettings
from presto.tests.conftest import skip_if_model_unavailable


@pytest.mark.parametrize(
    "angle, expected",
    [
        (0.5, 0.5),
        (math.pi + 0.5, math.pi - 0.5),
        (2 * math.pi + 0.5, 0.5),
        (0.0, 0.0),
        (math.pi, math.pi),
        (2 * math.pi, 0.0),
    ],
)
def test_reflect_angle(angle, expected):
    """Reflect angle."""
    if angle == 2 * math.pi:
        assert math.isclose(_reflect_angle(angle), expected, abs_tol=1e-9)
    else:
        assert math.isclose(_reflect_angle(angle), expected)


@pytest.mark.parametrize(
    "angle1, angle2, expected",
    [
        (1.0, 0.5, 1.5),
        (1.0, 3.0, math.pi),
        (1.0, -0.5, 0.5),
        (1.0, -2.0, 0.0),
    ],
)
def test_add_angle_within_range(angle1, angle2, expected):
    """Add angle within range."""
    assert math.isclose(_add_angle_within_range(angle1, angle2), expected)


def test_convert_to_smirnoff():
    """Convert to smirnoff."""
    # Test Bonds, Angles, ProperTorsions
    bond_pot = smee.TensorPotential(
        type="Bonds",
        fn="Harmonic",
        parameter_cols=("k", "length"),
        parameter_units=(
            off_unit.kilocalorie_per_mole / off_unit.angstrom**2,
            off_unit.angstrom,
        ),
        parameters=torch.tensor([[100.0, 1.0]]),
        parameter_keys=[
            openff.interchange.models.PotentialKey(id="[#6:1]-[#1:2]", mult=None)
        ],
    )
    angle_pot = smee.TensorPotential(
        type="Angles",
        fn="Harmonic",
        parameter_cols=("k", "angle"),
        parameter_units=(
            off_unit.kilocalorie_per_mole / off_unit.radians**2,
            off_unit.radians,
        ),
        parameters=torch.tensor([[50.0, 1.5]]),
        parameter_keys=[
            openff.interchange.models.PotentialKey(id="[#1:1]-[#6:2]-[#1:3]", mult=None)
        ],
    )
    torsion_pot = smee.TensorPotential(
        type="ProperTorsions",
        fn="Periodic",
        parameter_cols=("k", "periodicity", "phase", "idivf"),
        parameter_units=(
            off_unit.kilocalorie_per_mole,
            off_unit.dimensionless,
            off_unit.radians,
            off_unit.dimensionless,
        ),
        parameters=torch.tensor([[1.0, 3.0, 0.0, 1.0]]),
        parameter_keys=[
            openff.interchange.models.PotentialKey(
                id="[#6:1]-[#6:2]-[#6:3]-[#6:4]", mult=0
            )
        ],
    )

    ff = smee.TensorForceField(potentials=[bond_pot, angle_pot, torsion_pot])
    off_ff = convert_to_smirnoff(ff)

    assert "Bonds" in off_ff.registered_parameter_handlers
    assert "Angles" in off_ff.registered_parameter_handlers
    assert "ProperTorsions" in off_ff.registered_parameter_handlers

    assert len(off_ff["Bonds"].parameters) == 1
    assert off_ff["Bonds"].parameters[0].smirks == "[#6:1]-[#1:2]"
    assert math.isclose(off_ff["Bonds"].parameters[0].k.m, 100.0)
    assert math.isclose(off_ff["Bonds"].parameters[0].length.m, 1.0)

    assert len(off_ff["Angles"].parameters) == 1
    assert off_ff["Angles"].parameters[0].smirks == "[#1:1]-[#6:2]-[#1:3]"
    assert math.isclose(off_ff["Angles"].parameters[0].k.m, 50.0)
    assert math.isclose(off_ff["Angles"].parameters[0].angle.m, 1.5)

    assert len(off_ff["ProperTorsions"].parameters) == 1
    assert (
        off_ff["ProperTorsions"].parameters[0].smirks == "[#6:1]-[#6:2]-[#6:3]-[#6:4]"
    )
    assert off_ff["ProperTorsions"].parameters[0].k1.m == 1.0
    assert off_ff["ProperTorsions"].parameters[0].periodicity == [3]
    assert off_ff["ProperTorsions"].parameters[0].phase == [0.0]
    assert off_ff["ProperTorsions"].parameters[0].idivf == [1.0]


def test_convert_to_smirnoff_linear():
    """Convert to smirnoff linear."""
    # Test LinearBonds and LinearAngles
    linear_bond_pot = smee.TensorPotential(
        type="LinearBonds",
        fn="LinearHarmonic",
        parameter_cols=("k1", "k2", "length1", "length2"),
        parameter_units=(
            off_unit.kilocalorie_per_mole / off_unit.angstrom**2,
            off_unit.kilocalorie_per_mole / off_unit.angstrom**2,
            off_unit.angstrom,
            off_unit.angstrom,
        ),
        parameters=torch.tensor([[60.0, 40.0, 1.0, 1.1]]),
        parameter_keys=[
            openff.interchange.models.PotentialKey(id="[#6:1]-[#1:2]", mult=None)
        ],
    )
    linear_angle_pot = smee.TensorPotential(
        type="LinearAngles",
        fn="LinearHarmonic",
        parameter_cols=("k1", "k2", "angle1", "angle2"),
        parameter_units=(
            off_unit.kilocalorie_per_mole / off_unit.radians**2,
            off_unit.kilocalorie_per_mole / off_unit.radians**2,
            off_unit.radians,
            off_unit.radians,
        ),
        parameters=torch.tensor([[30.0, 20.0, 1.5, 1.6]]),
        parameter_keys=[
            openff.interchange.models.PotentialKey(id="[#1:1]-[#6:2]-[#1:3]", mult=None)
        ],
    )

    ff = smee.TensorForceField(potentials=[linear_bond_pot, linear_angle_pot])
    off_ff = convert_to_smirnoff(ff)

    assert "Bonds" in off_ff.registered_parameter_handlers
    assert "Angles" in off_ff.registered_parameter_handlers

    assert math.isclose(off_ff["Bonds"].parameters[0].k.m, 100.0, rel_tol=1e-6)
    assert math.isclose(off_ff["Bonds"].parameters[0].length.m, 1.04, rel_tol=1e-6)

    assert math.isclose(off_ff["Angles"].parameters[0].k.m, 50.0, rel_tol=1e-6)
    assert math.isclose(off_ff["Angles"].parameters[0].angle.m, 1.54, rel_tol=1e-6)


def test_linearise_harmonics_force_field():
    """Linearise harmonics force field."""
    bond_pot = smee.TensorPotential(
        type="Bonds",
        fn="Harmonic",
        parameter_cols=("k", "length"),
        parameter_units=(
            off_unit.kilocalorie_per_mole / off_unit.angstrom**2,
            off_unit.angstrom,
        ),
        parameters=torch.tensor([[100.0, 1.0]]),
        parameter_keys=[
            openff.interchange.models.PotentialKey(id="[#6:1]-[#1:2]", mult=None)
        ],
    )
    ff = smee.TensorForceField(potentials=[bond_pot])

    linear_ff = linearise_harmonics_force_field(ff, device_type="cpu")
    assert "LinearBonds" in linear_ff.potentials_by_type
    pot = linear_ff.potentials_by_type["LinearBonds"]
    # k=100 -> k1=50, k2=50; length=1.0 -> b1=0.6, b2=1.4
    assert torch.allclose(pot.parameters, torch.tensor([[50.0, 50.0, 0.6, 1.4]]))


def test_linearise_harmonics_topology():
    """Linearise harmonics topology."""
    top = smee.TensorTopology(
        atomic_nums=torch.tensor([6, 1, 1]),
        formal_charges=torch.tensor([0, 0, 0]),
        bond_idxs=torch.tensor([[0, 1], [0, 2]]),
        bond_orders=torch.tensor([1.0, 1.0]),
        parameters={
            "Bonds": smee.ValenceParameterMap(
                particle_idxs=torch.tensor([[0, 1]]),
                assignment_matrix=torch.eye(1).to_sparse(),
            ),
            "Angles": smee.ValenceParameterMap(
                particle_idxs=torch.tensor([[1, 0, 2]]),
                assignment_matrix=torch.eye(1).to_sparse(),
            ),
        },
    )

    linear_top = linearise_harmonics_topology(top, device_type="cpu")
    assert "LinearBonds" in linear_top.parameters
    assert "LinearAngles" in linear_top.parameters
    assert "Bonds" not in linear_top.parameters
    assert "Angles" not in linear_top.parameters


def test_parameterise(tmp_path):
    """Parameterise."""
    skip_if_model_unavailable("aimnet2")
    settings = ParamSettings(
        molecule_input_type="smiles",
        molecules="CC",
        initial_force_field="openff-2.1.0.offxml",
        msm_settings=MSMSettings(mlp_settings=MLPSettings(ml_potential="aimnet2")),
    )

    # We can use small molecules and real FF for a fast enough test
    # (openff-2.1.0 should be cached in most envs)
    mols, bespoke_ff, tensor_tops, tensor_ff = parameterise(settings, device="cpu")

    assert len(mols) == 1
    assert isinstance(bespoke_ff, openff.toolkit.ForceField)
    assert len(tensor_tops) == 1
    assert isinstance(tensor_ff, smee.TensorForceField)


def test_parameterise_linear(tmp_path):
    """Parameterise linear."""
    settings = ParamSettings(
        molecule_input_type="smiles",
        molecules="C",
        initial_force_field="openff-2.1.0.offxml",
        linearise_harmonics=True,
        msm_settings=None,
    )
    _mols, _bespoke_ff, tensor_tops, tensor_ff = parameterise(settings, device="cpu")
    assert "LinearBonds" in tensor_ff.potentials_by_type
    assert "LinearBonds" in tensor_tops[0].parameters


class TestConvertToSmirnoffExtended:
    """Extended tests for convert_to_smirnoff."""

    def test_improper_torsions_conversion(self):
        """Test conversion of ImproperTorsions."""
        improper_pot = smee.TensorPotential(
            type="ImproperTorsions",
            fn="Periodic",
            parameter_cols=("k", "periodicity", "phase", "idivf"),
            parameter_units=(
                off_unit.kilocalorie_per_mole,
                off_unit.dimensionless,
                off_unit.radians,
                off_unit.dimensionless,
            ),
            parameters=torch.tensor([[2.0, 2.0, 0.0, 1.0]]),
            parameter_keys=[
                openff.interchange.models.PotentialKey(
                    id="[#6:1]~[#6X4:2](~[#6:3])~[#6:4]", mult=0
                )
            ],
        )

        ff = smee.TensorForceField(potentials=[improper_pot])
        off_ff = convert_to_smirnoff(ff)

        assert "ImproperTorsions" in off_ff.registered_parameter_handlers
        assert len(off_ff["ImproperTorsions"].parameters) == 1

    def test_multiple_periodicities_proper_torsions(self):
        """Test proper torsions with multiple periodicities."""
        torsion_pot = smee.TensorPotential(
            type="ProperTorsions",
            fn="Periodic",
            parameter_cols=("k", "periodicity", "phase", "idivf"),
            parameter_units=(
                off_unit.kilocalorie_per_mole,
                off_unit.dimensionless,
                off_unit.radians,
                off_unit.dimensionless,
            ),
            parameters=torch.tensor(
                [
                    [1.0, 1.0, 0.0, 1.0],
                    [0.5, 2.0, math.pi, 1.0],
                    [0.2, 3.0, 0.0, 1.0],
                ]
            ),
            parameter_keys=[
                openff.interchange.models.PotentialKey(
                    id="[#6:1]-[#6:2]-[#6:3]-[#6:4]", mult=0
                ),
                openff.interchange.models.PotentialKey(
                    id="[#6:1]-[#6:2]-[#6:3]-[#6:4]", mult=1
                ),
                openff.interchange.models.PotentialKey(
                    id="[#6:1]-[#6:2]-[#6:3]-[#6:4]", mult=2
                ),
            ],
        )

        ff = smee.TensorForceField(potentials=[torsion_pot])
        off_ff = convert_to_smirnoff(ff)

        assert "ProperTorsions" in off_ff.registered_parameter_handlers
        handler = off_ff["ProperTorsions"]
        assert len(handler.parameters) == 1
        param = handler.parameters[0]

        # Check that all three periodicities are present
        assert param.k1.m == pytest.approx(1.0)
        assert param.k2.m == pytest.approx(0.5)
        assert param.k3.m == pytest.approx(0.2)

    def test_convert_with_base_forcefield(self):
        """Test conversion when base force field is provided."""
        bond_pot = smee.TensorPotential(
            type="Bonds",
            fn="Harmonic",
            parameter_cols=("k", "length"),
            parameter_units=(
                off_unit.kilocalorie_per_mole / off_unit.angstrom**2,
                off_unit.angstrom,
            ),
            parameters=torch.tensor([[150.0, 1.1]]),
            parameter_keys=[
                openff.interchange.models.PotentialKey(
                    id="[#6X3:1]-[#6X3:2]", mult=None
                )
            ],
        )

        ff = smee.TensorForceField(potentials=[bond_pot])
        base_ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")
        original_bond_count = len(base_ff.get_parameter_handler("Bonds").parameters)

        off_ff = convert_to_smirnoff(ff, base=base_ff)

        # Should have the original parameters plus the new one
        new_bond_count = len(off_ff.get_parameter_handler("Bonds").parameters)
        assert new_bond_count >= original_bond_count

    def test_unexpected_parameters_error_case1(self):
        """Test error when None and other mults coexist."""
        torsion_pot = smee.TensorPotential(
            type="ProperTorsions",
            fn="Periodic",
            parameter_cols=("k", "periodicity", "phase", "idivf"),
            parameter_units=(
                off_unit.kilocalorie_per_mole,
                off_unit.dimensionless,
                off_unit.radians,
                off_unit.dimensionless,
            ),
            parameters=torch.tensor(
                [
                    [1.0, 1.0, 0.0, 1.0],
                    [0.5, 2.0, 0.0, 1.0],
                ]
            ),
            parameter_keys=[
                openff.interchange.models.PotentialKey(
                    id="[#6:1]-[#6:2]-[#6:3]-[#6:4]", mult=None
                ),
                openff.interchange.models.PotentialKey(
                    id="[#6:1]-[#6:2]-[#6:3]-[#6:4]", mult=0
                ),
            ],
        )

        ff = smee.TensorForceField(potentials=[torsion_pot])

        with pytest.raises(NotImplementedError, match="unexpected parameters found"):
            convert_to_smirnoff(ff)

    def test_unexpected_parameters_error_case2(self):
        """Test error when mults are not sequential."""
        torsion_pot = smee.TensorPotential(
            type="ProperTorsions",
            fn="Periodic",
            parameter_cols=("k", "periodicity", "phase", "idivf"),
            parameter_units=(
                off_unit.kilocalorie_per_mole,
                off_unit.dimensionless,
                off_unit.radians,
                off_unit.dimensionless,
            ),
            parameters=torch.tensor(
                [
                    [1.0, 1.0, 0.0, 1.0],
                    [0.5, 3.0, 0.0, 1.0],
                ]
            ),
            parameter_keys=[
                openff.interchange.models.PotentialKey(
                    id="[#6:1]-[#6:2]-[#6:3]-[#6:4]", mult=0
                ),
                openff.interchange.models.PotentialKey(
                    id="[#6:1]-[#6:2]-[#6:3]-[#6:4]", mult=2
                ),
            ],
        )

        ff = smee.TensorForceField(potentials=[torsion_pot])

        with pytest.raises(NotImplementedError, match="unexpected parameters found"):
            convert_to_smirnoff(ff)


class TestExpandTorsions:
    """Test _expand_torsions function."""

    def test_expand_torsions_default(self):
        """Test expanding torsions to K0-4."""
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")
        expanded_ff = _expand_torsions(ff)

        # Check that torsions have been expanded
        torsion_handler = expanded_ff.get_parameter_handler("ProperTorsions")
        for param in torsion_handler.parameters:
            # All should now have 4 periodicities
            assert len(param.k) == 4
            assert len(param.periodicity) == 4
            assert param.periodicity == [1, 2, 3, 4]

    def test_expand_torsions_with_exclusions(self):
        """Test expanding torsions with exclusions."""
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")

        # Get a specific SMIRKS to exclude
        torsion_handler = ff.get_parameter_handler("ProperTorsions")
        if len(torsion_handler.parameters) > 0:
            excluded_smirks = torsion_handler.parameters[0].smirks

            expanded_ff = _expand_torsions(ff, excluded_smirks=[excluded_smirks])

            # The excluded parameter should not be expanded
            expanded_handler = expanded_ff.get_parameter_handler("ProperTorsions")
            _ = expanded_handler.get_parameter({"smirks": excluded_smirks})[0]

            # Others should be expanded
            expanded_count = 0
            for param in expanded_handler.parameters:
                if param.smirks != excluded_smirks:
                    assert len(param.k) == 4
                    expanded_count += 1

            assert expanded_count > 0

    def test_expand_torsions_preserves_existing_values(self):
        """Test that existing k values are preserved in the correct positions."""
        ff = openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml")
        torsion_handler = ff.get_parameter_handler("ProperTorsions")

        # Find a parameter with specific periodicity
        for param in torsion_handler.parameters:
            if 2 in param.periodicity:
                original_idx = param.periodicity.index(2)
                original_k2 = param.k[original_idx]
                original_phase2 = param.phase[original_idx]
                smirks = param.smirks
                break
        else:
            pytest.skip("No parameter with periodicity 2 found")

        expanded_ff = _expand_torsions(ff)
        expanded_handler = expanded_ff.get_parameter_handler("ProperTorsions")
        expanded_param = expanded_handler.get_parameter({"smirks": smirks})[0]

        # The k value for periodicity 2 should be preserved
        assert expanded_param.k[1] == original_k2
        assert expanded_param.phase[1] == original_phase2


class TestComputeLinearHarmonicParams:
    """Test _compute_linear_harmonic_params function."""

    def test_basic_computation(self):
        """Test basic parameter computation."""
        k = 100.0
        eq_value = 1.5

        k1, k2, eq1, eq2 = _compute_linear_harmonic_params(
            k,
            eq_value,
            lambda x: x - 0.5,
            lambda x: x + 0.5,
        )

        # Check bounds
        assert eq1 == pytest.approx(1.0)
        assert eq2 == pytest.approx(2.0)

        # Check force constants sum to original
        assert k1 + k2 == pytest.approx(k)

        # Check weighted average gives equilibrium value
        assert (k1 * eq1 + k2 * eq2) / (k1 + k2) == pytest.approx(eq_value)

    def test_asymmetric_bounds(self):
        """Test with asymmetric bounds."""
        k = 200.0
        eq_value = 1.0

        k1, k2, eq1, eq2 = _compute_linear_harmonic_params(
            k,
            eq_value,
            lambda x: 0.5,
            lambda x: 2.0,
        )

        # Equilibrium closer to lower bound, so k1 should be larger
        assert k1 > k2

        # Verify relationships
        assert (k1 * eq1 + k2 * eq2) / (k1 + k2) == pytest.approx(eq_value)


class TestLinearizeParameters:
    """Test _linearize_bond_parameters and _linearize_angle_parameters."""

    def test_linearize_bond_parameters(self):
        """Test bond parameter linearization."""
        potential = smee.TensorPotential(
            type="Bonds",
            fn="Harmonic",
            parameter_cols=("k", "length"),
            parameter_units=(
                off_unit.kilocalorie_per_mole / off_unit.angstrom**2,
                off_unit.angstrom,
            ),
            parameters=torch.tensor([[100.0, 1.5]]),
            parameter_keys=[
                openff.interchange.models.PotentialKey(id="[#6:1]-[#6:2]", mult=None)
            ],
        )

        linearized = _linearize_bond_parameters(potential, "cpu")

        assert linearized.type == "LinearBonds"
        assert linearized.parameter_cols == ("k1", "k2", "b1", "b2")
        assert len(linearized.parameters) == 1

        # Check that b1 and b2 are symmetric around equilibrium
        params = linearized.parameters[0]
        b1, b2 = params[2].item(), params[3].item()
        assert b1 == pytest.approx(1.5 - 0.4)
        assert b2 == pytest.approx(1.5 + 0.4)

    def test_linearize_angle_parameters(self):
        """Test angle parameter linearization."""
        potential = smee.TensorPotential(
            type="Angles",
            fn="Harmonic",
            parameter_cols=("k", "angle"),
            parameter_units=(
                off_unit.kilocalorie_per_mole / off_unit.radians**2,
                off_unit.radians,
            ),
            parameters=torch.tensor([[50.0, math.pi / 2]]),
            parameter_keys=[
                openff.interchange.models.PotentialKey(
                    id="[#1:1]-[#6:2]-[#1:3]", mult=None
                )
            ],
        )

        linearized = _linearize_angle_parameters(potential, "cpu")

        assert linearized.type == "LinearAngles"
        assert linearized.parameter_cols == ("k1", "k2", "angle1", "angle2")
        assert len(linearized.parameters) == 1

        # Check that angles are bounded correctly
        params = linearized.parameters[0]
        angle1, angle2 = params[2].item(), params[3].item()
        assert 0 <= angle1 <= math.pi
        assert 0 <= angle2 <= math.pi
        assert angle1 < angle2

    def test_linearize_angle_near_zero(self):
        """Test angle linearization near 0."""
        potential = smee.TensorPotential(
            type="Angles",
            fn="Harmonic",
            parameter_cols=("k", "angle"),
            parameter_units=(
                off_unit.kilocalorie_per_mole / off_unit.radians**2,
                off_unit.radians,
            ),
            parameters=torch.tensor([[50.0, 0.2]]),
            parameter_keys=[
                openff.interchange.models.PotentialKey(
                    id="[#1:1]-[#6:2]-[#1:3]", mult=None
                )
            ],
        )

        linearized = _linearize_angle_parameters(potential, "cpu")
        params = linearized.parameters[0]
        angle1 = params[2].item()

        # Should be clamped to 0
        assert angle1 == pytest.approx(0.0)

    def test_linearize_angle_near_pi(self):
        """Test angle linearization near π."""
        potential = smee.TensorPotential(
            type="Angles",
            fn="Harmonic",
            parameter_cols=("k", "angle"),
            parameter_units=(
                off_unit.kilocalorie_per_mole / off_unit.radians**2,
                off_unit.radians,
            ),
            parameters=torch.tensor([[50.0, math.pi - 0.1]]),
            parameter_keys=[
                openff.interchange.models.PotentialKey(
                    id="[#1:1]-[#6:2]-[#1:3]", mult=None
                )
            ],
        )

        linearized = _linearize_angle_parameters(potential, "cpu")
        params = linearized.parameters[0]
        angle2 = params[3].item()

        # Should be clamped to π
        assert angle2 == pytest.approx(math.pi)


class TestParameteriseExtended:
    """Extended tests for parameterise function."""

    def test_parameterise_with_expand_torsions(self):
        """Test parameterise with expand_torsions enabled."""
        settings = ParamSettings(
            molecule_input_type="smiles",
            molecules="CCCC",
            initial_force_field="openff_unconstrained-2.3.0.offxml",
            expand_torsions=True,
            msm_settings=None,
        )

        _mols, bespoke_ff, _tensor_tops, _tensor_ff = parameterise(
            settings, device="cpu"
        )

        # Check that torsions were expanded
        torsion_handler = bespoke_ff.get_parameter_handler("ProperTorsions")
        expanded_count = 0
        for param in torsion_handler.parameters:
            if "bespoke" not in param.id and len(param.k) > 1:
                # Check if this one was expanded - should have 4 periodicities
                if any(p in param.periodicity for p in [1, 2, 3, 4]):
                    assert len(param.k) == 4
                    expanded_count += 1

        # At least some parameters should have been expanded
        assert expanded_count > 0

    def test_parameterise_multiple_molecules(self):
        """Test parameterise with multiple molecules."""
        settings = ParamSettings(
            molecule_input_type="smiles",
            molecules=["C", "CC"],
            initial_force_field="openff_unconstrained-2.3.0.offxml",
            msm_settings=None,
        )

        mols, _bespoke_ff, tensor_tops, tensor_ff = parameterise(settings, device="cpu")

        assert len(mols) == 2
        assert len(tensor_tops) == 2
        # Force field should contain parameters for both molecules
        assert isinstance(tensor_ff, smee.TensorForceField)

    def test_parameterise_reports_both_stages_for_one_molecule(self):
        """A molecule failing MSM is still charged, so one report lists both problems."""
        settings = ParamSettings(
            molecule_input_type="smiles",
            molecules=["C", "CC"],
            initial_force_field="openff_unconstrained-2.3.0.offxml",
            expand_torsions=False,
            msm_settings=MSMSettings(mlp_settings=MLPSettings(ml_potential="aimnet2")),
        )

        with mock.patch(
            "presto.convert.apply_msm_to_molecules",
            return_value=(
                openff.toolkit.ForceField(settings.initial_force_field),
                {0: "ValueError: methane conformers"},
            ),
        ):
            with mock.patch.object(
                openff.interchange.Interchange,
                "from_smirnoff",
                side_effect=[ValueError("methane charges"), mock.sentinel.ok],
            ) as from_smirnoff:
                with pytest.raises(MoleculeParameterisationError) as exc_info:
                    parameterise(settings, device="cpu")

        # The MSM failure does not stop the molecule being tried at the charge stage.
        assert from_smirnoff.call_count == 2
        message = str(exc_info.value)
        assert "1 of 2 molecules" in message
        assert (
            "molecule 0 (C): ValueError: methane conformers; "
            "ValueError: methane charges" in message
        )

    def test_parameterise_reports_every_failing_molecule(self):
        """Every molecule is attempted, and one error lists all the failures."""
        settings = ParamSettings(
            molecule_input_type="smiles",
            molecules=["C", "CC", "CCC"],
            initial_force_field="openff_unconstrained-2.3.0.offxml",
            expand_torsions=False,
            msm_settings=None,
        )

        with mock.patch.object(
            openff.interchange.Interchange,
            "from_smirnoff",
            side_effect=[
                ValueError("methane failure"),
                mock.sentinel.ok,
                ValueError("propane failure"),
            ],
        ) as from_smirnoff:
            with pytest.raises(MoleculeParameterisationError) as exc_info:
                parameterise(settings, device="cpu")

        # No early exit: the failure report is only worth waiting for if it is complete.
        assert from_smirnoff.call_count == 3
        message = str(exc_info.value)
        assert "2 of 3 molecules" in message
        assert "molecule 0 (C): ValueError: methane failure" in message
        assert "molecule 2 (CCC): ValueError: propane failure" in message
        assert "molecule 1" not in message
