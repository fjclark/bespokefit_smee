"""Unit tests for msm.py.

msm.py implements the modified Seminario method (for computing bond and angle parameters).
See https://doi.org/10.1021/acs.jctc.7b00785.
"""

import json
import math
from importlib.resources import files
from unittest.mock import MagicMock, patch, sentinel

import numpy as np
import pytest
import torch
from openff.toolkit import ForceField, Molecule
from openff.units import unit as off_unit

from presto.msm import (
    AngleParams,
    AngleScalingCalculator,
    BondParams,
    HessianDecomposer,
    _calculate_linear_angle_force_constant,
    _dot_product,
    _get_arbitrary_perpendicular,
    _is_linear_angle,
    _mean_angle_params,
    _mean_bond_params,
    apply_msm_to_molecule,
    apply_msm_to_molecules,
    calculate_angle_force_constant,
    calculate_angle_params,
    calculate_bond_force_constant,
    calculate_bond_params,
    compute_perpendicular_vector_in_plane,
    unit_vector_along_bond,
    unit_vector_normal_to_plane,
)
from presto.settings import MLPSettings, MSMSettings

# Unit constants for testing (kcal/mol is the standard energy unit throughout)
_BOND_K_UNIT = off_unit.kilocalorie_per_mole / off_unit.nanometer**2
_BOND_LENGTH_UNIT = off_unit.nanometer
_ANGLE_K_UNIT = off_unit.kilocalorie_per_mole / off_unit.radian**2
_ANGLE_UNIT = off_unit.radian

# --- Fixtures ---


@pytest.fixture
def water_coords():
    """Coordinates for a water-like molecule with ~104 degree angle.

    Note: Coordinates are in nm to match the internal unit convention.
    """
    # O at origin, H atoms at ~104 degrees
    return np.array(
        [
            [0.0, 0.0, 0.0],  # O
            [0.09572, 0.0, 0.0],  # H1 at 0.9572 Å (typical O-H)
            [-0.02399, 0.09270, 0.0],  # H2 to give ~104 degrees
        ]
    )


@pytest.fixture
def ethanol_coords():
    """Approximate coordinates for ethanol molecule.

    Note: Coordinates are in nm to match the internal unit convention.
    """
    # Rough ethanol geometry (9 atoms: C-C-O with 6 H)
    return np.array(
        [
            [0.0, 0.0, 0.0],  # C1
            [0.154, 0.0, 0.0],  # C2 (in nm)
            [0.216, 0.122, 0.0],  # O
            [-0.036, -0.051, -0.089],  # H1
            [-0.036, -0.051, 0.089],  # H2
            [-0.036, 0.102, 0.0],  # H3
            [0.19, -0.051, 0.089],  # H4
            [0.19, -0.051, -0.089],  # H5
            [0.31, 0.11, 0.0],  # H6 (on O)
        ]
    )


@pytest.fixture
def simple_3x3_hessian():
    """Simple 3x3 Hessian for a 1-atom system (for basic tests)."""
    return np.array(
        [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0],
        ]
    )


@pytest.fixture
def two_atom_hessian():
    """Simple 6x6 Hessian for a 2-atom diatomic system."""
    # Block structure for a simple spring along x-axis
    k = 500.0  # Force constant
    hessian = np.zeros((6, 6))
    # On-diagonal blocks
    hessian[0:3, 0:3] = np.diag([k, 0.0, 0.0])
    hessian[3:6, 3:6] = np.diag([k, 0.0, 0.0])
    # Off-diagonal blocks (coupling)
    hessian[0:3, 3:6] = np.diag([-k, 0.0, 0.0])
    hessian[3:6, 0:3] = np.diag([-k, 0.0, 0.0])
    return hessian


@pytest.fixture
def two_atom_coords():
    """Coordinates for a diatomic molecule along x-axis.

    Note: Coordinates are in nm to match the internal unit convention
    used by the MSM functions (Hessian is in kcal/mol/nm^2).
    """
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],  # Bond length 0.1 nm (= 1.0 Å)
        ]
    )


@pytest.fixture(scope="module")
def base_forcefield():
    """Shared OpenFF force field instance for MSM integration tests."""
    return ForceField("openff_unconstrained-2.3.0.offxml")


@pytest.fixture(scope="module")
def msm_settings():
    """Default MSM settings for testing with minimal conformer count."""
    return MSMSettings(
        n_conformers=1,
        mlp_settings=MLPSettings(ml_potential="aimnet2"),
    )


# --- Unit Vector Tests ---


class TestUnitVectorAlongBond:
    """Tests for unit_vector_along_bond function."""

    def test_unit_vector_x_axis(self, two_atom_coords):
        """Test unit vector along x-axis."""
        u = unit_vector_along_bond(two_atom_coords, 0, 1)
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(u, expected, atol=1e-10)

    def test_unit_vector_reversed(self, two_atom_coords):
        """Test unit vector is reversed when atom order is swapped."""
        u_forward = unit_vector_along_bond(two_atom_coords, 0, 1)
        u_backward = unit_vector_along_bond(two_atom_coords, 1, 0)
        np.testing.assert_allclose(u_forward, -u_backward, atol=1e-10)

    def test_unit_vector_is_normalized(self, water_coords):
        """Test that returned vector has unit length."""
        u = unit_vector_along_bond(water_coords, 0, 1)
        assert np.abs(np.linalg.norm(u) - 1.0) < 1e-10

    def test_diagonal_vector(self):
        """Test unit vector along a diagonal."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        u = unit_vector_along_bond(coords, 0, 1)
        expected = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        np.testing.assert_allclose(u, expected, atol=1e-10)


class TestUnitVectorNormalToPlane:
    """Tests for unit_vector_normal_to_plane function."""

    def test_perpendicular_to_xy_plane(self):
        """Test normal to xy plane is along z."""
        u_x = np.array([1.0, 0.0, 0.0])
        u_y = np.array([0.0, 1.0, 0.0])
        u_n = unit_vector_normal_to_plane(u_x, u_y)
        # Should be +/- z
        assert np.abs(np.abs(u_n[2]) - 1.0) < 1e-10
        assert np.abs(u_n[0]) < 1e-10
        assert np.abs(u_n[1]) < 1e-10

    def test_is_normalized(self):
        """Test that normal vector is normalized."""
        u1 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        u2 = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
        u_n = unit_vector_normal_to_plane(u1, u2)
        assert np.abs(np.linalg.norm(u_n) - 1.0) < 1e-10

    def test_parallel_vectors_returns_perpendicular(self):
        """Test handling of parallel vectors."""
        u = np.array([1.0, 0.0, 0.0])
        u_n = unit_vector_normal_to_plane(u, u)
        # Should return arbitrary perpendicular, still normalized
        assert np.abs(np.linalg.norm(u_n) - 1.0) < 1e-10
        # Should be perpendicular to original
        assert np.abs(np.dot(u_n, u)) < 1e-10


class TestGetArbitraryPerpendicular:
    """Tests for _get_arbitrary_perpendicular function."""

    def test_perpendicular_to_x_axis(self):
        """Test perpendicular to x-axis."""
        u = np.array([1.0, 0.0, 0.0])
        perp = _get_arbitrary_perpendicular(u)
        assert np.abs(np.dot(perp, u)) < 1e-10
        assert np.abs(np.linalg.norm(perp) - 1.0) < 1e-10

    def test_perpendicular_to_y_axis(self):
        """Test perpendicular to y-axis."""
        u = np.array([0.0, 1.0, 0.0])
        perp = _get_arbitrary_perpendicular(u)
        assert np.abs(np.dot(perp, u)) < 1e-10
        assert np.abs(np.linalg.norm(perp) - 1.0) < 1e-10

    def test_perpendicular_to_diagonal(self):
        """Test perpendicular to diagonal vector."""
        u = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        perp = _get_arbitrary_perpendicular(u)
        assert np.abs(np.dot(perp, u)) < 1e-10
        assert np.abs(np.linalg.norm(perp) - 1.0) < 1e-10


class TestComputePerpendicularVectorInPlane:
    """Tests for compute_perpendicular_vector_in_plane function."""

    def test_water_angle(self, water_coords):
        """Test perpendicular vector for water H-O-H angle."""
        angle = (1, 0, 2)  # H1-O-H2
        u_pa = compute_perpendicular_vector_in_plane(water_coords, angle)
        # Should be perpendicular to the O-H1 bond
        u_oh1 = unit_vector_along_bond(water_coords, 0, 1)
        assert np.abs(np.dot(u_pa, u_oh1)) < 1e-10
        # Should be normalized
        assert np.abs(np.linalg.norm(u_pa) - 1.0) < 1e-10

    def test_linear_angle(self):
        """Test perpendicular vector for near-linear angle."""
        # Nearly linear A-B-C (180 degrees)
        coords = np.array(
            [
                [0.0, 0.0, 0.0],  # A
                [1.0, 0.0, 0.0],  # B (central)
                [2.0, 0.001, 0.0],  # C (slightly off-axis)
            ]
        )
        angle = (0, 1, 2)
        u_pa = compute_perpendicular_vector_in_plane(coords, angle)
        # Should still be normalized
        assert np.abs(np.linalg.norm(u_pa) - 1.0) < 1e-10


# --- Dot Product Tests ---


class TestDotProduct:
    """Tests for _dot_product function."""

    def test_real_vectors(self):
        """Test dot product with real vectors."""
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.5, 0.5, 0.0])
        result = _dot_product(u, v)
        assert np.abs(result - 0.5) < 1e-10

    def test_complex_eigenvector(self):
        """Test dot product with complex eigenvector."""
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([1.0 + 1.0j, 0.0, 0.0])
        result = _dot_product(u, v)
        # Conjugate: 1 - 1j, so result should be 1 - 1j
        expected = 1.0 - 1.0j
        assert np.abs(result - expected) < 1e-10

    def test_orthogonal_vectors(self):
        """Test dot product of orthogonal vectors."""
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        result = _dot_product(u, v)
        assert np.abs(result) < 1e-10


# --- HessianDecomposer Tests ---


class TestHessianDecomposer:
    """Tests for HessianDecomposer class."""

    def test_decomposer_initialization(self, two_atom_hessian, two_atom_coords):
        """Test HessianDecomposer initializes correctly."""
        decomposer = HessianDecomposer(two_atom_hessian, two_atom_coords)
        assert decomposer.n_atoms == 2
        assert decomposer.eigenvals.shape == (2, 2, 3)
        assert decomposer.eigenvecs.shape == (3, 3, 2, 2)
        assert decomposer.bond_lengths.shape == (2, 2)

    def test_bond_lengths_computed(self, two_atom_hessian, two_atom_coords):
        """Test that bond lengths are computed correctly."""
        decomposer = HessianDecomposer(two_atom_hessian, two_atom_coords)
        # Bond length between atoms 0 and 1 should be 0.1 nm (coordinates are in nm)
        np.testing.assert_allclose(decomposer.bond_lengths[0, 1], 0.1, atol=1e-10)
        np.testing.assert_allclose(decomposer.bond_lengths[1, 0], 0.1, atol=1e-10)
        # Self distances should be 0
        assert np.abs(decomposer.bond_lengths[0, 0]) < 1e-10

    def test_eigenvalues_computed(self, two_atom_hessian, two_atom_coords):
        """Test that eigenvalues are computed."""
        decomposer = HessianDecomposer(two_atom_hessian, two_atom_coords)
        # The off-diagonal block [0,1] has [-k, 0, 0] on diagonal
        # Eigenvalues should be -k, 0, 0 (k=500)
        eigenvals = decomposer.eigenvals[0, 1, :]
        sorted_eigenvals = np.sort(np.real(eigenvals))
        assert np.abs(sorted_eigenvals[0] - (-500.0)) < 1e-10
        assert np.abs(sorted_eigenvals[1]) < 1e-10
        assert np.abs(sorted_eigenvals[2]) < 1e-10


# --- AngleScalingCalculator Tests ---


class TestAngleScalingCalculator:
    """Tests for AngleScalingCalculator class."""

    def test_single_angle(self, water_coords):
        """Test scaling factors for a single angle."""
        angles = [(1, 0, 2)]  # H1-O-H2
        calculator = AngleScalingCalculator(angles, water_coords, n_atoms=3)
        scalings = calculator.compute_scaling_factors()
        # Single angle with no neighbors should have scaling = (1.0, 1.0)
        assert len(scalings) == 1
        assert np.abs(scalings[0][0] - 1.0) < 1e-10
        assert np.abs(scalings[0][1] - 1.0) < 1e-10

    def test_multiple_angles_same_central(self, ethanol_coords):
        """Test scaling factors when multiple angles share a central atom."""
        # Create angles sharing C2 (index 1) as central
        angles = [
            (0, 1, 2),  # C1-C2-O
            (0, 1, 6),  # C1-C2-H4
            (0, 1, 7),  # C1-C2-H5
        ]
        calculator = AngleScalingCalculator(angles, ethanol_coords, n_atoms=9)
        scalings = calculator.compute_scaling_factors()
        # Should have scaling factors > 1 for angles sharing bonds
        assert len(scalings) == 3
        # All angles share C1-C2 bond, so first scaling should be > 1
        assert all(s[0] >= 1.0 for s in scalings)


# --- Bond Force Constant Tests ---


class TestCalculateBondForceConstant:
    """Tests for calculate_bond_force_constant function."""

    def test_simple_diatomic(self, two_atom_hessian, two_atom_coords):
        """Test force constant for simple diatomic."""
        decomposer = HessianDecomposer(two_atom_hessian, two_atom_coords)
        bond = (0, 1)
        k = calculate_bond_force_constant(
            bond,
            decomposer.eigenvals,
            decomposer.eigenvecs,
            decomposer.coords,
        )
        # Force constant should be positive
        assert k > 0

    def test_symmetric_bond(self, two_atom_hessian, two_atom_coords):
        """Test that force constant is similar in both directions."""
        decomposer = HessianDecomposer(two_atom_hessian, two_atom_coords)
        k_01 = calculate_bond_force_constant(
            (0, 1),
            decomposer.eigenvals,
            decomposer.eigenvecs,
            decomposer.coords,
        )
        k_10 = calculate_bond_force_constant(
            (1, 0),
            decomposer.eigenvals,
            decomposer.eigenvecs,
            decomposer.coords,
        )
        # Should be similar (symmetric Hessian)
        np.testing.assert_allclose(k_01, k_10, rtol=0.1)


# --- Angle Force Constant Tests ---


class TestCalculateAngleForceConstant:
    """Tests for calculate_angle_force_constant function."""

    def test_returns_positive_force_constant(self, water_coords):
        """Test that force constant is positive."""
        # Create a mock Hessian with negative eigenvalues (as expected from bonded atoms)
        # For angle bending, we need negative eigenvalues in the off-diagonal blocks
        n_atoms = 3
        hessian = np.zeros((9, 9))
        # Fill with realistic spring-like values
        k_val = -500.0  # Negative for off-diagonal blocks
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    hessian[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3] = np.diag(
                        [k_val, k_val, k_val]
                    )
                else:
                    # Positive on diagonal blocks
                    hessian[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3] = np.diag(
                        [-k_val * 2, -k_val * 2, -k_val * 2]
                    )
        decomposer = HessianDecomposer(hessian, water_coords)

        angle = (1, 0, 2)  # H1-O-H2
        scalings = (1.0, 1.0)
        k_theta, _theta_0 = calculate_angle_force_constant(
            angle,
            decomposer.bond_lengths,
            decomposer.eigenvals,
            decomposer.eigenvecs,
            decomposer.coords,
            scalings,
        )
        assert k_theta > 0

    def test_equilibrium_angle_reasonable(self, water_coords):
        """Test that equilibrium angle is reasonable."""
        n_atoms = 3
        hessian = np.zeros((9, 9))
        k_val = -500.0
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    hessian[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3] = np.diag(
                        [k_val, k_val, k_val]
                    )
                else:
                    hessian[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3] = np.diag(
                        [-k_val * 2, -k_val * 2, -k_val * 2]
                    )
        decomposer = HessianDecomposer(hessian, water_coords)

        angle = (1, 0, 2)
        scalings = (1.0, 1.0)
        _k_theta, theta_0 = calculate_angle_force_constant(
            angle,
            decomposer.bond_lengths,
            decomposer.eigenvals,
            decomposer.eigenvecs,
            decomposer.coords,
            scalings,
        )
        # Water angle should be around 104 degrees, our mock coords
        # give approximately this
        assert 90.0 < theta_0 < 120.0


class TestCalculateLinearAngleForceConstant:
    """Tests for _calculate_linear_angle_force_constant function."""

    @pytest.fixture
    def linear_inputs(self):
        """Return a simple exactly linear case with an analytic result."""
        # Linear configuration along x
        u_ab = np.array([1.0, 0.0, 0.0])
        u_cb = np.array([-1.0, 0.0, 0.0])  # Opposite direction (linear)
        bond_lens = (1.0, 1.0)
        # Simple eigenvalues/eigenvectors
        eigenvals = (
            np.array([100.0, 50.0, 50.0]),
            np.array([100.0, 50.0, 50.0]),
        )
        eigenvecs = (np.eye(3, dtype=complex), np.eye(3, dtype=complex))
        return u_ab, u_cb, bond_lens, eigenvals, eigenvecs

    def test_exact_linear_angle_has_analytic_force_constant(self, linear_inputs):
        """The one-direction two-spring result is 1 / (1/50 + 1/50) = 25."""
        k_theta, theta_0 = _calculate_linear_angle_force_constant(
            *linear_inputs, n_samples=1
        )
        assert k_theta == pytest.approx(25.0)
        assert theta_0 == pytest.approx(180.0)

    def test_matches_qubekit_final_linear_force_constant(self, linear_inputs):
        """Compare with QUBEKit's published 200-direction special case.

        QUBEKit's helper returns half the projected curvature and its caller
        multiplies the value by two when constructing the final angle
        parameter. The reference below is that final exported value, evaluated
        with QUBEKit's integer-radian sampling grid. Presto uses a uniformly
        spaced grid, which accounts for the small numerical tolerance.

        Source: QUBEKit 2.1.1, ``ModSemMaths.f_c_a_special_case`` and
        ``ModSeminario.calculate_angles``.
        """
        u_ab, _u_cb, bond_lens, eigenvals, eigenvecs = linear_inputs
        angle = np.deg2rad(175.0)
        u_cb = np.array([np.cos(angle), np.sin(angle), 0.0])
        k_theta, theta_0 = _calculate_linear_angle_force_constant(
            u_ab, u_cb, bond_lens, eigenvals, eigenvecs
        )

        qubekit_reference = _QUBEKIT_REFERENCE_DATA["near_linear_angle_reference"]
        qubekit_final_k = qubekit_reference["final_k_kcal_mol_rad2"]
        assert qubekit_final_k == pytest.approx(
            2.0 * qubekit_reference["helper_k_kcal_mol_rad2"]
        )
        np.testing.assert_allclose(k_theta, qubekit_final_k, rtol=5e-4)
        assert theta_0 == pytest.approx(175.0)

    def test_production_dispatches_exact_linear_angle_to_special_case(self):
        """Exercise the production branch rather than only its helper."""
        coords = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        decomposer = HessianDecomposer(create_mock_hessian(3), coords)

        with patch(
            "presto.msm._calculate_linear_angle_force_constant",
            wraps=_calculate_linear_angle_force_constant,
        ) as linear_fn:
            calculate_angle_force_constant(
                (0, 1, 2),
                decomposer.bond_lengths,
                decomposer.eigenvals,
                decomposer.eigenvecs,
                decomposer.coords,
                (1.0, 1.0),
            )

        linear_fn.assert_called_once()

    def test_linear_angle_predicate_boundaries(self):
        """Keep the audit and production definition of a linear angle identical."""
        assert _is_linear_angle(np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]))
        assert not _is_linear_angle(
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        )


# --- Parameter Calculation Tests ---


class TestCalculateBondParams:
    """Tests for calculate_bond_params function."""

    def test_returns_dict_of_bond_params(self, two_atom_hessian, two_atom_coords):
        """Test that function returns dictionary of BondParams."""
        decomposer = HessianDecomposer(two_atom_hessian, two_atom_coords)
        bond_indices = [(0, 1)]
        vib_scaling = 1.0

        result = calculate_bond_params(bond_indices, decomposer, vib_scaling)

        assert isinstance(result, dict)
        assert (0, 1) in result
        assert isinstance(result[(0, 1)], BondParams)

    def test_force_constant_scaled(self, two_atom_hessian, two_atom_coords):
        """Test that vibrational scaling is applied."""
        decomposer = HessianDecomposer(two_atom_hessian, two_atom_coords)
        bond_indices = [(0, 1)]

        result_1 = calculate_bond_params(bond_indices, decomposer, vib_scaling=1.0)
        result_2 = calculate_bond_params(bond_indices, decomposer, vib_scaling=0.5)

        # With scaling 0.5, force constant should be 0.25x
        ratio = result_2[(0, 1)].force_constant / result_1[(0, 1)].force_constant
        np.testing.assert_allclose(ratio, 0.25, rtol=0.01)

    def test_bond_length_preserved(self, two_atom_hessian, two_atom_coords):
        """Test that bond length is correctly reported in nm."""
        decomposer = HessianDecomposer(two_atom_hessian, two_atom_coords)
        bond_indices = [(0, 1)]

        result = calculate_bond_params(bond_indices, decomposer, vib_scaling=1.0)

        # Bond length is 1.0 Å = 0.1 nm
        np.testing.assert_allclose(
            result[(0, 1)].length.m_as(_BOND_LENGTH_UNIT), 0.1, rtol=0.01
        )


def create_mock_hessian(
    n_atoms: int,
    k_diagonal: float = 500.0,
    units: str = "kcal_mol_nm2",
) -> np.ndarray:
    """Create a mock Hessian matrix for testing.

    Creates a diagonal-dominated Hessian that represents harmonic restoring forces.
    This is the canonical mock Hessian generator used across all MSM tests.

    The base k_diagonal is specified in kcal/mol/Å² units (a common QM output unit),
    then converted to the requested output units.

    Args:
        n_atoms: Number of atoms
        k_diagonal: Force constant for diagonal elements in kcal/mol/Å²
        units: Output unit system. Options:
            - "kcal_mol_nm2": kcal/mol/nm² (default, for MSM functions - internal units)
            - "kj_mol_nm2": kJ/mol/nm²
            - "kcal_mol_A2": kcal/mol/Å² (raw QM-like units)
            - "hartree_bohr2": Hartree/Bohr² (atomic units, for QUBEKit)

    Returns:
        Hessian matrix of shape (3*n_atoms, 3*n_atoms) in requested units
    """
    KCAL_TO_KJ = 4.184

    size = 3 * n_atoms
    hessian = np.zeros((size, size))

    # Set diagonal blocks (self-interaction)
    for i in range(n_atoms):
        block = np.diag([k_diagonal, k_diagonal, k_diagonal])
        hessian[i * 3 : (i + 1) * 3, i * 3 : (i + 1) * 3] = block

    # Set off-diagonal blocks (interactions between atoms)
    k_coupling = -k_diagonal / (n_atoms - 1)
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                hessian[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3] = np.diag(
                    [k_coupling, k_coupling, k_coupling]
                )

    # Ensure symmetry
    hessian = 0.5 * (hessian + hessian.T)

    # Convert to requested units (base is kcal/mol/Å²)
    if units == "kcal_mol_A2":
        # Already in kcal/mol/Å²
        return hessian
    elif units == "kcal_mol_nm2":
        # 1 nm = 10 Å, so 1/nm² = 1/100 Å²
        # k [kcal/mol/nm²] = k [kcal/mol/Å²] * 100
        return hessian * 100.0
    elif units == "kj_mol_nm2":
        # Convert kcal→kJ and Å²→nm²
        # k [kJ/mol/nm²] = k [kcal/mol/Å²] * 100 * 4.184
        return hessian * 100.0 * KCAL_TO_KJ
    elif units == "hartree_bohr2":
        # QUBEKit conversion: hessian *= HA_TO_KCAL_P_MOL / BOHR_TO_ANGS²
        # So reverse: hessian_au = hessian_kcal_A2 / (HA_TO_KCAL_P_MOL / BOHR_TO_ANGS²)
        HA_TO_KCAL_P_MOL = 627.509474  # From QUBEKit constants
        BOHR_TO_ANGS = 0.529177249
        conversion = HA_TO_KCAL_P_MOL / (BOHR_TO_ANGS**2)
        return hessian / conversion
    else:
        raise ValueError(f"Unknown units: {units}")


class TestCalculateAngleParams:
    """Tests for calculate_angle_params function."""

    def test_returns_dict_of_angle_params(self, water_coords):
        """Test that function returns dictionary of AngleParams."""
        hessian = create_mock_hessian(3)  # Uses default kcal_mol_nm2
        decomposer = HessianDecomposer(hessian, water_coords)
        angle_indices = [(1, 0, 2)]
        vib_scaling = 1.0

        result = calculate_angle_params(angle_indices, decomposer, vib_scaling)

        assert isinstance(result, dict)
        assert (1, 0, 2) in result
        assert isinstance(result[(1, 0, 2)], AngleParams)

    def test_empty_angles_returns_empty_dict(self, water_coords):
        """Test that empty angle list returns empty dict."""
        hessian = create_mock_hessian(3)  # Uses default kcal_mol_nm2
        decomposer = HessianDecomposer(hessian, water_coords)

        result = calculate_angle_params([], decomposer, vib_scaling=1.0)

        assert result == {}

    def test_angle_in_radians(self, water_coords):
        """Test that angle is returned in radians."""
        hessian = create_mock_hessian(3)  # Uses default kcal_mol_nm2
        decomposer = HessianDecomposer(hessian, water_coords)
        angle_indices = [(1, 0, 2)]

        result = calculate_angle_params(angle_indices, decomposer, vib_scaling=1.0)

        # Angle should be in radians (water angle ~104 deg = ~1.8 rad)
        angle_rad = result[(1, 0, 2)].angle.m_as(_ANGLE_UNIT)
        assert 1.5 < angle_rad < 2.1


# --- Mean Parameter Tests ---


class TestMeanBondParams:
    """Tests for _mean_bond_params function."""

    def test_single_param(self):
        """Test mean of single parameter."""
        params = [BondParams.from_values(force_constant=1000.0, length=0.15)]
        result = _mean_bond_params(params)
        np.testing.assert_allclose(
            result.force_constant.m_as(_BOND_K_UNIT), 1000.0, rtol=1e-10
        )
        np.testing.assert_allclose(
            result.length.m_as(_BOND_LENGTH_UNIT), 0.15, rtol=1e-10
        )

    def test_multiple_params(self):
        """Test mean of multiple parameters."""
        params = [
            BondParams.from_values(force_constant=1000.0, length=0.14),
            BondParams.from_values(force_constant=2000.0, length=0.16),
        ]
        result = _mean_bond_params(params)
        np.testing.assert_allclose(
            result.force_constant.m_as(_BOND_K_UNIT), 1500.0, rtol=1e-10
        )
        np.testing.assert_allclose(
            result.length.m_as(_BOND_LENGTH_UNIT), 0.15, rtol=1e-10
        )

    def test_empty_raises_error(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="No bond parameters"):
            _mean_bond_params([])


class TestMeanAngleParams:
    """Tests for _mean_angle_params function."""

    def test_single_param(self):
        """Test mean of single parameter."""
        params = [AngleParams.from_values(force_constant=500.0, angle=1.9)]
        result = _mean_angle_params(params)
        np.testing.assert_allclose(
            result.force_constant.m_as(_ANGLE_K_UNIT), 500.0, rtol=1e-10
        )
        np.testing.assert_allclose(result.angle.m_as(_ANGLE_UNIT), 1.9, rtol=1e-10)

    def test_multiple_params(self):
        """Test mean of multiple parameters."""
        params = [
            AngleParams.from_values(force_constant=400.0, angle=1.8),
            AngleParams.from_values(force_constant=600.0, angle=2.0),
        ]
        result = _mean_angle_params(params)
        np.testing.assert_allclose(
            result.force_constant.m_as(_ANGLE_K_UNIT), 500.0, rtol=1e-10
        )
        np.testing.assert_allclose(result.angle.m_as(_ANGLE_UNIT), 1.9, rtol=1e-10)

    def test_empty_raises_error(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="No angle parameters"):
            _mean_angle_params([])


# --- Dataclass Tests ---


class TestBondParams:
    """Tests for BondParams dataclass."""

    def test_creation_from_values(self):
        """Test BondParams creation using from_values factory method."""
        params = BondParams.from_values(force_constant=1000.0, length=0.15)
        np.testing.assert_allclose(
            params.force_constant.m_as(_BOND_K_UNIT), 1000.0, rtol=1e-10
        )
        np.testing.assert_allclose(
            params.length.m_as(_BOND_LENGTH_UNIT), 0.15, rtol=1e-10
        )

    def test_has_units(self):
        """Test that BondParams has proper units."""
        params = BondParams.from_values(force_constant=1000.0, length=0.15)
        # Check that values are Quantity objects with correct units
        assert params.force_constant.is_compatible_with(_BOND_K_UNIT)
        assert params.length.is_compatible_with(_BOND_LENGTH_UNIT)

    def test_equality(self):
        """Test BondParams equality."""
        p1 = BondParams.from_values(force_constant=1000.0, length=0.15)
        p2 = BondParams.from_values(force_constant=1000.0, length=0.15)
        assert p1 == p2


class TestAngleParams:
    """Tests for AngleParams dataclass."""

    def test_creation_from_values(self):
        """Test AngleParams creation using from_values factory method."""
        params = AngleParams.from_values(force_constant=500.0, angle=1.9)
        np.testing.assert_allclose(
            params.force_constant.m_as(_ANGLE_K_UNIT), 500.0, rtol=1e-10
        )
        np.testing.assert_allclose(params.angle.m_as(_ANGLE_UNIT), 1.9, rtol=1e-10)

    def test_has_units(self):
        """Test that AngleParams has proper units."""
        params = AngleParams.from_values(force_constant=500.0, angle=1.9)
        # Check that values are Quantity objects with correct units
        assert params.force_constant.is_compatible_with(_ANGLE_K_UNIT)
        assert params.angle.is_compatible_with(_ANGLE_UNIT)

    def test_equality(self):
        """Test AngleParams equality."""
        p1 = AngleParams.from_values(force_constant=500.0, angle=1.9)
        p2 = AngleParams.from_values(force_constant=500.0, angle=1.9)
        assert p1 == p2


# --- Integration Tests (marked as slow) ---


@pytest.mark.slow
class TestApplyMSMToMolecule:
    """Integration tests for apply_msm_to_molecule function."""

    def test_ethanol_molecule(self, msm_settings, base_forcefield):
        """Test MSM on ethanol molecule."""
        mol = Molecule.from_smiles("CCO")
        ff = base_forcefield

        # Get bond and angle indices from force field
        labels = ff.label_molecules(mol.to_topology())[0]
        bond_indices = list(labels["Bonds"].keys())
        angle_indices = list(labels["Angles"].keys())

        bond_params, angle_params = apply_msm_to_molecule(
            mol, bond_indices, angle_indices, msm_settings, device=torch.device("cpu")
        )

        # Check that we got parameters for all bonds and angles
        assert len(bond_params) == len(bond_indices)
        assert len(angle_params) == len(angle_indices)

        # Check all force constants are positive (Quantity objects support > comparison)
        for bp in bond_params.values():
            assert bp.force_constant.magnitude > 0
            assert bp.length.magnitude > 0

        for ap in angle_params.values():
            assert ap.force_constant.magnitude > 0
            assert 0 < ap.angle.m_as(_ANGLE_UNIT) < np.pi

    def test_returns_correct_types(self, msm_settings, base_forcefield):
        """Test that return types are correct."""
        mol = Molecule.from_smiles("C")  # Methane - simple
        ff = base_forcefield
        labels = ff.label_molecules(mol.to_topology())[0]
        bond_indices = list(labels["Bonds"].keys())
        angle_indices = list(labels["Angles"].keys())

        bond_params, angle_params = apply_msm_to_molecule(
            mol, bond_indices, angle_indices, msm_settings, device=torch.device("cpu")
        )

        assert isinstance(bond_params, dict)
        assert isinstance(angle_params, dict)
        for k, v in bond_params.items():
            assert isinstance(k, tuple)
            assert isinstance(v, BondParams)

    def test_multiple_conformers(self, base_forcefield):
        """Test MSM with multiple conformers averages parameters."""
        mol = Molecule.from_smiles("CCO")
        ff = base_forcefield
        labels = ff.label_molecules(mol.to_topology())[0]
        bond_indices = list(labels["Bonds"].keys())
        angle_indices = list(labels["Angles"].keys())

        # Test with 3 conformers
        settings_multi = MSMSettings(
            n_conformers=2,
            mlp_settings=MLPSettings(ml_potential="aimnet2"),
        )

        bond_params_multi, angle_params_multi = apply_msm_to_molecule(
            mol, bond_indices, angle_indices, settings_multi, device=torch.device("cpu")
        )

        # Check that we got parameters for all bonds and angles
        assert len(bond_params_multi) == len(bond_indices)
        assert len(angle_params_multi) == len(angle_indices)

        # Check all force constants are positive
        for bp in bond_params_multi.values():
            assert bp.force_constant.magnitude > 0
            assert bp.length.magnitude > 0

        for ap in angle_params_multi.values():
            assert ap.force_constant.magnitude > 0
            assert 0 < ap.angle.m_as(_ANGLE_UNIT) < np.pi

    def test_supplied_starting_conformers(
        self, base_forcefield, tmp_path, write_multiconformer_sdf
    ):
        """MSM runs from a supplied conformer SDF, iterating the actual conformers."""
        import presto.msm as msm_module

        n_supplied = 3
        source = Molecule.from_smiles("CCO")
        source.generate_conformers(
            n_conformers=n_supplied, rms_cutoff=0.0 * off_unit.angstrom
        )
        assert source.n_conformers == n_supplied
        sdf = tmp_path / "confs.sdf"
        write_multiconformer_sdf(source, sdf)

        mol = Molecule.from_smiles("CCO")
        ff = base_forcefield
        labels = ff.label_molecules(mol.to_topology())[0]
        bond_indices = list(labels["Bonds"].keys())
        angle_indices = list(labels["Angles"].keys())

        # n_conformers deliberately differs from the file's conformer count. The Hessian
        # (computed once per conformer) is spied on to prove the loop iterates every
        # supplied conformer rather than n_conformers of them.
        settings = MSMSettings(
            n_conformers=1,
            mlp_settings=MLPSettings(ml_potential="aimnet2"),
            starting_conformers=sdf,
        )

        real_calculate_hessian = msm_module.calculate_hessian
        with patch.object(
            msm_module, "calculate_hessian", side_effect=real_calculate_hessian
        ) as spy_hessian:
            bond_params, angle_params = apply_msm_to_molecule(
                mol, bond_indices, angle_indices, settings, device=torch.device("cpu")
            )

        # One Hessian per supplied conformer, independent of n_conformers=1.
        assert spy_hessian.call_count == n_supplied
        assert len(bond_params) == len(bond_indices)
        assert len(angle_params) == len(angle_indices)


def test_apply_msm_to_molecule_cleans_up_on_failure():
    """A failing molecule releases its simulation, so the next molecule has its GPU."""
    import presto.msm as msm_module

    mol = Molecule.from_smiles("CCO")
    simulation = MagicMock()
    simulation.minimizeEnergy.side_effect = RuntimeError("CUDA out of memory")
    settings = MSMSettings(
        n_conformers=1, mlp_settings=MLPSettings(ml_potential="aimnet2")
    )

    with (
        patch.object(
            msm_module,
            "_build_ml_simulation",
            return_value=(simulation, sentinel.integrator),
        ),
        patch.object(msm_module, "cleanup_simulation") as cleanup,
        pytest.raises(RuntimeError, match="CUDA out of memory"),
    ):
        apply_msm_to_molecule(mol, [], [], settings, device=torch.device("cpu"))

    cleanup.assert_called_once_with(simulation, sentinel.integrator)


def test_apply_msm_to_molecules_collects_failures(base_forcefield, msm_settings):
    """A molecule MSM cannot handle is collected, not raised, so all failures surface."""
    mols = [Molecule.from_smiles("CCO"), Molecule.from_smiles("CC")]

    def fake_apply(mol, bond_indices, angle_indices, settings, device):
        if mol.n_atoms == mols[1].n_atoms:
            raise ValueError("RDKit conformer generation failed.")
        return {}, {}

    with patch("presto.msm.apply_msm_to_molecule", side_effect=fake_apply):
        _modified_ff, failures = apply_msm_to_molecules(
            mols, base_forcefield, msm_settings, device=torch.device("cpu")
        )

    assert list(failures) == [1]
    assert "RDKit conformer generation failed." in failures[1]


@pytest.mark.slow
class TestApplyMSMToMolecules:
    """Integration tests for apply_msm_to_molecules function."""

    def test_single_molecule(self, msm_settings, base_forcefield):
        """Test MSM on single molecule."""
        mol = Molecule.from_smiles("CCO")
        ff = base_forcefield

        modified_ff, _failures = apply_msm_to_molecules(
            [mol], ff, msm_settings, device=torch.device("cpu")
        )

        assert isinstance(modified_ff, ForceField)

    def test_multiple_molecules(self, msm_settings, base_forcefield):
        """Test MSM on multiple molecules."""
        mols = [
            Molecule.from_smiles("CCO"),  # Ethanol
            Molecule.from_smiles("CC"),  # Ethane
        ]
        ff = base_forcefield

        modified_ff, _failures = apply_msm_to_molecules(
            mols, ff, msm_settings, device=torch.device("cpu")
        )

        assert isinstance(modified_ff, ForceField)

    def test_force_field_not_modified_in_place(self, msm_settings, base_forcefield):
        """Test that original force field is not modified."""
        mol = Molecule.from_smiles("C")
        ff = base_forcefield

        # Get original bond parameter
        bond_handler = ff.get_parameter_handler("Bonds")
        original_params = [(p.smirks, p.k, p.length) for p in bond_handler.parameters]

        _modified_ff, _failures = apply_msm_to_molecules(
            [mol], ff, msm_settings, device=torch.device("cpu")
        )

        # Check original is unchanged
        new_params = [(p.smirks, p.k, p.length) for p in bond_handler.parameters]
        assert original_params == new_params

    def test_modified_ff_has_different_params(self, msm_settings, base_forcefield):
        """Test that modified force field has different parameters."""
        mol = Molecule.from_smiles("CCO")
        ff = base_forcefield

        # Get SMIRKS patterns used by the molecule
        labels = ff.label_molecules(mol.to_topology())[0]
        used_bond_smirks = {p.smirks for p in labels["Bonds"].values()}

        # Get original parameters
        bond_handler = ff.get_parameter_handler("Bonds")
        original_k = {
            p.smirks: p.k
            for p in bond_handler.parameters
            if p.smirks in used_bond_smirks
        }

        modified_ff, _failures = apply_msm_to_molecules(
            [mol], ff, msm_settings, device=torch.device("cpu")
        )

        # Get modified parameters
        mod_bond_handler = modified_ff.get_parameter_handler("Bonds")
        modified_k = {
            p.smirks: p.k
            for p in mod_bond_handler.parameters
            if p.smirks in used_bond_smirks
        }

        # At least one parameter should be different
        # (MSM derives from MLP Hessian, different from original FF)
        different = any(original_k[s] != modified_k[s] for s in used_bond_smirks)
        assert different

    def test_units_preserved_from_original_ff(self, msm_settings, base_forcefield):
        """Test that units from the original force field are preserved.

        Note: The `.to()` method may format units differently (e.g.,
        'kilocalorie_per_mole / angstrom ** 2' vs 'kilocalorie / angstrom ** 2 / mole')
        even though they are equivalent. So we check unit compatibility rather than
        exact string equality.
        """
        mol = Molecule.from_smiles("CCO")
        ff = base_forcefield

        # Get original units
        bond_handler = ff.get_parameter_handler("Bonds")
        angle_handler = ff.get_parameter_handler("Angles")

        # Store original units for one parameter of each type
        original_bond_k_units = bond_handler.parameters[0].k.units
        original_bond_length_units = bond_handler.parameters[0].length.units
        original_angle_k_units = angle_handler.parameters[0].k.units
        original_angle_angle_units = angle_handler.parameters[0].angle.units

        modified_ff, _failures = apply_msm_to_molecules(
            [mol], ff, msm_settings, device=torch.device("cpu")
        )

        # Check units are compatible (dimensionally equivalent) in modified force field
        mod_bond_handler = modified_ff.get_parameter_handler("Bonds")
        mod_angle_handler = modified_ff.get_parameter_handler("Angles")

        for param in mod_bond_handler.parameters:
            # Check dimensional compatibility by attempting conversion
            assert param.k.is_compatible_with(original_bond_k_units), (
                f"Bond k units {param.k.units} not compatible with {original_bond_k_units}"
            )
            assert param.length.is_compatible_with(original_bond_length_units), (
                f"Bond length units {param.length.units} not compatible with {original_bond_length_units}"
            )

        for param in mod_angle_handler.parameters:
            assert param.k.is_compatible_with(original_angle_k_units), (
                f"Angle k units {param.k.units} not compatible with {original_angle_k_units}"
            )
            assert param.angle.is_compatible_with(original_angle_angle_units), (
                f"Angle units {param.angle.units} not compatible with {original_angle_angle_units}"
            )


# =============================================================================
# QUBEKit Reference Comparison Tests
# =============================================================================
#
# These tests compare the MSM implementation against QUBEKit's reference
# implementation.
#
# Reference values were generated using QUBEKit installed at:
#
# Test Molecule: Fluorochlorobromomethanol (OC(F)(Cl)Br)
# - 6 atoms: O, C, F, Cl, Br, H
# - 5 bonds (all unique): C-O, C-F, C-Cl, C-Br, O-H
# - 7 angles (all unique)
# - Fully asymmetric to avoid QUBEKit's internal symmetry averaging
# =============================================================================

# Load reference data from QUBEKit JSON file using importlib.resources
_QUBEKIT_REFERENCE_FILE = files("presto.data.msm") / "qubekit_reference_values.json"


def _load_qubekit_reference_data():
    """Load QUBEKit reference data, returning None if file not found."""
    try:
        return json.loads(_QUBEKIT_REFERENCE_FILE.read_text())
    except FileNotFoundError:
        return None


# Load reference data at module level
_QUBEKIT_REFERENCE_DATA = _load_qubekit_reference_data()


def _parse_reference_data():
    """Parse the reference JSON data into usable format."""
    if _QUBEKIT_REFERENCE_DATA is None:
        return None, None, None, None, None

    # Conversion factor from kJ/mol to kcal/mol
    KJ_TO_KCAL = 1.0 / 4.184

    # Coordinates in Angstroms (from QUBEKit/RDKit), converted to nm
    coords_angstrom = np.array(_QUBEKIT_REFERENCE_DATA["coordinates_angstrom"])
    coords_nm = coords_angstrom / 10.0

    # Bond and angle lists
    bonds = [tuple(b) for b in _QUBEKIT_REFERENCE_DATA["bonds"]]
    angles = [tuple(a) for a in _QUBEKIT_REFERENCE_DATA["angles"]]

    # QUBEKit bond parameters (SMIRNOFF / OpenMM convention: U = k*x^2/2)
    # Reference data is in kJ/mol/nm², convert to kcal/mol/nm²
    bond_params = {
        tuple(map(int, k.strip("()").split(", "))): (
            v["length_nm"],
            v["k_kj_mol_nm2"] * KJ_TO_KCAL,
        )
        for k, v in _QUBEKIT_REFERENCE_DATA["bond_params"].items()
    }

    # QUBEKit angle parameters (SMIRNOFF / OpenMM convention: U = k*x^2/2)
    # Reference data is in kJ/mol/rad², convert to kcal/mol/rad²
    angle_params = {
        tuple(map(int, k.strip("()").split(", "))): (
            v["angle_deg"],
            v["k_kj_mol_rad2"] * KJ_TO_KCAL,
        )
        for k, v in _QUBEKIT_REFERENCE_DATA["angle_params"].items()
    }

    return coords_nm, bonds, angles, bond_params, angle_params


# Parse reference data at module level
(
    _QUBEKIT_COORDS_NM,
    _QUBEKIT_BONDS,
    _QUBEKIT_ANGLES,
    _QUBEKIT_BOND_PARAMS,
    _QUBEKIT_ANGLE_PARAMS,
) = _parse_reference_data()


class TestMSMQubekitComparison:
    """Test MSM implementation against QUBEKit reference values.

    These tests verify that our MSM implementation produces the same results
    as QUBEKit's ModSeminario method for the same input Hessian and geometry.

    The reference values were generated using generate_qubekit_reference.py
    located in presto/data/msm/.
    """

    @pytest.fixture
    def decomposer(self):
        """Create HessianDecomposer for the test molecule."""
        # Create the same mock Hessian structure as QUBEKit uses (kcal/mol/nm² - default)
        hessian = create_mock_hessian(len(_QUBEKIT_COORDS_NM), k_diagonal=500.0)
        return HessianDecomposer(hessian, _QUBEKIT_COORDS_NM)

    @pytest.fixture
    def bond_list(self):
        """Bond connectivity."""
        return _QUBEKIT_BONDS

    @pytest.fixture
    def angle_list(self):
        """Angle connectivity."""
        return _QUBEKIT_ANGLES

    def test_bond_lengths(self, decomposer, bond_list):
        """Test that calculated bond lengths match QUBEKit within tolerance."""
        bond_params = calculate_bond_params(bond_list, decomposer, vib_scaling=1.0)

        for bond in bond_list:
            expected_length = _QUBEKIT_BOND_PARAMS[bond][0]
            calculated_length = bond_params[bond].length.m_as(_BOND_LENGTH_UNIT)

            # Bond lengths should match very closely (geometry is the same)
            np.testing.assert_allclose(
                calculated_length,
                expected_length,
                rtol=1e-4,
                err_msg=f"Bond length mismatch for bond {bond}",
            )

    def test_bond_force_constants(self, decomposer, bond_list):
        """Test that calculated bond force constants match QUBEKit within tolerance."""
        bond_params = calculate_bond_params(bond_list, decomposer, vib_scaling=1.0)

        for bond in bond_list:
            expected_k = _QUBEKIT_BOND_PARAMS[bond][1]
            calculated_k = bond_params[bond].force_constant.m_as(_BOND_K_UNIT)

            # Force constants should match within ~5%
            np.testing.assert_allclose(
                calculated_k,
                expected_k,
                rtol=0.05,
                err_msg=f"Bond force constant mismatch for bond {bond}: "
                f"calculated={calculated_k:.2f}, expected={expected_k:.2f}",
            )

    def test_angle_values(self, decomposer, angle_list):
        """Test that calculated angle values match QUBEKit within tolerance."""
        angle_params = calculate_angle_params(angle_list, decomposer, vib_scaling=1.0)

        for angle in angle_list:
            expected_angle_deg = _QUBEKIT_ANGLE_PARAMS[angle][0]
            calculated_angle_rad = angle_params[angle].angle.m_as(_ANGLE_UNIT)
            calculated_angle_deg = math.degrees(calculated_angle_rad)

            # Angles should match very closely (geometry is the same)
            np.testing.assert_allclose(
                calculated_angle_deg,
                expected_angle_deg,
                rtol=1e-3,
                err_msg=f"Angle mismatch for angle {angle}",
            )

    def test_angle_force_constants(self, decomposer, angle_list):
        """Test that calculated angle force constants match QUBEKit within tolerance."""
        angle_params = calculate_angle_params(angle_list, decomposer, vib_scaling=1.0)

        for angle in angle_list:
            expected_k = _QUBEKIT_ANGLE_PARAMS[angle][1]
            calculated_k = angle_params[angle].force_constant.m_as(_ANGLE_K_UNIT)

            # Force constants should match within ~10%
            np.testing.assert_allclose(
                calculated_k,
                expected_k,
                rtol=0.10,
                err_msg=f"Angle force constant mismatch for angle {angle}: "
                f"calculated={calculated_k:.2f}, expected={expected_k:.2f}",
            )

    def test_all_bonds_have_reference(self, bond_list):
        """Verify all test bonds have reference values."""
        for bond in bond_list:
            assert bond in _QUBEKIT_BOND_PARAMS, f"Missing reference for bond {bond}"

    def test_all_angles_have_reference(self, angle_list):
        """Verify all test angles have reference values."""
        for angle in angle_list:
            assert angle in _QUBEKIT_ANGLE_PARAMS, (
                f"Missing reference for angle {angle}"
            )

    def test_print_comparison_summary(self, decomposer, bond_list, angle_list):
        """Print a summary comparing calculated vs QUBEKit values."""
        bond_params = calculate_bond_params(bond_list, decomposer, vib_scaling=1.0)
        angle_params = calculate_angle_params(angle_list, decomposer, vib_scaling=1.0)

        print("\n" + "=" * 70)
        print("MSM vs QUBEKit Comparison for Fluorochlorobromomethanol")
        print("=" * 70)

        print("\nBOND PARAMETERS:")
        print("-" * 70)
        print(f"{'Bond':<10} {'Length (nm)':<18} {'Force Const (kcal/mol/nm²)':<30}")
        print(f"{'':10} {'Calc':<9}{'Ref':<9} {'Calc':<14}{'Ref':<14}{'Diff %':<8}")
        print("-" * 70)

        for bond in bond_list:
            calc_length = bond_params[bond].length.m_as(_BOND_LENGTH_UNIT)
            calc_k = bond_params[bond].force_constant.m_as(_BOND_K_UNIT)
            ref_length, ref_k = _QUBEKIT_BOND_PARAMS[bond]
            k_diff_pct = 100 * (calc_k - ref_k) / ref_k

            print(
                f"{bond!s:<10} {calc_length:<9.5f}{ref_length:<9.5f} "
                f"{calc_k:<14.2f}{ref_k:<14.2f}{k_diff_pct:+.2f}%"
            )

        print("\nANGLE PARAMETERS:")
        print("-" * 70)
        print(f"{'Angle':<12} {'Value (deg)':<18} {'Force Const (kcal/mol/rad²)':<28}")
        print(f"{'':12} {'Calc':<9}{'Ref':<9} {'Calc':<13}{'Ref':<13}{'Diff %':<8}")
        print("-" * 70)

        for angle in angle_list:
            calc_angle = math.degrees(angle_params[angle].angle.m_as(_ANGLE_UNIT))
            calc_k = angle_params[angle].force_constant.m_as(_ANGLE_K_UNIT)
            ref_angle, ref_k = _QUBEKIT_ANGLE_PARAMS[angle]
            k_diff_pct = 100 * (calc_k - ref_k) / ref_k

            print(
                f"{angle!s:<12} {calc_angle:<9.2f}{ref_angle:<9.2f} "
                f"{calc_k:<13.2f}{ref_k:<13.2f}{k_diff_pct:+.2f}%"
            )

        print("=" * 70)
