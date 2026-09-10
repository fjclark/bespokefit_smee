"""Functionality for applying the modified Seminario method.

Originally written by Alice E. A. Allen, TCM, University of Cambridge
Modified by Joshua T. Horton and rewritten by Chris Ringrose, Newcastle University
Further adapted for presto.

Reference: AEA Allen, MC Payne, DJ Cole, J. Chem. Theory Comput. (2018),
doi:10.1021/acs.jctc.7b00785
"""

import copy
from collections import defaultdict
from dataclasses import dataclass
from operator import itemgetter
from typing import Any

import loguru
import numpy as np
import openff.interchange
import openff.toolkit
import openmm.unit
import torch
from numpy import typing as npt
from openff.toolkit.typing.engines.smirnoff import (
    AngleHandler,
    BondHandler,
)
from openff.units import unit as off_unit
from rich.progress import track

from .hessian import calculate_hessian
from .sample import _build_ml_simulation, _copy_mol_and_add_conformers
from .settings import MSMSettings
from .utils.gpu import cleanup_simulation

logger = loguru.logger

# =============================================================================
# Unit Definitions
# =============================================================================
# We define standard units for internal calculations. The MSM algorithm works
# with unitless numpy arrays internally, so we strip units at boundaries and
# re-attach them when creating output objects.
#
# Internal units (used in numpy calculations):
#   - Length: nm
#   - Energy: kcal/mol (consistent throughout the codebase)
#   - Hessian: kcal/mol/nm² (for bonds) and kcal/mol/rad² (for angles)
#
# OpenMM uses kcal/mol internally, which matches our internal convention.
# =============================================================================

# OpenMM unit constants (for simulation setup)
_OMM_KELVIN = openmm.unit.kelvin
_OMM_FEMTOSECOND = openmm.unit.femtoseconds
_OMM_NM = openmm.unit.nanometer

# Internal calculation units (unitless floats in these dimensions)
_INTERNAL_LENGTH_UNIT = openmm.unit.nanometer
_INTERNAL_ENERGY_UNIT = openmm.unit.kilocalorie_per_mole
_INTERNAL_HESSIAN_UNIT = _INTERNAL_ENERGY_UNIT / (_INTERNAL_LENGTH_UNIT**2)

# OpenFF unit constants for MSM outputs (with explicit units attached)
_BOND_K_UNIT = off_unit.kilocalorie_per_mole / off_unit.nanometer**2
_BOND_LENGTH_UNIT = off_unit.nanometer
_ANGLE_K_UNIT = off_unit.kilocalorie_per_mole / off_unit.radian**2
_ANGLE_UNIT = off_unit.radian


@dataclass
class BondParams:
    """Parameters for a harmonic bond potential with explicit units.

    Force constant is in kcal/mol/nm² and length is in nm.
    Both are stored as OpenFF Quantity objects with explicit units.
    """

    force_constant: Any  # off_unit.Quantity, kcal/mol/nm²
    length: Any  # off_unit.Quantity, nm

    @classmethod
    def from_values(cls, force_constant: float, length: float) -> "BondParams":
        """Create BondParams from raw float values (assumed kcal/mol/nm² and nm)."""
        return cls(
            force_constant=force_constant * _BOND_K_UNIT,
            length=length * _BOND_LENGTH_UNIT,
        )


@dataclass
class AngleParams:
    """Parameters for a harmonic angle potential with explicit units.

    Force constant is in kcal/mol/rad² and angle is in radians.
    Both are stored as OpenFF Quantity objects with explicit units.
    """

    force_constant: Any  # off_unit.Quantity, kcal/mol/rad²
    angle: Any  # off_unit.Quantity, radians

    @classmethod
    def from_values(cls, force_constant: float, angle: float) -> "AngleParams":
        """Create AngleParams from raw float values (assumed kcal/mol/rad² and radians)."""
        return cls(
            force_constant=force_constant * _ANGLE_K_UNIT,
            angle=angle * _ANGLE_UNIT,
        )


# --- Vector Calculation Functions ---


def unit_vector_along_bond(
    coords: npt.NDArray[np.floating], atom_a: int, atom_b: int
) -> npt.NDArray[np.floating]:
    """Calculate unit vector along a bond from atom_a to atom_b.

    Args:
        coords: Atomic coordinates array of shape (n_atoms, 3).
        atom_a: Index of first atom.
        atom_b: Index of second atom.

    Returns:
        Unit vector from atom_a to atom_b.
    """
    diff_ab = coords[atom_b] - coords[atom_a]
    result: npt.NDArray[np.floating[Any]] = diff_ab / np.linalg.norm(diff_ab)
    return result


def unit_vector_normal_to_plane(
    u_bc: npt.NDArray[np.floating], u_ab: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Calculate unit vector normal to the plane defined by two vectors.

    Args:
        u_bc: First unit vector.
        u_ab: Second unit vector.

    Returns:
        Unit vector normal to the plane defined by u_bc and u_ab.
    """
    cross = np.cross(u_bc, u_ab)
    norm = np.linalg.norm(cross)
    if norm < 1e-10:
        # Vectors are nearly parallel; return arbitrary perpendicular vector
        return _get_arbitrary_perpendicular(u_ab)
    result: npt.NDArray[np.floating[Any]] = cross / norm
    return result


def _get_arbitrary_perpendicular(
    u: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Get an arbitrary unit vector perpendicular to the given vector.

    Args:
        u: Input unit vector.

    Returns:
        Unit vector perpendicular to u.
    """
    # Find a vector not parallel to u
    if abs(u[0]) < 0.9:
        v = np.array([1.0, 0.0, 0.0])
    else:
        v = np.array([0.0, 1.0, 0.0])

    # Use Gram-Schmidt
    perp = v - np.dot(v, u) * u
    result: npt.NDArray[np.floating[Any]] = perp / np.linalg.norm(perp)
    return result


def compute_perpendicular_vector_in_plane(
    coords: npt.NDArray[np.floating], angle: tuple[int, int, int]
) -> npt.NDArray[np.floating]:
    """Compute vector in angle plane and perpendicular to the first bond.

    For angle a-b-c, computes the vector in plane abc that is perpendicular
    to the bond a-b.

    Args:
        coords: Atomic coordinates array.
        angle: Tuple of (atom_a, atom_b, atom_c) indices.

    Returns:
        Unit vector in the plane perpendicular to bond a-b.
    """
    atom_a, atom_b, atom_c = angle

    u_ab = unit_vector_along_bond(coords, atom_a, atom_b)
    u_cb = unit_vector_along_bond(coords, atom_c, atom_b)

    u_n = unit_vector_normal_to_plane(u_cb, u_ab)

    return unit_vector_normal_to_plane(u_n, u_ab)


# --- Force Constant Calculation Functions ---


def calculate_bond_force_constant(
    bond: tuple[int, int],
    eigenvals: npt.NDArray[np.complexfloating],
    eigenvecs: npt.NDArray[np.complexfloating],
    coords: npt.NDArray[np.floating],
) -> float:
    """Calculate force constant for a bond using the Seminario method.

    Implements Equation 10 from the Seminario paper.

    Args:
        bond: Tuple of (atom_a, atom_b) indices.
        eigenvals: Eigenvalues array of shape (n_atoms, n_atoms, 3).
        eigenvecs: Eigenvectors array of shape (3, 3, n_atoms, n_atoms).
        coords: Atomic coordinates in Angstroms.

    Returns:
        Force constant in kcal/mol/Å^2.
    """
    atom_a, atom_b = bond
    eigenvals_ab = eigenvals[atom_a, atom_b, :]
    eigenvecs_ab = eigenvecs[:, :, atom_a, atom_b]

    u_ab = unit_vector_along_bond(coords, atom_a, atom_b)

    # Sum over eigenvalues weighted by projection onto bond direction
    k_bond = -1 * sum(
        eigenvals_ab[i] * abs(np.dot(u_ab, eigenvecs_ab[:, i])) for i in range(3)
    )

    return float(np.real(k_bond))


def calculate_angle_force_constant(
    angle: tuple[int, int, int],
    bond_lengths: npt.NDArray[np.floating],
    eigenvals: npt.NDArray[np.complexfloating],
    eigenvecs: npt.NDArray[np.complexfloating],
    coords: npt.NDArray[np.floating],
    scalings: tuple[float, float],
) -> tuple[float, float]:
    """Calculate force constant and equilibrium angle using Modified Seminario Method.

    Implements Equation 14 from the Seminario paper with modifications for
    additional angle contributions.

    Args:
        angle: Tuple of (atom_a, atom_b, atom_c) indices.
        bond_lengths: Matrix of bond lengths.
        eigenvals: Eigenvalues array.
        eigenvecs: Eigenvectors array.
        coords: Atomic coordinates in Angstroms.
        scalings: Scaling factors (scale_ab, scale_cb) for Modified Seminario.

    Returns:
        Tuple of (force_constant, equilibrium_angle) where force constant is
        in kcal/mol/rad^2 and angle is in degrees.
    """
    atom_a, atom_b, atom_c = angle

    u_ab = unit_vector_along_bond(coords, atom_a, atom_b)
    u_cb = unit_vector_along_bond(coords, atom_c, atom_b)

    bond_len_ab = bond_lengths[atom_a, atom_b]
    eigenvals_ab = eigenvals[atom_a, atom_b, :]
    eigenvecs_ab = eigenvecs[:3, :3, atom_a, atom_b]

    bond_len_bc = bond_lengths[atom_b, atom_c]
    eigenvals_cb = eigenvals[atom_c, atom_b, :]
    eigenvecs_cb = eigenvecs[:3, :3, atom_c, atom_b]

    # Check for linear angle
    if _is_linear_angle(u_ab, u_cb):
        return _calculate_linear_angle_force_constant(
            u_ab,
            u_cb,
            (bond_len_ab, bond_len_bc),
            (eigenvals_ab, eigenvals_cb),
            (eigenvecs_ab, eigenvecs_cb),
        )

    # Normal case: compute perpendicular vectors
    u_n = unit_vector_normal_to_plane(u_cb, u_ab)
    u_pa = unit_vector_normal_to_plane(u_n, u_ab)
    u_pc = unit_vector_normal_to_plane(u_cb, u_n)

    # Compute weighted sums with scaling factors
    sum_first = (
        sum(
            eigenvals_ab[i] * abs(_dot_product(u_pa, eigenvecs_ab[:, i]))
            for i in range(3)
        )
        / scalings[0]
    )
    sum_second = (
        sum(
            eigenvals_cb[i] * abs(_dot_product(u_pc, eigenvecs_cb[:, i]))
            for i in range(3)
        )
        / scalings[1]
    )

    # Combine as two springs in series
    k_theta = 1.0 / (
        (1 / ((bond_len_ab**2) * sum_first)) + (1 / ((bond_len_bc**2) * sum_second))
    )

    k_theta = abs(float(np.real(k_theta)))

    # Equilibrium angle
    theta_0 = np.degrees(np.arccos(np.clip(np.dot(u_ab, u_cb), -1.0, 1.0)))

    return k_theta, theta_0


def _is_linear_angle(
    u_ab: npt.NDArray[np.floating], u_cb: npt.NDArray[np.floating]
) -> bool:
    """Return whether two bond vectors use the MSM linear-angle treatment."""
    diff_norm = abs(np.linalg.norm(u_cb - u_ab))
    return bool(diff_norm < 0.01 or 1.99 < diff_norm < 2.01)


def _calculate_linear_angle_force_constant(
    u_ab: npt.NDArray[np.floating],
    u_cb: npt.NDArray[np.floating],
    bond_lens: tuple[float, float],
    eigenvals: tuple[npt.NDArray[np.complexfloating], npt.NDArray[np.complexfloating]],
    eigenvecs: tuple[npt.NDArray[np.complexfloating], npt.NDArray[np.complexfloating]],
    n_samples: int = 200,
) -> tuple[float, float]:
    """Calculate force constant for a linear or near-linear angle.

    For linear angles (e.g., nitrile groups), the perpendicular vector is not
    uniquely defined. We sample different perpendicular directions and average.

    Args:
        u_ab: Unit vector along bond a-b.
        u_cb: Unit vector along bond c-b.
        bond_lens: Tuple of (bond_len_ab, bond_len_bc).
        eigenvals: Tuple of eigenvalue arrays for (a-b, c-b).
        eigenvecs: Tuple of eigenvector arrays for (a-b, c-b).
        n_samples: Number of samples around the bond direction.

    Returns:
        Tuple of (force_constant, equilibrium_angle). The force constant uses
        the SMIRNOFF / OpenMM harmonic convention ``U = k * delta**2 / 2``.
    """
    k_theta_array = np.zeros(n_samples)

    for theta_idx in range(n_samples):
        theta = 2 * np.pi * theta_idx / n_samples
        # Generate a sample normal vector
        u_n = np.array(
            [
                np.sin(theta) * np.cos(theta),
                np.sin(theta) * np.sin(theta),
                np.cos(theta),
            ]
        )

        u_pa = unit_vector_normal_to_plane(u_n, u_ab)
        u_pc = unit_vector_normal_to_plane(u_cb, u_n)

        sum_first = sum(
            eigenvals[0][i] * abs(_dot_product(u_pa, eigenvecs[0][:, i]))
            for i in range(3)
        )
        sum_second = sum(
            eigenvals[1][i] * abs(_dot_product(u_pc, eigenvecs[1][:, i]))
            for i in range(3)
        )

        k_theta_i = 1.0 / (
            (1 / ((bond_lens[0] ** 2) * sum_first))
            + (1 / ((bond_lens[1] ** 2) * sum_second))
        )
        # QUBEKit multiplies this intermediate value by 0.5, then multiplies
        # angle force constants by 2 when exporting them. Presto returns the
        # final SMIRNOFF / OpenMM force constant directly, so those factors
        # cancel here, as they do in the non-linear angle path above.
        k_theta_array[theta_idx] = abs(float(np.real(k_theta_i)))

    k_theta = float(np.mean(k_theta_array))
    theta_0 = np.degrees(np.arccos(np.clip(np.dot(u_ab, u_cb), -1.0, 1.0)))

    return k_theta, theta_0


def _dot_product(
    u_pa: npt.NDArray[np.floating[Any]],
    eig_vec: npt.NDArray[np.complexfloating[Any, Any]],
) -> complex:
    """Compute dot product handling complex eigenvectors."""
    return complex(sum(u_pa[i] * eig_vec[i].conjugate() for i in range(3)))


# --- Scaling Factor Calculation ---


class AngleScalingCalculator:
    """Calculator for Modified Seminario Method scaling factors.

    The scaling factors account for the contribution of other angles sharing
    the same central atom and bond.
    """

    def __init__(
        self,
        angles: list[tuple[int, int, int]],
        coords: npt.NDArray[np.floating],
        n_atoms: int,
    ):
        """Initialize the scaling factor calculator.

        Args:
            angles: List of angle tuples (atom_a, atom_b, atom_c).
            coords: Atomic coordinates.
            n_atoms: Number of atoms in the molecule.
        """
        self.angles = angles
        self.coords = coords
        self.n_atoms = n_atoms

        # Build data structures for efficient lookup
        self._build_central_atom_lookup()
        self._compute_perpendicular_vectors()

    def _build_central_atom_lookup(self) -> None:
        """Build lookup structure for angles organized by central atom."""
        # central_atoms_angles[b] = [(a, c, angle_idx), (c, a, angle_idx), ...]
        self.central_atoms_angles: list[list[tuple[int, int, int]]] = [
            [] for _ in range(self.n_atoms)
        ]

        for angle_idx, (atom_a, atom_b, atom_c) in enumerate(self.angles):
            # Add both orderings for the angle
            self.central_atoms_angles[atom_b].append((atom_a, atom_c, angle_idx))
            self.central_atoms_angles[atom_b].append((atom_c, atom_a, angle_idx))

        # Sort by first atom for consistent ordering
        for coord in range(self.n_atoms):
            self.central_atoms_angles[coord] = sorted(
                self.central_atoms_angles[coord], key=itemgetter(0)
            )

    def _compute_perpendicular_vectors(self) -> None:
        """Compute perpendicular vectors for all angle contributions."""
        self.unit_pa_all_angles: list[list[npt.NDArray[np.floating]]] = [
            [] for _ in range(self.n_atoms)
        ]

        for central_atom in range(self.n_atoms):
            for entry in self.central_atoms_angles[central_atom]:
                first_atom, third_atom, _ = entry
                angle = (first_atom, central_atom, third_atom)
                u_pa = compute_perpendicular_vector_in_plane(self.coords, angle)
                self.unit_pa_all_angles[central_atom].append(u_pa)

    def compute_scaling_factors(self) -> list[tuple[float, float]]:
        """Compute scaling factors for all angles.

        Returns:
            List of (scale_ab, scale_cb) tuples for each angle.
        """
        # Compute raw scaling factors organized by central atom
        scaling_factor_all_angles: list[list[tuple[float, int]]] = [
            [] for _ in range(self.n_atoms)
        ]

        for central_atom in range(self.n_atoms):
            n_entries = len(self.central_atoms_angles[central_atom])

            for j in range(n_entries):
                angle_idx = self.central_atoms_angles[central_atom][j][2]
                first_atom = self.central_atoms_angles[central_atom][j][0]

                # Count additional contributions from angles sharing the same bond
                extra_contribs = 0.0
                n_neighbors = 0

                # Forward direction
                n = 1
                while (
                    j + n < n_entries
                    and self.central_atoms_angles[central_atom][j + n][0] == first_atom
                ):
                    dot_val = np.dot(
                        self.unit_pa_all_angles[central_atom][j],
                        self.unit_pa_all_angles[central_atom][j + n],
                    )
                    extra_contribs += abs(dot_val) ** 2
                    n += 1
                    n_neighbors += 1

                # Backward direction
                m = 1
                while (
                    j - m >= 0
                    and self.central_atoms_angles[central_atom][j - m][0] == first_atom
                ):
                    dot_val = np.dot(
                        self.unit_pa_all_angles[central_atom][j],
                        self.unit_pa_all_angles[central_atom][j - m],
                    )
                    extra_contribs += abs(dot_val) ** 2
                    m += 1
                    n_neighbors += 1

                # Compute scaling factor
                scale = 1.0
                if n_neighbors > 0:
                    scale += extra_contribs / n_neighbors

                scaling_factor_all_angles[central_atom].append((scale, angle_idx))

        # Reorganize scaling factors by angle index
        scaling_factors_by_angle: list[list[float]] = [
            [] for _ in range(len(self.angles))
        ]

        for central_atom in range(self.n_atoms):
            for scale, angle_idx in scaling_factor_all_angles[central_atom]:
                scaling_factors_by_angle[angle_idx].append(scale)

        # Return as list of tuples
        return [
            (scales[0], scales[1]) if len(scales) >= 2 else (1.0, 1.0)
            for scales in scaling_factors_by_angle
        ]


# --- Hessian Decomposition ---


class HessianDecomposer:
    """Decomposes a molecular Hessian matrix into atom-pair contributions."""

    def __init__(
        self, hessian: npt.NDArray[np.floating], coords: npt.NDArray[np.floating]
    ):
        """Initialize with Hessian and coordinates.

        Args:
            hessian: Full Hessian matrix of shape (3*n_atoms, 3*n_atoms).
            coords: Atomic coordinates of shape (n_atoms, 3).
        """
        self.hessian = hessian
        self.coords = coords
        self.n_atoms = len(coords)

        # Compute eigenvalues, eigenvectors, and bond lengths
        self._decompose()

    def _decompose(self) -> None:
        """Compute eigendecomposition of partial Hessian matrices."""
        n = self.n_atoms

        self.eigenvecs = np.empty((3, 3, n, n), dtype=complex)
        self.eigenvals = np.empty((n, n, 3), dtype=complex)
        self.bond_lengths = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                diff_i_j = self.coords[i] - self.coords[j]
                self.bond_lengths[i, j] = np.linalg.norm(diff_i_j)

                partial_hessian = self.hessian[
                    (i * 3) : ((i + 1) * 3), (j * 3) : ((j + 1) * 3)
                ]

                self.eigenvals[i, j, :], self.eigenvecs[:, :, i, j] = np.linalg.eig(
                    partial_hessian
                )


# --- Main MSM Functions ---


def calculate_bond_params(
    bond_indices: list[tuple[int, int]],
    hessian_decomposer: HessianDecomposer,
    vib_scaling: float,
) -> dict[tuple[int, int], BondParams]:
    """Calculate bond parameters using the Modified Seminario Method.

    Args:
        bond_indices: List of bond tuples (atom_i, atom_j).
        hessian_decomposer: Decomposed Hessian data (in internal units: kcal/mol/nm²).
        vib_scaling: Vibrational scaling factor.

    Returns:
        Dictionary mapping bond indices to BondParams.
    """
    bond_params: dict[tuple[int, int], BondParams] = {}

    for bond in bond_indices:
        # Calculate force constant in both directions and average
        k_ab = calculate_bond_force_constant(
            bond,
            hessian_decomposer.eigenvals,
            hessian_decomposer.eigenvecs,
            hessian_decomposer.coords,
        )
        k_ba = calculate_bond_force_constant(
            (bond[1], bond[0]),
            hessian_decomposer.eigenvals,
            hessian_decomposer.eigenvecs,
            hessian_decomposer.coords,
        )

        # Average and apply vibrational scaling
        # Force constant is in kcal/mol/nm² (from internal units)
        k_bond = ((k_ab + k_ba) / 2.0) * (vib_scaling**2)

        # Get equilibrium length in nm (from internal units)
        bond_length_nm = hessian_decomposer.bond_lengths[bond[0], bond[1]]

        bond_params[bond] = BondParams.from_values(
            force_constant=k_bond, length=bond_length_nm
        )

    return bond_params


def calculate_angle_params(
    angle_indices: list[tuple[int, int, int]],
    hessian_decomposer: HessianDecomposer,
    vib_scaling: float,
) -> dict[tuple[int, int, int], AngleParams]:
    """Calculate angle parameters using the Modified Seminario Method.

    Args:
        angle_indices: List of angle tuples (atom_i, atom_j, atom_k).
        hessian_decomposer: Decomposed Hessian data (in internal units: kcal/mol/nm²).
        vib_scaling: Vibrational scaling factor.

    Returns:
        Dictionary mapping angle indices to AngleParams.
    """
    if not angle_indices:
        return {}

    # Compute scaling factors for all angles
    scaling_calculator = AngleScalingCalculator(
        angle_indices, hessian_decomposer.coords, hessian_decomposer.n_atoms
    )
    all_scaling_factors = scaling_calculator.compute_scaling_factors()

    angle_params: dict[tuple[int, int, int], AngleParams] = {}

    for angle_idx, angle in enumerate(angle_indices):
        scalings = all_scaling_factors[angle_idx]

        # Calculate in both directions and average
        k_theta_ab, theta_0_ab = calculate_angle_force_constant(
            angle,
            hessian_decomposer.bond_lengths,
            hessian_decomposer.eigenvals,
            hessian_decomposer.eigenvecs,
            hessian_decomposer.coords,
            scalings,
        )
        k_theta_ba, theta_0_ba = calculate_angle_force_constant(
            (angle[2], angle[1], angle[0]),
            hessian_decomposer.bond_lengths,
            hessian_decomposer.eigenvals,
            hessian_decomposer.eigenvecs,
            hessian_decomposer.coords,
            (scalings[1], scalings[0]),
        )

        # Average and apply vibrational scaling
        # Force constant is in kcal/mol/rad² (derived from internal units)
        k_theta = ((k_theta_ab + k_theta_ba) / 2.0) * (vib_scaling**2)

        # Angle is returned in degrees, convert to radians
        theta_0_rad = np.radians((theta_0_ab + theta_0_ba) / 2.0)

        angle_params[angle] = AngleParams.from_values(
            force_constant=k_theta, angle=theta_0_rad
        )

    return angle_params


def apply_msm_to_molecule(
    mol: openff.toolkit.Molecule,
    bond_indices: list[tuple[int, int]],
    angle_indices: list[tuple[int, int, int]],
    settings: MSMSettings,
    device: torch.device,
) -> tuple[dict[tuple[int, int], BondParams], dict[tuple[int, int, int], AngleParams]]:
    """Apply Modified Seminario Method to calculate bond and angle parameters.

    Sets up an ML potential simulation, minimizes the structure, computes the
    Hessian, and calculates force field parameters. If settings.n_conformers > 1,
    parameters are calculated for each conformer and averaged.

    Args:
        mol: OpenFF Molecule to parameterize.
        bond_indices: List of bond indices to calculate parameters for.
        angle_indices: List of angle indices to calculate parameters for.
        settings: MSM settings including ML potential, scaling factors, and n_conformers.
        device: Torch device to use for ML potential calculations.

    Returns:
        Tuple of (bond_params_dict, angle_params_dict). If multiple conformers are used,
        the parameters are averaged over all conformers.
    """
    # Generate conformers (or load the supplied starting conformers)
    mol_with_conformers = _copy_mol_and_add_conformers(
        mol,
        n_conformers=settings.n_conformers,
        starting_conformers=settings.starting_conformers,
    )
    simulation, integrator = _build_ml_simulation(
        mol_with_conformers,
        mol_with_conformers.to_topology().to_openmm(),
        settings.mlp_settings,
        300 * _OMM_KELVIN,
        1.0 * _OMM_FEMTOSECOND,
        device=device,
    )

    # Collect parameters from each conformer
    all_bond_params: defaultdict[tuple[int, int], list[BondParams]] = defaultdict(list)
    all_angle_params: defaultdict[tuple[int, int, int], list[AngleParams]] = (
        defaultdict(list)
    )

    for conformer in track(
        mol_with_conformers.conformers,
        description="Finding MSM parameters for conformers",
    ):
        # Set positions for this conformer
        simulation.context.setPositions(conformer.to_openmm())

        # Minimize to local energy minimum
        simulation.minimizeEnergy(maxIterations=0, tolerance=settings.tolerance)
        positions = simulation.context.getState(getPositions=True).getPositions(
            asNumpy=True
        )

        # Convert positions to internal units (nm)
        coords = positions.value_in_unit(_INTERNAL_LENGTH_UNIT)

        # Compute Hessian at the minimum and convert to internal units (kcal/mol/nm²)
        # The hessian is returned with OpenMM units attached, so we use automatic
        # unit conversion to our internal standard units
        hessian_with_units = calculate_hessian(
            simulation,
            positions,
            finite_step=settings.finite_step,
        )
        hessian = hessian_with_units.value_in_unit(_INTERNAL_HESSIAN_UNIT)

        # Decompose Hessian (works with unitless numpy arrays in internal units)
        hessian_decomposer = HessianDecomposer(hessian, coords)

        # Calculate bond and angle parameters for this conformer
        conformer_bond_params = calculate_bond_params(
            bond_indices, hessian_decomposer, settings.vib_scaling
        )
        conformer_angle_params = calculate_angle_params(
            angle_indices, hessian_decomposer, settings.vib_scaling
        )

        # Collect parameters
        for bond_idx, bond_param in conformer_bond_params.items():
            all_bond_params[bond_idx].append(bond_param)
        for angle_idx, angle_param in conformer_angle_params.items():
            all_angle_params[angle_idx].append(angle_param)

    # Clean up OpenMM objects to free GPU memory
    cleanup_simulation(simulation, integrator)

    # Average parameters over all conformers
    bond_params = {
        bond_idx: _mean_bond_params(params_list)
        for bond_idx, params_list in all_bond_params.items()
    }
    angle_params = {
        angle_idx: _mean_angle_params(params_list)
        for angle_idx, params_list in all_angle_params.items()
    }

    return bond_params, angle_params


def _mean_bond_params(params_list: list[BondParams]) -> BondParams:
    """Calculate mean bond parameters from a list."""
    if not params_list:
        raise ValueError("No bond parameters provided for averaging.")
    n = len(params_list)
    mean_k = sum(p.force_constant for p in params_list) / n
    mean_length = sum(p.length for p in params_list) / n
    return BondParams(force_constant=mean_k, length=mean_length)


def _mean_angle_params(params_list: list[AngleParams]) -> AngleParams:
    """Calculate mean angle parameters from a list."""
    if not params_list:
        raise ValueError("No angle parameters provided for averaging.")
    n = len(params_list)
    mean_k = sum(p.force_constant for p in params_list) / n
    mean_angle = sum(p.angle for p in params_list) / n
    return AngleParams(force_constant=mean_k, angle=mean_angle)


def apply_msm_to_molecules(
    mols: list[openff.toolkit.Molecule],
    off_ff: openff.toolkit.ForceField,
    settings: MSMSettings,
    device: torch.device,
) -> openff.toolkit.ForceField:
    """Apply Modified Seminario Method to molecules and update force field.

    Calculates bond and angle parameters for all molecules, averages parameters
    that share the same SMIRKS pattern, and updates the force field accordingly.

    Args:
        mols: List of molecules to parameterize.
        off_ff: OpenFF ForceField to modify.
        settings: MSM settings.
        device: Torch device for ML potential calculations.

    Returns:
        Modified ForceField with MSM-derived parameters.

    Reference:
        AEA Allen, MC Payne, DJ Cole, J. Chem. Theory Comput. (2018),
        doi:10.1021/acs.jctc.7b00785
    """
    # Make a copy of the force field to avoid modifying the original
    off_ff = copy.deepcopy(off_ff)

    # Collect parameters by SMIRKS across all molecules
    bond_params_by_smirks: defaultdict[str, list[BondParams]] = defaultdict(list)
    angle_params_by_smirks: defaultdict[str, list[AngleParams]] = defaultdict(list)

    for mol in track(mols, description="Applying MSM to molecules", transient=True):
        # Get parameter labels for this molecule
        labels = off_ff.label_molecules(mol.to_topology())[0]

        # Build index-to-SMIRKS mappings
        bond_indices_to_smirks = {
            bond_indices: bond_param.smirks
            for (bond_indices, bond_param) in labels["Bonds"].items()
        }
        angle_indices_to_smirks = {
            angle_indices: angle_param.smirks
            for (angle_indices, angle_param) in labels["Angles"].items()
        }

        bond_indices = list(bond_indices_to_smirks.keys())
        angle_indices = list(angle_indices_to_smirks.keys())

        # Calculate MSM parameters for this molecule
        mol_bond_params, mol_angle_params = apply_msm_to_molecule(
            mol, bond_indices, angle_indices, settings, device
        )

        # Collect parameters by SMIRKS
        for bond_idx, bond_params in mol_bond_params.items():
            smirks = bond_indices_to_smirks[bond_idx]
            bond_params_by_smirks[smirks].append(bond_params)

        for angle_idx, angle_params in mol_angle_params.items():
            smirks = angle_indices_to_smirks[angle_idx]
            angle_params_by_smirks[smirks].append(angle_params)

    # Average parameters and update force field
    bond_handler: BondHandler = off_ff.get_parameter_handler("Bonds")
    for smirks, bond_params_list in bond_params_by_smirks.items():
        mean_bond_params = _mean_bond_params(bond_params_list)
        bond_param = bond_handler.get_parameter({"smirks": smirks})[0]

        # Get the original units from the force field parameter
        original_k_units = bond_param.k.units
        original_length_units = bond_param.length.units

        # Params already have units, just convert to original force field units
        bond_param.k = mean_bond_params.force_constant.to(original_k_units)
        bond_param.length = mean_bond_params.length.to(original_length_units)

        logger.debug(
            f"Updated bond {smirks}: k={bond_param.k}, length={bond_param.length}"
        )

    angle_handler: AngleHandler = off_ff.get_parameter_handler("Angles")
    for smirks, angle_params_list in angle_params_by_smirks.items():
        mean_angle_params = _mean_angle_params(angle_params_list)
        angle_param = angle_handler.get_parameter({"smirks": smirks})[0]

        # Get the original units from the force field parameter
        original_k_units = angle_param.k.units
        original_angle_units = angle_param.angle.units

        # Params already have units, just convert to original force field units
        angle_param.k = mean_angle_params.force_constant.to(original_k_units)
        angle_param.angle = mean_angle_params.angle.to(original_angle_units)

        logger.debug(
            f"Updated angle {smirks}: k={angle_param.k}, angle={angle_param.angle}"
        )

    return off_ff
