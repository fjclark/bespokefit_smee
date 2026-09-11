"""Generate reference MSM values using QUBEKit's ModSeminario implementation.

This script uses QUBEKit directly to compute bond and angle parameters using
the Modified Seminario Method, which can then be used as reference values
for testing our implementation.

Usage:
    conda create -n qubekit-2.1.1 -c conda-forge python=3.9 qubekit=2.1.1 \
        pydantic=1.10 openff-toolkit-base=0.10.4 openmm
    conda run -n qubekit-2.1.1 python \
        presto/data/msm/generate_qubekit_reference.py

Requirements:
    - QUBEKit 2.1.1 and its contemporary Pydantic/OpenFF dependencies

The script will:
    1. Generate the existing asymmetric-molecule reference
    2. Create acetonitrile with its C-C#N angle bent to 175 degrees
    3. Run QUBEKit's complete ModSeminario method for both molecules
    4. Output the resulting final bond and angle parameters

These values can be compared against our implementation to verify correctness.

Note on mock Hessian:
    This script contains its own mock Hessian generation functions because it must
    run in a separate QUBEKit environment where presto is not installed.
    The create_mock_hessian function in tests/unit/test_msm.py uses the same
    algorithm and parameters (k_diagonal=500.0 kcal/mol/Å²).

Note on test molecule:
    We use fluorochlorobromomethanol (OC(F)(Cl)Br), a fully asymmetric molecule,
    to avoid QUBEKit's internal symmetry averaging when comparing against our
    implementation (in our workflow, symmetry averaging is handled by having
    symmetric atoms share the same SMIRKS types). A separate acetonitrile fixture
    exercises QUBEKit's full near-linear-angle path.
"""

import json
import os
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

# Check if QUBEKit is available
try:
    from qubekit.bonded import ModSeminario
    from qubekit.molecules import Ligand
    from qubekit.utils import constants
except ImportError as error:
    raise ImportError(
        "QUBEKit and its runtime dependencies are required; run this script "
        "in an environment with QUBEKit 2.1.1 installed."
    ) from error


_QUBEKIT_VERSION = "2.1.1"
_NEAR_LINEAR_ANGLE_DEGREES = 175.0


def _run_mod_seminario(molecule: Ligand) -> Ligand:
    """Run QUBEKit without leaving its text reports in the source tree."""
    previous_directory = os.getcwd()
    with TemporaryDirectory() as temporary_directory:
        try:
            os.chdir(temporary_directory)
            return ModSeminario(vibrational_scaling=1.0).run(molecule=molecule)
        finally:
            os.chdir(previous_directory)


def _rotation_matrix(pair_index: int) -> np.ndarray:
    """Return a deterministic rotation used to orient a Hessian pair block."""
    alpha, beta, gamma = (
        0.13 * pair_index,
        0.17 * pair_index,
        0.19 * pair_index,
    )
    sin_a, cos_a = np.sin(alpha), np.cos(alpha)
    sin_b, cos_b = np.sin(beta), np.cos(beta)
    sin_g, cos_g = np.sin(gamma), np.cos(gamma)
    rotate_x = np.array([[1.0, 0.0, 0.0], [0.0, cos_a, -sin_a], [0.0, sin_a, cos_a]])
    rotate_y = np.array([[cos_b, 0.0, sin_b], [0.0, 1.0, 0.0], [-sin_b, 0.0, cos_b]])
    rotate_z = np.array([[cos_g, -sin_g, 0.0], [sin_g, cos_g, 0.0], [0.0, 0.0, 1.0]])
    return np.asarray(rotate_z @ rotate_y @ rotate_x, dtype=np.float64)


def create_nondegenerate_hessian_angstrom(n_atoms: int) -> np.ndarray:
    """Create a deterministic PSD Hessian in kcal/mol/Angstrom**2.

    Distinct, anisotropic atom-pair blocks avoid arbitrary eigenvectors in the
    QUBEKit/Presto differential test. The block-Laplacian construction makes
    the complete matrix symmetric, positive semidefinite, and translationally
    invariant.
    """
    hessian = np.zeros((3 * n_atoms, 3 * n_atoms))
    pair_index = 0
    for atom_i in range(n_atoms):
        for atom_j in range(atom_i + 1, n_atoms):
            pair_index += 1
            rotation = _rotation_matrix(pair_index)
            eigenvalues = np.array(
                [
                    40.0 + 3.0 * pair_index,
                    70.0 + 5.0 * pair_index,
                    110.0 + 7.0 * pair_index,
                ]
            )
            pair_block = rotation @ np.diag(eigenvalues) @ rotation.T
            slice_i = slice(3 * atom_i, 3 * (atom_i + 1))
            slice_j = slice(3 * atom_j, 3 * (atom_j + 1))
            hessian[slice_i, slice_i] += pair_block
            hessian[slice_j, slice_j] += pair_block
            hessian[slice_i, slice_j] -= pair_block
            hessian[slice_j, slice_i] -= pair_block

    np.testing.assert_allclose(hessian, hessian.T, atol=1e-12)
    np.testing.assert_allclose(
        hessian.reshape(n_atoms, 3, n_atoms, 3).sum(axis=2), 0.0, atol=1e-10
    )
    assert np.linalg.eigvalsh(hessian).min() > -1e-10
    return hessian


def _measure_angle_degrees(coords: np.ndarray, angle: tuple[int, int, int]) -> float:
    """Measure an angle from Cartesian coordinates in degrees."""
    atom_a, atom_b, atom_c = angle
    vector_ba = coords[atom_a] - coords[atom_b]
    vector_bc = coords[atom_c] - coords[atom_b]
    cosine = np.dot(vector_ba, vector_bc) / (
        np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc)
    )
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def create_near_linear_acetonitrile() -> tuple[Ligand, tuple[int, int, int]]:
    """Create acetonitrile with a deterministic 175-degree C-C-N angle."""
    molecule = Ligand.from_smiles("CC#N", "near_linear_acetonitrile")
    topology = molecule.to_topology()

    nitrogen_indices = [
        index
        for index, atom in enumerate(molecule.atoms)
        if atom.atomic_symbol == "N" and topology.degree[index] == 1
    ]
    assert len(nitrogen_indices) == 1
    nitrogen = nitrogen_indices[0]
    nitrile_carbon = next(iter(topology.neighbors(nitrogen)))
    assert molecule.atoms[nitrile_carbon].atomic_symbol == "C"
    carbon_substituents = [
        index
        for index in topology.neighbors(nitrile_carbon)
        if index != nitrogen and molecule.atoms[index].atomic_symbol == "C"
    ]
    assert len(carbon_substituents) == 1
    methyl_carbon = carbon_substituents[0]
    target_angle = (methyl_carbon, nitrile_carbon, nitrogen)
    assert target_angle in molecule.angles or target_angle[::-1] in molecule.angles

    coords = np.array(molecule.coordinates, copy=True)
    central_position = coords[nitrile_carbon]
    substituent_vector = coords[methyl_carbon] - central_position
    substituent_unit = substituent_vector / np.linalg.norm(substituent_vector)
    reference_axis = np.eye(3)[np.argmin(np.abs(substituent_unit))]
    perpendicular = (
        reference_axis - np.dot(reference_axis, substituent_unit) * substituent_unit
    )
    perpendicular /= np.linalg.norm(perpendicular)
    nitrile_length = np.linalg.norm(coords[nitrogen] - central_position)
    target_radians = np.deg2rad(_NEAR_LINEAR_ANGLE_DEGREES)
    coords[nitrogen] = central_position + nitrile_length * (
        np.cos(target_radians) * substituent_unit
        + np.sin(target_radians) * perpendicular
    )
    molecule.coordinates = coords

    np.testing.assert_allclose(
        _measure_angle_degrees(coords, target_angle),
        _NEAR_LINEAR_ANGLE_DEGREES,
        rtol=0.0,
        atol=1e-10,
    )
    return molecule, target_angle


def create_mock_hessian_angstrom(
    n_atoms: int, k_diagonal: float = 500.0
) -> "np.ndarray[tuple[int, int], np.dtype[np.float64]]":
    """Create a mock Hessian matrix in kcal/mol/Å² units.

    This creates a simple diagonal-dominated Hessian that represents
    harmonic restoring forces.

    Args:
        n_atoms: Number of atoms
        k_diagonal: Force constant for diagonal elements (kcal/mol/Å²)

    Returns:
        Hessian matrix of shape (3*n_atoms, 3*n_atoms) in kcal/mol/Å²
    """
    size = 3 * n_atoms
    hessian = np.zeros((size, size))

    # Set diagonal blocks (self-interaction)
    for i in range(n_atoms):
        block = np.diag([k_diagonal, k_diagonal, k_diagonal])
        hessian[i * 3 : (i + 1) * 3, i * 3 : (i + 1) * 3] = block

    # Set off-diagonal blocks (interactions between atoms)
    # Use a smaller coupling constant
    k_coupling = -k_diagonal / (n_atoms - 1)
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                # Simple coupling along the bond direction would be more realistic,
                # but for testing we use a simple isotropic coupling
                hessian[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3] = np.diag(
                    [k_coupling, k_coupling, k_coupling]
                )

    # Ensure symmetry
    hessian = 0.5 * (hessian + hessian.T)

    return hessian


def create_mock_hessian_atomic_units(
    n_atoms: int, k_diagonal: float = 500.0
) -> "np.ndarray[tuple[int, int], np.dtype[np.float64]]":
    """Create a mock Hessian matrix in atomic units (Hartree/Bohr²).

    QUBEKit expects the Hessian in atomic units and converts internally.

    Args:
        n_atoms: Number of atoms
        k_diagonal: Force constant for diagonal elements in kcal/mol/Å²

    Returns:
        Hessian matrix in Hartree/Bohr²
    """
    # First create in kcal/mol/Å²
    hessian_kcal_angstrom = create_mock_hessian_angstrom(n_atoms, k_diagonal)

    # Convert to atomic units (Hartree/Bohr²)
    # QUBEKit does: hessian *= constants.HA_TO_KCAL_P_MOL / (constants.BOHR_TO_ANGS**2)
    # So to go backwards: hessian_au = hessian_kcal_A2 / (HA_TO_KCAL_P_MOL / BOHR_TO_ANGS**2)
    conversion = constants.HA_TO_KCAL_P_MOL / (constants.BOHR_TO_ANGS**2)
    hessian_au: np.ndarray[tuple[int, int], np.dtype[np.float64]] = (
        hessian_kcal_angstrom / conversion
    )

    return hessian_au


def main() -> None:
    """Generate reference values using QUBEKit's ModSeminario."""
    qubekit_version = version("qubekit")
    if qubekit_version != _QUBEKIT_VERSION:
        raise RuntimeError(
            f"Reference data requires QUBEKit {_QUBEKIT_VERSION}, found {qubekit_version}."
        )

    print("=" * 70)
    print("QUBEKit Modified Seminario Method - Reference Value Generator")
    print("=" * 70)
    print()

    # Create a fully asymmetric halogenated molecule using QUBEKit
    # Fluorochlorobromomethanol: FC(Cl)(Br)O - all atoms unique, no symmetry
    # SMILES: OC(F)(Cl)Br
    print("Creating fluorochlorobromomethanol molecule (fully asymmetric)...")
    mol = Ligand.from_smiles("OC(F)(Cl)Br", "fluorochlorobromomethanol")

    print(f"  Number of atoms: {mol.n_atoms}")
    print(f"  Number of bonds: {mol.n_bonds}")
    print(f"  Number of angles: {mol.n_angles}")
    print()

    # Print coordinates
    print("Coordinates (Angstroms):")
    for i, atom in enumerate(mol.atoms):
        coord = mol.coordinates[i]
        print(
            f"  {i}: {atom.atomic_symbol:2s} [{coord[0]:10.6f}, {coord[1]:10.6f}, {coord[2]:10.6f}]"
        )
    print()

    # Print bonds
    print("Bonds:")
    for bond in mol.bonds:
        print(f"  ({bond.atom1_index}, {bond.atom2_index})")
    print()

    # Print angles
    print("Angles:")
    for angle in mol.angles:
        print(f"  {angle}")
    print()

    # Create mock Hessian in atomic units (QUBEKit's expected input)
    print("Creating mock Hessian in atomic units (Hartree/Bohr²)...")
    hessian_au = create_mock_hessian_atomic_units(mol.n_atoms, k_diagonal=500.0)
    mol.hessian = hessian_au
    print(f"  Hessian shape: {hessian_au.shape}")
    print()

    # Run ModSeminario
    print("Running QUBEKit ModSeminario...")
    mol = _run_mod_seminario(mol)
    print("  Done!")
    print()

    # Extract and print bond parameters
    print("=" * 70)
    print("BOND PARAMETERS (QUBEKit output)")
    print("=" * 70)
    print("Units: length in nm, k in kJ/mol/nm²")
    print()

    bond_results = {}
    for bond in mol.bonds:
        bond_key = (bond.atom1_index, bond.atom2_index)
        param = mol.BondForce[bond_key]
        bond_results[str(bond_key)] = {
            "length_nm": param.length,
            "k_kj_mol_nm2": param.k,
        }
        print(f"  Bond {bond_key}:")
        print(f"    length = {param.length:.6f} nm")
        print(f"    k = {param.k:.2f} kJ/mol/nm²")
    print()

    # Extract and print angle parameters
    print("=" * 70)
    print("ANGLE PARAMETERS (QUBEKit output)")
    print("=" * 70)
    print("Units: angle in radians, k in kJ/mol/rad²")
    print()

    angle_results = {}
    for angle in mol.angles:
        param = mol.AngleForce[angle]
        angle_results[str(angle)] = {
            "angle_rad": param.angle,
            "angle_deg": np.degrees(param.angle),
            "k_kj_mol_rad2": param.k,
        }
        print(f"  Angle {angle}:")
        print(f"    angle = {param.angle:.6f} rad ({np.degrees(param.angle):.2f}°)")
        print(f"    k = {param.k:.2f} kJ/mol/rad²")
    print()

    print("Generating full-pipeline near-linear acetonitrile reference...")
    nitrile, nitrile_angle = create_near_linear_acetonitrile()
    nitrile_coords = np.array(nitrile.coordinates, copy=True)
    nitrile_hessian = create_nondegenerate_hessian_angstrom(nitrile.n_atoms)

    # The target pair blocks must have a unique eigenbasis so this reference is
    # independent of LAPACK's basis choice for degenerate eigenvalues.
    for terminal_atom in (nitrile_angle[0], nitrile_angle[2]):
        pair_block = nitrile_hessian[
            nitrile_angle[1] * 3 : (nitrile_angle[1] + 1) * 3,
            terminal_atom * 3 : (terminal_atom + 1) * 3,
        ]
        eigenvalue_gaps = np.diff(np.sort(np.linalg.eigvalsh(pair_block)))
        assert np.min(np.abs(eigenvalue_gaps)) > 1e-6

    atomic_unit_conversion = constants.HA_TO_KCAL_P_MOL / (constants.BOHR_TO_ANGS**2)
    nitrile.hessian = nitrile_hessian / atomic_unit_conversion
    nitrile = _run_mod_seminario(nitrile)
    np.testing.assert_allclose(nitrile.coordinates, nitrile_coords, atol=0.0, rtol=0.0)
    nitrile_parameter = nitrile.AngleForce[nitrile_angle]
    near_linear_nitrile_reference = {
        "molecule": {
            "name": "near_linear_acetonitrile",
            "smiles": "CC#N",
            "atoms": [
                {"index": index, "element": atom.atomic_symbol}
                for index, atom in enumerate(nitrile.atoms)
            ],
            "bonds": [[bond.atom1_index, bond.atom2_index] for bond in nitrile.bonds],
            "angles": [list(angle) for angle in nitrile.angles],
            "target_angle": list(nitrile_angle),
        },
        "inputs": {
            "coordinates_angstrom": nitrile_coords.tolist(),
            "hessian_kcal_mol_angstrom2": nitrile_hessian.tolist(),
            "target_angle_degrees": _NEAR_LINEAR_ANGLE_DEGREES,
            "vibrational_scaling": 1.0,
        },
        "qubekit_output": {
            "angle_radians": float(nitrile_parameter.angle),
            "k_kj_mol_rad2": float(nitrile_parameter.k),
            "potential": "U = k * (theta - theta0)**2 / 2",
        },
        "provenance": {
            "qubekit_version": qubekit_version,
            "numpy_version": np.__version__,
            "pydantic_version": version("pydantic"),
            "openff_toolkit_version": version("openff.toolkit"),
            "openmm_version": version("openmm"),
            "qubekit_source": "https://github.com/qubekit/QUBEKit/blob/2.1.1/qubekit/bonded/mod_seminario.py",
            "generation_command": "conda run -n qubekit-2.1.1 python presto/data/msm/generate_qubekit_reference.py",
            "generation": "Full ModSeminario.run pipeline; final molecule.AngleForce parameter",
            "hessian": "Stored nondegenerate anisotropic PSD block Laplacian",
        },
    }
    print(
        "  C-C#N angle: "
        f"{np.degrees(nitrile_parameter.angle):.8f} degrees, "
        f"k={nitrile_parameter.k:.8f} kJ/mol/rad^2"
    )
    print()

    # Print summary as Python dict for copy-paste
    print("=" * 70)
    print("PYTHON REFERENCE DATA (copy-paste into test file)")
    print("=" * 70)
    print()

    # Coordinates
    print("# Fluorochlorobromomethanol coordinates in Angstroms")
    print("REFERENCE_COORDS_ANGSTROM = np.array([")
    for coord in mol.coordinates:
        print(f"    [{coord[0]:12.8f}, {coord[1]:12.8f}, {coord[2]:12.8f}],")
    print("])")
    print()

    # Bonds
    print("# Fluorochlorobromomethanol bonds (0-indexed atom pairs)")
    print("REFERENCE_BONDS = [")
    for bond in mol.bonds:
        print(f"    ({bond.atom1_index}, {bond.atom2_index}),")
    print("]")
    print()

    # Angles
    print("# Fluorochlorobromomethanol angles (central atom is middle index)")
    print("REFERENCE_ANGLES = [")
    for angle in mol.angles:
        print(f"    {angle},")
    print("]")
    print()

    # Bond reference values
    print("# QUBEKit bond parameters")
    print("# Units: length in nm, k in kJ/mol/nm² (OpenMM convention: U = k*(r-r0)²/2)")
    print("QUBEKIT_BOND_PARAMS = {")
    for bond in mol.bonds:
        bond_key = (bond.atom1_index, bond.atom2_index)
        param = mol.BondForce[bond_key]
        print(f"    {bond_key}: {{'length': {param.length:.10f}, 'k': {param.k:.6f}}},")
    print("}")
    print()

    # Angle reference values
    print("# QUBEKit angle parameters")
    print(
        "# Units: angle in degrees, k in kJ/mol/rad² "
        "(OpenMM convention: U = k*(theta-theta0)²/2)"
    )
    print("QUBEKIT_ANGLE_PARAMS = {")
    for angle in mol.angles:
        param = mol.AngleForce[angle]
        print(
            f"    {angle}: {{'angle': {np.degrees(param.angle):.10f}, 'k': {param.k:.6f}}},"
        )
    print("}")
    print()

    # Save to JSON file
    output_data = {
        "coordinates_angstrom": mol.coordinates.tolist(),
        "bonds": [(b.atom1_index, b.atom2_index) for b in mol.bonds],
        "angles": list(mol.angles),
        "bond_params": bond_results,
        "angle_params": angle_results,
        "near_linear_nitrile_reference": near_linear_nitrile_reference,
        "notes": {
            "qubekit_version": qubekit_version,
            "qubekit_source": "https://github.com/qubekit/QUBEKit/blob/2.1.1/qubekit/bonded/mod_seminario.py",
            "hessian_type": "mock_diagonal_dominated",
            "hessian_k_diagonal_kcal_mol_A2": 500.0,
            "vibrational_scaling": 1.0,
            "units": {
                "length": "nm",
                "bond_k": "kJ/mol/nm² (OpenMM convention: U = k*(r-r0)²/2)",
                "angle": "radians (also provided in degrees)",
                "angle_k": "kJ/mol/rad² (OpenMM convention: U = k*(theta-theta0)²/2)",
            },
        },
    }

    output_path = Path(__file__).parent / "qubekit_reference_values.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Reference values saved to: {output_path}")
    print()

    print("=" * 70)
    print("UNIT CONVERSION NOTES")
    print("=" * 70)
    print(
        """
QUBEKit internal workflow:
1. Input Hessian is in atomic units (Hartree/Bohr²)
2. Converts to kcal/mol/Å² using: hessian *= HA_TO_KCAL_P_MOL / BOHR_TO_ANGS²
3. ModSeminario calculates force constants in kcal/mol/Å² (bonds) or kcal/mol/rad² (angles)
4. Output is converted to OpenMM units:
   - Bonds: kJ/mol/nm² using KCAL_TO_KJ * 200 (= 4.184 * 200 = 836.8)
     Factor of 200 = 100 (Å² → nm²) x 2 (potential convention)
   - Angles: kJ/mol/rad² using KCAL_TO_KJ * 2 (= 4.184 * 2 = 8.368)
     Factor of 2 is for potential convention

QUBEKit's internal ``0.5`` factors and final ``2`` conversion factors cancel.
Its final values use the same U = (k/2)*delta² convention as OpenMM and
OpenFF/SMIRNOFF, so final force constants should be compared directly after
unit conversion.
"""
    )


if __name__ == "__main__":
    main()
