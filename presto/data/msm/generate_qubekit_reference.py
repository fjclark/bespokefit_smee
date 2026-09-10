"""Generate reference MSM values using QUBEKit's ModSeminario implementation.

This script uses QUBEKit directly to compute bond and angle parameters using
the Modified Seminario Method, which can then be used as reference values
for testing our implementation.

Usage:
    conda run -n qubekit python presto/data/msm/generate_qubekit_reference.py

Requirements:
    - QUBEKit must be installed (available in the 'qubekit' conda environment)

The script will:
    1. Create an asymmetric halogenated molecule with QUBEKit
    2. Generate a mock Hessian matrix
    3. Run QUBEKit's ModSeminario method
    4. Output the resulting bond and angle parameters

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
    symmetric atoms share the same SMIRKS types).
"""

import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

# Check if QUBEKit is available
try:
    from qubekit.bonded import ModSeminario
    from qubekit.bonded.mod_seminario import ModSemMaths
    from qubekit.molecules import Ligand
    from qubekit.utils import constants
except ImportError:
    print("ERROR: QUBEKit is not installed in this environment.")
    print("Please run this script using:")
    print(
        "    conda run -n qubekit python presto/data/msm/generate_qubekit_reference.py"
    )
    sys.exit(1)


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
    for i, (atom, coord) in enumerate(zip(mol.atoms, mol.coordinates, strict=True)):
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
    mod_sem = ModSeminario(vibrational_scaling=1.0)
    mol = mod_sem.run(molecule=mol)
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

    # Exercise QUBEKit's linear-angle helper directly. Running an exactly linear
    # molecule through the complete QUBEKit stage is not possible because its
    # scaling-factor setup constructs a plane normal before dispatching to the
    # special case. The final stage multiplies the helper result by two.
    linear_u_ab = np.array([1.0, 0.0, 0.0])
    linear_u_cb = np.array([-1.0, 0.0, 0.0])
    linear_bond_lengths = [1.0, 1.0]
    linear_eigenvalues = [
        np.array([100.0, 50.0, 50.0]),
        np.array([100.0, 50.0, 50.0]),
    ]
    linear_eigenvectors = [
        np.eye(3, dtype=complex),
        np.eye(3, dtype=complex),
    ]
    linear_raw_k, linear_angle = ModSemMaths.f_c_a_special_case(
        linear_u_ab,
        linear_u_cb,
        linear_bond_lengths,
        linear_eigenvalues,
        linear_eigenvectors,
    )
    linear_angle_reference = {
        "u_ab": linear_u_ab.tolist(),
        "u_cb": linear_u_cb.tolist(),
        "bond_lengths": linear_bond_lengths,
        "eigenvalues": [values.tolist() for values in linear_eigenvalues],
        "eigenvectors": [np.real(vectors).tolist() for vectors in linear_eigenvectors],
        "n_samples": 200,
        "helper_k_kcal_mol_rad2": float(linear_raw_k),
        "final_k_kcal_mol_rad2": float(linear_raw_k * 2.0),
        "angle_degrees": float(linear_angle),
        "notes": "Final k includes QUBEKit calculate_angles conversion factor of 2.",
    }
    near_linear_u_cb = np.array(
        [np.cos(np.deg2rad(175.0)), np.sin(np.deg2rad(175.0)), 0.0]
    )
    near_linear_raw_k, near_linear_angle = ModSemMaths.f_c_a_special_case(
        linear_u_ab,
        near_linear_u_cb,
        linear_bond_lengths,
        linear_eigenvalues,
        linear_eigenvectors,
    )
    near_linear_angle_reference = {
        **linear_angle_reference,
        "u_cb": near_linear_u_cb.tolist(),
        "helper_k_kcal_mol_rad2": float(near_linear_raw_k),
        "final_k_kcal_mol_rad2": float(near_linear_raw_k * 2.0),
        "angle_degrees": float(near_linear_angle),
    }

    # Print summary as Python dict for copy-paste
    print("=" * 70)
    print("PYTHON REFERENCE DATA (copy-paste into test file)")
    print("=" * 70)
    print()

    # Coordinates
    print("# Ethanol coordinates in Angstroms (QUBEKit native format)")
    print("ETHANOL_COORDS_ANGSTROM = np.array([")
    for coord in mol.coordinates:
        print(f"    [{coord[0]:12.8f}, {coord[1]:12.8f}, {coord[2]:12.8f}],")
    print("])")
    print()

    # Bonds
    print("# Ethanol bonds (0-indexed atom pairs)")
    print("ETHANOL_BONDS = [")
    for bond in mol.bonds:
        print(f"    ({bond.atom1_index}, {bond.atom2_index}),")
    print("]")
    print()

    # Angles
    print("# Ethanol angles (central atom is middle index)")
    print("ETHANOL_ANGLES = [")
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
        "linear_angle_reference": linear_angle_reference,
        "near_linear_angle_reference": near_linear_angle_reference,
        "notes": {
            "qubekit_version": version("qubekit"),
            "qubekit_source": "https://github.com/qubekit/QUBEKit/blob/main/qubekit/bonded/mod_seminario.py",
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
