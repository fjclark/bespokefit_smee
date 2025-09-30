"""Functionality for finding and sampling torsions in a molecule."""

from openff.toolkit.topology import Molecule

# SMARTS pattern to find rotatable bonds in a molecule
_TORSIONS_TO_INCLUDE_SMARTS: list[str] = [
    "[!#1:1]~[!$(*#*)&!D1:2]-!@[!$(*#*)&!D1:3]~[!#1:4]"
]
# _TORSIONS_TO_EXCLUDE_SMARTS = ["[#6X3:1](=[#8X1])-[#7X3:2]"]  # Amides
_TORSIONS_TO_EXCLUDE_SMARTS: list[str] = []


def get_single_torsion_by_rot_bond(
    mol: Molecule,
    smarts: str,
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """
    Get a single torsion for each rotatable bond matching the provided SMARTS pattern.

    Parameters
    ----------
    mol : openff.toolkit.topology.Molecule
        The molecule to search.
    smarts : str
        SMARTS pattern to match rotatable bonds. This should specify the entire
        torsion, not just the rotatable bond.

    Returns
    -------
    dict of tuple of int to tuple of int
        A dictionary mapping each rotatable bond (as a tuple of atom indices) to a single torsion
        (as a tuple of four atom indices).
    """
    all_torsions = mol.chemical_environment_matches(smarts, unique=True)
    torsions_by_rot_bonds = {}

    for torsion in all_torsions:
        if len(torsion) != 4:
            raise ValueError(
                f"Expected torsion to have 4 atoms, but got {len(torsion)}: {torsion}."
                " Ensure the SMARTS patterns match full torsions."
            )

        rot_bond = tuple(
            sorted((torsion[1], torsion[2]))
        )  # Middle two atoms are the rotatable bond
        if rot_bond not in torsions_by_rot_bonds:
            torsions_by_rot_bonds[rot_bond] = torsion
        else:
            # If we already have a torsion for this rotatable bond, skip it
            continue

    return torsions_by_rot_bonds


def get_unwanted_bonds(mol: Molecule, smarts: str) -> set[tuple[int, int]]:
    """
    Get a set of unwanted bonds in the molecule based on the provided SMARTS patterns.

    Parameters
    ----------
    mol : openff.toolkit.topology.Molecule
        The molecule to search.
    smarts : str
        SMARTS pattern to match unwanted bonds. This should match only the rotatable bond,
        not the full torsion.

    Returns
    -------
    set of tuple of int
        A set of tuples representing the unwanted bonds, where each tuple contains the indices of the two
        atoms forming the bond.
    """
    bonds = mol.chemical_environment_matches(smarts, unique=True)
    for bond in bonds:
        if len(bond) != 2:
            raise ValueError(
                f"Expected bond to have 2 atoms, but got {len(bond)}: {bond}."
                " Ensure the SMARTS pattern matches only the rotatable bond."
            )

    return {tuple(sorted(bond)) for bond in bonds}


def get_rot_torsions_by_rot_bond(
    molecule: Molecule,
    include_smarts: list[str] = _TORSIONS_TO_INCLUDE_SMARTS,
    exclude_smarts: list[str] = _TORSIONS_TO_EXCLUDE_SMARTS,
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """
    Find rotatable torsions in the molecule based on SMARTS patterns.

    Parameters
    ----------
    molecule : openff.toolkit.topology.Molecule
        The molecule to search.
    include_smarts : list of str, optional
        List of SMARTS patterns to include. Defaults to _TORSIONS_TO_INCLUDE_SMARTS.
        These should match the entire torsion, not just the rotatable bond.
    exclude_smarts : list of str, optional
        List of SMARTS patterns to exclude. Defaults to _TORSIONS_TO_EXCLUDE_SMARTS.
        These should match only the rotatable bond, not the full torsion.

    Returns
    -------
    dict of tuple of int to tuple of int
        A dictionary mapping each rotatable bond (as a tuple of atom indices) to a single torsion
        (as a tuple of four atom indices).
    """

    torsions_by_rot_bonds = {}

    for smarts in include_smarts:
        torsions = get_single_torsion_by_rot_bond(molecule, smarts)
        torsions_by_rot_bonds.update(torsions)

    for smarts in exclude_smarts:
        unwanted_bonds = get_unwanted_bonds(molecule, smarts)
        print(f"Excluding unwanted bonds: {unwanted_bonds}")
        for rot_bond in unwanted_bonds:
            torsions_by_rot_bonds.pop(rot_bond, None)

    return torsions_by_rot_bonds
