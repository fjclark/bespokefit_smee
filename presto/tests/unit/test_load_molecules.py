"""Unit tests for load_molecules.py, focusing on starting-conformer loading."""

import numpy as np
import pytest
from openff.toolkit import Molecule
from openff.units import unit
from scipy.spatial.distance import pdist

from presto.load_molecules import (
    find_problematic_functional_groups,
    load_conformers_for_molecule,
)


def _sorted_pairwise_distances(conformer: unit.Quantity) -> np.ndarray:
    """Atom-relabelling-invariant fingerprint of a conformer's geometry."""
    return np.sort(pdist(conformer.m_as(unit.angstrom)))


def _structures_match(loaded, originals, tol: float = 1e-2) -> bool:
    """Check each loaded conformer matches some original geometry up to atom relabelling."""
    original_fps = [_sorted_pairwise_distances(c) for c in originals]
    for conformer in loaded:
        fp = _sorted_pairwise_distances(conformer)
        if not any(np.max(np.abs(fp - o)) < tol for o in original_fps):
            return False
    return True


class TestFindProblematicFunctionalGroups:
    """Tests for evidence-backed problematic functional-group matching."""

    @pytest.mark.parametrize(
        ("smiles", "pattern"),
        [
            ("COP(=O)(O)O", "[#15]"),
            ("CS(=O)(=O)N", "[SX4](=[OX1])(=[OX1])[NX3]"),
            ("CS(=O)(=O)[NH-]", "[SX4](=[OX1])(=[OX1])[NX2]"),
        ],
    )
    def test_matches_supported_patterns(self, smiles, pattern):
        """Each supported chemical environment matches its intended SMARTS."""
        molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
        assert pattern in find_problematic_functional_groups([molecule])

    @pytest.mark.parametrize("smiles", ["CCO", "[O-]c1ccccc1"])
    def test_does_not_match_unrelated_or_long_range_examples(self, smiles):
        """Unrelated and geometry-dependent examples do not cause warnings."""
        molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
        assert find_problematic_functional_groups([molecule]) == {}

    def test_aggregates_indices_names_and_smiles(self):
        """Descriptions identify all matches by index, name, and SMILES."""
        molecules = [
            Molecule.from_smiles("COP(=O)(O)O"),
            Molecule.from_smiles("CP(=O)(O)O"),
        ]
        molecules[1].name = "phosphonate-2"

        matches = find_problematic_functional_groups(molecules)["[#15]"]

        assert len(matches) == 2
        assert "molecule 0" in matches[0]
        assert "COP(=O)(O)O" in matches[0]
        assert "molecule 1" in matches[1]
        assert "phosphonate-2" in matches[1]


@pytest.fixture
def pentanol_with_conformers():
    """Pentanol with several distinct conformers (rms pruning disabled)."""
    molecule = Molecule.from_smiles("CCCCCO")
    molecule.generate_conformers(n_conformers=4, rms_cutoff=0.0 * unit.angstrom)
    return molecule


def test_loads_all_matching_conformers(
    pentanol_with_conformers, tmp_path, write_multiconformer_sdf
):
    """All records for the molecule are returned as conformers."""
    sdf = tmp_path / "confs.sdf"
    write_multiconformer_sdf(pentanol_with_conformers, sdf)

    target = Molecule.from_smiles("CCCCCO")
    conformers = load_conformers_for_molecule(target, sdf)

    assert len(conformers) == pentanol_with_conformers.n_conformers
    assert all(c.shape == (target.n_atoms, 3) for c in conformers)
    assert _structures_match(conformers, pentanol_with_conformers.conformers)


def test_atom_order_is_aligned_to_target(
    pentanol_with_conformers, tmp_path, write_multiconformer_sdf
):
    """Records with a permuted atom order are realigned to the target's ordering."""
    n_atoms = pentanol_with_conformers.n_atoms
    permutation = list(range(n_atoms))
    np.random.default_rng(1).shuffle(permutation)
    mapping = {i: permutation[i] for i in range(n_atoms)}
    remapped = pentanol_with_conformers.remap(mapping, current_to_new=True)

    sdf = tmp_path / "permuted.sdf"
    write_multiconformer_sdf(remapped, sdf)

    target = Molecule.from_smiles("CCCCCO")
    conformers = load_conformers_for_molecule(target, sdf)

    # Returned conformers use the target's atom count/order, and the physical geometry is
    # preserved despite the permuted input ordering.
    assert len(conformers) == pentanol_with_conformers.n_conformers
    assert all(c.shape == (target.n_atoms, 3) for c in conformers)
    assert _structures_match(conformers, pentanol_with_conformers.conformers)


def test_multi_molecule_sdf_returns_only_matching(tmp_path, write_multiconformer_sdf):
    """An SDF holding conformers of two molecules yields only the matching ones."""
    pentanol = Molecule.from_smiles("CCCCCO")
    pentanol.generate_conformers(n_conformers=3, rms_cutoff=0.0 * unit.angstrom)
    butane = Molecule.from_smiles("CCCC")
    butane.generate_conformers(n_conformers=2, rms_cutoff=0.0 * unit.angstrom)

    sdf = tmp_path / "mixed.sdf"
    write_multiconformer_sdf([pentanol, butane], sdf)

    pentanol_target = Molecule.from_smiles("CCCCCO")
    butane_target = Molecule.from_smiles("CCCC")

    assert (
        len(load_conformers_for_molecule(pentanol_target, sdf)) == pentanol.n_conformers
    )
    assert len(load_conformers_for_molecule(butane_target, sdf)) == butane.n_conformers


def test_no_matching_molecule_raises(
    pentanol_with_conformers, tmp_path, write_multiconformer_sdf
):
    """A molecule absent from the SDF raises a clear error."""
    sdf = tmp_path / "confs.sdf"
    write_multiconformer_sdf(pentanol_with_conformers, sdf)

    benzene = Molecule.from_smiles("c1ccccc1")
    with pytest.raises(ValueError, match="no conformers matching"):
        load_conformers_for_molecule(benzene, sdf)


def test_missing_file_raises(tmp_path):
    """A missing SDF path raises."""
    target = Molecule.from_smiles("CCO")
    with pytest.raises(ValueError, match="does not exist"):
        load_conformers_for_molecule(target, tmp_path / "missing.sdf")


def test_non_sdf_suffix_raises(tmp_path):
    """A non-.sdf path raises."""
    target = Molecule.from_smiles("CCO")
    other = tmp_path / "confs.mol2"
    other.write_text("")
    with pytest.raises(ValueError, match=r"ending in \.sdf"):
        load_conformers_for_molecule(target, other)
