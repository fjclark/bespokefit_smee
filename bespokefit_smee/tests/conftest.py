from pathlib import Path

import pytest
from openff.toolkit import Molecule


# From Simon Boothroyd
@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.fixture
def jnk1_lig():
    return Molecule.from_smiles(
        "C(C(Oc1nc(c(c(N([H])[H])c1C#N)[H])N(C(=O)C(c1c(c(C([H])([H])[H])c(c(c1[H])[H])[H])[H])([H])[H])[H])([H])[H])([H])([H])[H]"
    )
