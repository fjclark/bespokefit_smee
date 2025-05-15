from pathlib import Path

import pytest


# From Simon Boothroyd
@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    yield tmp_path
