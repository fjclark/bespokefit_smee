"""Typing utilities for the bespokefit_smee package."""

from pathlib import Path
from typing import Literal

PathLike = str | Path
TorchDevice = Literal["cpu", "cuda"]
