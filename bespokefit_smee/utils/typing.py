"""Typing utilities for the bespokefit_smee package."""

from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

PathLike = str | Path
TorchDevice = Literal["cpu", "cuda"]

OptimiserName = Literal["adam", "lm"]
"""Allowed optimiser names. 'adam' is Adam, 'lm' is Levenberg-Marquardt."""

FnTypeVar = TypeVar("FnTypeVar", bound=Callable[..., Any])
