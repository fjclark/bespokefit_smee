"""Functionality for creating Open"""

import atexit
import os
import tempfile
import urllib.request
from typing import Literal, get_args

import loguru
from openmmml import MLPotential

from .utils import aimnet2

logger = loguru.logger


AvailableModels = Literal[
    "mace-off23-small",
    "mace-off23-medium",
    "mace-off23-large",
    "egret-1",
    "aimnet2_b973c_d3",
    "aimnet2_wb97m_d3",
]


def get_egret_1() -> MLPotential:
    """Get the Egret-1 MLPotential from GitHub."""
    # Model accessed 24/05/25
    url = "https://github.com/rowansci/egret-public/raw/227d6641e6851eb1037d48712462e4ce61c1518f/compiled_models/EGRET_1.model"
    tmp_file = tempfile.NamedTemporaryFile(suffix=".model", delete=False)
    tmp_file.close()  # Close so urllib can write to it
    logger.info(f"Downloading Egret-1 model from {url}")
    urllib.request.urlretrieve(url, filename=tmp_file.name)

    # Register file for deletion at program exit
    atexit.register(
        lambda: os.remove(tmp_file.name) if os.path.exists(tmp_file.name) else None
    )

    return MLPotential("mace", modelPath=tmp_file.name)


def get_mlp(model: AvailableModels) -> MLPotential:
    """Get the MLPotential model based on the specified model name."""

    if model not in get_args(AvailableModels):
        raise ValueError(
            f"Invalid model name: {model}. Available models are: {get_args(AvailableModels)}"
        )

    if model in aimnet2._AVAILABLE_MODELS:
        # Ensure AIMNet2 models registered
        aimnet2._register_aimnet2_potentials()

    if model == "egret-1":
        return get_egret_1()
    else:
        return MLPotential(model)
