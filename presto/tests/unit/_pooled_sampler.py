"""A stand-in sampling protocol for exercising the real worker pool.

A spawned worker looks its sampling function up by settings type in its own
process, so the stand-in cannot be monkeypatched in from the parent. Unpickling
``PooledSamplingSettings`` imports this module in the worker, and importing it is
what registers the function below.
"""

from __future__ import annotations

import os
import pathlib
from typing import Literal

import datasets
import openff.toolkit
import torch

from presto import sample
from presto.outputs import OutputType
from presto.sample import _register_sampling_fn
from presto.settings import _SamplingSettingsBase


class PooledSamplingSettings(_SamplingSettingsBase):
    """Settings selecting the stand-in sampling function below."""

    sampling_protocol: Literal["pooled_test"] = "pooled_test"

    @property
    def output_types(self) -> set[OutputType]:
        return set()


@_register_sampling_fn(PooledSamplingSettings)
def sample_pooled_test(
    *,
    mols: list[openff.toolkit.Molecule],
    off_ff: openff.toolkit.ForceField,
    device: torch.device,
    settings: PooledSamplingSettings,
    output_paths: dict[OutputType, pathlib.Path],
) -> list[datasets.Dataset]:
    """Report where and how the worker was called instead of running dynamics."""
    return [
        datasets.Dataset.from_dict(
            {
                "smiles": [mol.to_smiles()],
                "mol_index": [sample._MOL_INDEX_OFFSET],
                "pid": [os.getpid()],
                "device": [str(device)],
                "n_ff_parameters": [
                    len(off_ff.get_parameter_handler("Bonds").parameters)
                ],
            }
        )
        for mol in mols
    ]
