"""
Utilities for working with AIMNet2. Mainly taken from
https://github.com/openmm/openmm-ml/pull/64 and
https://github.com/SimonBoothroyd/befit/blob/main/befit/utils/aimnet2.py

See discussion at https://github.com/isayevlab/AIMNet2/issues/15 re ensemble
models.

See https://github.com/isayevlab/aimnetcentral/blob/47969eb3e29e34824d82a648dd756669c875ecdb/scripts/compile/compile_off.yaml
for available models. May compile the ensemble models in future.
"""

import tempfile
import urllib.request
from typing import Iterable, Literal, Optional, get_args

import openmm
import openmmtorch
import torch
from openmm import unit
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory

from .typing import TorchDevice

_MODEL_URL = "https://storage.googleapis.com/aimnetcentral/AIMNet2/"
AvailableModels = Literal["aimnet2_b973c_d3", "aimnet2_wb97m_d3"]
_AVAILABLE_MODELS = get_args(AvailableModels)


def _download_model(
    method: str, version: int = 0, device: TorchDevice | None = None
) -> torch.jit.ScriptModule:
    """Download an AIMNet2 model directly from storage."""
    url = f"{_MODEL_URL}{method}_{version}.jpt"

    with tempfile.NamedTemporaryFile(suffix=".jpt") as tmp_file:
        urllib.request.urlretrieve(url, filename=tmp_file.name)
        model: torch.jit.ScriptModule = torch.jit.load(  # type: ignore[no-untyped-call]
            tmp_file.name, map_location=device
        )
        return model


class AIMNet2PotentialImplFactory(MLPotentialImplFactory):  # type: ignore[misc]
    """This is the factory that creates AIMNet2PotentialImpl objects."""

    def createImpl(self, name: str, **args) -> MLPotentialImpl:  # type: ignore[no-untyped-def]
        return AIMNet2PotentialImpl(name)


class AIMNet2PotentialImpl(MLPotentialImpl):  # type: ignore[misc]
    """This is the MLPotentialImpl implementing the AIMNet2 potential."""

    def __init__(self, name: str):
        self.name = name

    def addForces(  # type: ignore[no-untyped-def]
        self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: Optional[Iterable[int]],
        forceGroup: int,
        charge: int,
        **args,
    ) -> None:
        # Load the AIMNet2 model.

        if self.name in _AVAILABLE_MODELS:
            model = _download_model(self.name)
        else:
            raise ValueError(
                f"Unsupported AIMNet2 model: {self.name}. Please use one of: {_AVAILABLE_MODELS}."
            )

        # Create the PyTorch model that will be invoked by OpenMM.

        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
        numbers = torch.tensor([[atom.element.atomic_number for atom in includedAtoms]])
        charge_tensor = torch.tensor([charge], dtype=torch.float32)

        class AIMNet2Force(torch.nn.Module):
            def __init__(
                self,
                model: torch.jit.ScriptModule,
                numbers: torch.Tensor,
                charge: torch.Tensor,
                atoms: Optional[Iterable[int]],
            ):
                super(AIMNet2Force, self).__init__()
                self.model = model
                self.numbers = torch.nn.Parameter(numbers, requires_grad=False)
                self.charge = torch.nn.Parameter(charge, requires_grad=False)
                self.energyScale = (unit.ev / unit.item).conversion_factor_to(
                    unit.kilojoules_per_mole
                )
                if atoms is None:
                    self.indices = None
                else:
                    self.indices = torch.tensor(sorted(atoms), dtype=torch.int64)

            def forward(
                self, positions: torch.Tensor, boxvectors: Optional[torch.Tensor] = None
            ) -> torch.Tensor:
                positions = positions.to(torch.float32)
                if self.indices is not None:
                    positions = positions[self.indices]
                args = {
                    "coord": 10.0 * positions.unsqueeze(0),
                    "numbers": self.numbers,
                    "charge": self.charge,
                }
                result = self.model(args)
                energy: torch.Tensor = self.energyScale * result["energy"]
                return energy

        # Create the TorchForce and add it to the System.

        module = torch.jit.script(AIMNet2Force(model, numbers, charge_tensor, atoms))
        force = openmmtorch.TorchForce(module)
        force.setForceGroup(forceGroup)
        force.setOutputsForces(False)
        system.addForce(force)


def _register_aimnet2_potentials() -> None:
    """Register the AIMNET2 potential implementation factory."""
    for model in _AVAILABLE_MODELS:
        MLPotential.registerImplFactory(model, AIMNet2PotentialImplFactory())
