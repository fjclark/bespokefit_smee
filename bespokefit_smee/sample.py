"""
Functionality to obtain samples to fit the force field to.
"""

import copy
import functools
import pathlib
from typing import Callable, Protocol, TypedDict, Unpack

import datasets
import datasets.table
import descent.targets.energy
import loguru
import numpy
import numpy as np
import openff.interchange
import openff.toolkit
import openmm
import torch
from openff.units import unit as off_unit
from openmm import LangevinMiddleIntegrator
from openmm.app import PDBReporter, Simulation
from openmm.unit import Quantity, angstrom
from tqdm import tqdm

from . import mlp, settings
from .find_torsions import (
    _TORSIONS_TO_EXCLUDE_SMARTS,
    _TORSIONS_TO_INCLUDE_SMARTS,
    get_rot_torsions_by_rot_bond,
)
from .metadynamics import Metadynamics
from .outputs import OutputType
from .utils.register import get_registry_decorator

logger = loguru.logger

_ANGSTROM = off_unit.angstrom

_OMM_KELVIN = openmm.unit.kelvin
_OMM_PS = openmm.unit.picosecond
_OMM_ANGS = openmm.unit.angstrom
_OMM_KCAL_PER_MOL = openmm.unit.kilocalorie_per_mole
_OMM_KCAL_PER_MOL_ANGS = openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom


class SampleFnArgs(TypedDict):
    """Arguments for sampling functions."""

    mol: openff.toolkit.Molecule
    off_ff: openff.toolkit.ForceField
    device: torch.device
    settings: settings.SamplingSettings
    output_paths: dict[OutputType, pathlib.Path]


class SampleFn(Protocol):
    """A protocol for sampling functions."""

    def __call__(self, **kwargs: Unpack[SampleFnArgs]) -> datasets.Dataset: ...


_SAMPLING_FNS_REGISTRY: dict[type[settings.SamplingSettings], SampleFn] = {}
"""Registry of sampling functions for different sampling settings types."""

_register_sampling_fn = get_registry_decorator(_SAMPLING_FNS_REGISTRY)


def _copy_mol_and_add_conformers(
    mol: openff.toolkit.Molecule,
    n_conformers: int,
) -> openff.toolkit.Molecule:
    """Copy a molecule and add conformers to it."""
    mol = copy.deepcopy(mol)
    mol.generate_conformers(n_conformers=n_conformers, rms_cutoff=0.0 * _ANGSTROM)
    n_gen_conformers = len(mol.conformers)
    if n_gen_conformers < n_conformers:
        logger.warning(
            f"Only {n_gen_conformers} conformers were generated, which is less than the requested {n_conformers}."
            f" As a result, {n_gen_conformers / n_conformers * 100:.1f}% of the requested samples will be generated."
        )
    return mol


def _get_integrator(
    temp: openmm.unit.Quantity, timestep: openmm.unit.Quantity
) -> LangevinMiddleIntegrator:
    return LangevinMiddleIntegrator(temp, 1 / _OMM_PS, timestep)


def _run_md(
    mol: openff.toolkit.Molecule,
    simulation: Simulation,
    step_fn: Callable[[int], None],
    equilibration_n_steps_per_conformer: int,
    production_n_snapshots_per_conformer: int,
    production_n_steps_per_snapshot_per_conformer: int,
    pdb_reporter_path: str | None = None,
) -> datasets.Dataset:
    """Run MD on a molecule and return a dataset of the coordinates,
    energies, and forces of the snapshots.

    Parameters
    ----------
    mol : openff.toolkit.Molecule
        The molecule to run MD on. Should have conformers already generated.

    simulation : openmm.app.Simulation
        The OpenMM simulation to use for MD.

    step_fn : Callable[[int], None]
        A function that takes the number of steps to run and runs them
        in the simulation. This is to allow for different types of MD
        (e.g. with or without metadynamics).

    equilibration_n_steps_per_conformer : int
        The number of equilibration steps to run per conformer.

    production_n_snapshots_per_conformer : int
        The number of production snapshots to take per conformer.

    production_n_steps_per_snapshot_per_conformer : int
        The number of production steps to run between each snapshot
        per conformer.

    pdb_reporter_path : str | None, optional
        The path to write a PDB trajectory of the MD
        simulation to. The frames saved correspond
        to the production snapshots. If None, no trajectory is saved.

    Returns
    -------
    datasets.Dataset
        The dataset of snapshots with coordinates, energies, and forces.
    """

    coords, energy, forces = [], [], []
    if pdb_reporter_path is not None:
        reporter = PDBReporter(
            pdb_reporter_path, production_n_steps_per_snapshot_per_conformer
        )

    for conf_idx, initial_positions in tqdm(
        enumerate(mol.conformers),
        leave=False,
        colour="green",
        desc="Generating Snapshots",
        total=len(mol.conformers),
    ):
        simulation.context.setPositions(initial_positions.to_openmm())

        # Equilibration
        simulation.minimizeEnergy(maxIterations=100)
        step_fn(equilibration_n_steps_per_conformer)

        # Production
        if pdb_reporter_path is not None:
            simulation.reporters.append(reporter)

        for _ in tqdm(
            range(production_n_snapshots_per_conformer),
            leave=False,
            colour="red",
            desc=f"Running MD for conformer {conf_idx + 1}",
        ):
            step_fn(production_n_steps_per_snapshot_per_conformer)
            state = simulation.context.getState(
                getEnergy=True, getForces=True, getPositions=True
            )
            coords.append(state.getPositions().value_in_unit(_OMM_ANGS))
            energy.append(state.getPotentialEnergy().value_in_unit(_OMM_KCAL_PER_MOL))
            forces.append(
                state.getForces(asNumpy=True).value_in_unit(_OMM_KCAL_PER_MOL_ANGS)
            )

        # Remove the reporter to avoid the next round of equilibration sampling
        if pdb_reporter_path is not None:
            simulation.reporters.remove(reporter)

    # Return a Dataset with energies relative to the first snapshot
    smiles = mol.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)
    coords_out = torch.tensor(np.array(coords))
    energy_0 = energy[0]
    energy_out = torch.tensor(np.array([x - energy_0 for x in energy]))
    forces_out = torch.tensor(np.array(forces))

    return descent.targets.energy.create_dataset(
        [
            {
                "smiles": smiles,
                "coords": coords_out,
                "energy": energy_out,
                "forces": forces_out,
            }
        ]
    )


def _get_ml_omm_system(
    mol: openff.toolkit.Molecule, mlp_name: mlp.AvailableModels
) -> openmm.System:
    """Get an OpenMM system for a molecule using a machine learning potential."""
    potential = mlp.get_mlp(mlp_name)
    # with open("/dev/null", "w") as f:
    #     with redirect_stdout(f):
    system = potential.createSystem(
        mol.to_topology().to_openmm(),
        charge=mol.total_charge.m_as(off_unit.e),
    )

    return system


def recalculate_energies_and_forces(
    dataset: datasets.Dataset, simulation: Simulation
) -> datasets.Dataset:
    """Recalculate energies and forces for a dataset using a given OpenMM simulation."""

    recalc_energies = []
    recalc_forces = []

    assert len(dataset) == 1, "Dataset should contain exactly one entry."

    entry = dataset[0]
    n_conf = len(entry["energy"])
    coords = entry["coords"].reshape(n_conf, -1, 3)

    for i in tqdm(
        range(n_conf),
        leave=False,
        colour="blue",
        desc="Recalculating energies and forces",
    ):
        my_pos = Quantity(numpy.array(coords[i]), angstrom)
        simulation.context.setPositions(my_pos)
        state = simulation.context.getState(getEnergy=True, getForces=True)
        recalc_energies.append(
            state.getPotentialEnergy().value_in_unit(_OMM_KCAL_PER_MOL)
        )
        recalc_forces.append(
            state.getForces(asNumpy=True).value_in_unit(_OMM_KCAL_PER_MOL_ANGS)
        )

    return descent.targets.energy.create_dataset(
        [
            {
                "smiles": entry["smiles"],
                "coords": entry["coords"],
                "energy": torch.tensor(recalc_energies),
                "forces": torch.tensor(recalc_forces),
            }
        ]
    )


@_register_sampling_fn(settings.MMMDSamplingSettings)
def sample_mmmd(
    mol: openff.toolkit.Molecule,
    off_ff: openff.toolkit.ForceField,
    device: torch.device,
    settings: settings.MMMDSamplingSettings,
    output_paths: dict[OutputType, pathlib.Path],
) -> datasets.Dataset:
    """Generate a dataset of samples from MD with the given MM force field,
    and energies and forces of snapshots from the ML potential.

    Parameters
    ----------
    mol : openff.toolkit.Molecule
        The molecule to sample.
    off_ff : openff.toolkit.ForceField
        The MM force field to use for sampling.
    device : torch.device
        The device to use for any MD or ML calculations.
    settings : _SamplingSettings
        The sampling settings to use.
    output_paths: dict[OutputType, PathLike]
        A mapping of output types to filesystem paths.

    Returns
    -------
    datasets.Dataset
        The generated dataset of samples with energies and forces.
    """
    mol = _copy_mol_and_add_conformers(mol, settings.n_conformers)
    interchange = openff.interchange.Interchange.from_smirnoff(
        off_ff, openff.toolkit.Topology.from_molecules(mol)
    )
    system = interchange.to_openmm_system()
    simulation = Simulation(
        interchange.topology.to_openmm(),
        system,
        _get_integrator(settings.temperature, settings.timestep),
    )

    # First, generate the MD snapshots using the MM potential
    mm_dataset = _run_md(
        mol,
        simulation,
        simulation.step,
        settings.equilibration_n_steps_per_conformer,
        settings.production_n_snapshots_per_conformer,
        settings.production_n_steps_per_snapshot_per_conformer,
        str(output_paths.get(OutputType.PDB_TRAJECTORY, None)),
    )

    # Now, recalculate energies and forces using the ML potential
    ml_system = _get_ml_omm_system(mol, settings.ml_potential)
    ml_simulation = Simulation(
        interchange.topology,
        ml_system,
        _get_integrator(settings.temperature, settings.timestep),
    )
    ml_dataset = recalculate_energies_and_forces(mm_dataset, ml_simulation)

    return ml_dataset


@_register_sampling_fn(settings.MLMDSamplingSettings)
def sample_mlmd(
    mol: openff.toolkit.Molecule,
    off_ff: openff.toolkit.ForceField,
    device: torch.device,
    settings: settings.MLMDSamplingSettings,
    output_paths: dict[OutputType, pathlib.Path],
) -> datasets.Dataset:
    """Generate a dataset of samples (with energies and forces) all
    from MD with an ML potential.

    Parameters
    ----------
    mol : openff.toolkit.Molecule
        The molecule to sample.
    off_ff : openff.toolkit.ForceField
        The MM force field. Kept for consistency with other sampling functions,
        but not used here.
    device : torch.device
        The device to use for any MD or ML calculations.
    settings : _SamplingSettings
        The sampling settings to use.
    output_paths: dict[OutputType, PathLike]
        A mapping of output types to filesystem paths.

    Returns
    -------
    datasets.Dataset
        The generated dataset of samples with energies and forces.
    """
    mol = _copy_mol_and_add_conformers(mol, settings.n_conformers)
    ml_system = _get_ml_omm_system(mol, settings.ml_potential)
    integrator = _get_integrator(settings.temperature, settings.timestep)
    ml_simulation = Simulation(mol.to_topology().to_openmm(), ml_system, integrator)

    # Generate the MD snapshots using the ML potential
    ml_dataset = _run_md(
        mol,
        ml_simulation,
        ml_simulation.step,
        settings.equilibration_n_steps_per_conformer,
        settings.production_n_snapshots_per_conformer,
        settings.production_n_steps_per_snapshot_per_conformer,
    )

    return ml_dataset


def _get_torsion_bias_forces(
    mol: openff.toolkit.Molecule,
    torsions_to_include: list[str] = _TORSIONS_TO_INCLUDE_SMARTS,
    torsions_to_exclude: list[str] = _TORSIONS_TO_EXCLUDE_SMARTS,
    bias_width: float = np.pi / 10,
) -> list[openmm.app.metadynamics.BiasVariable]:
    """
    Find important torsions in a molecule and return a list of BiasVariable objects -
    one for each torsion.

    Args:
        mol: OpenFF Molecule.
        torsions_to_include: List of SMARTS patterns to include.
        torsions_to_exclude: List of SMARTS patterns to exclude.
        bias_width: Width of the bias to apply to each torsion.

    Returns:
        List of BiasVariable objects for each torsion.
    """
    torsions = get_rot_torsions_by_rot_bond(
        mol,
        include_smarts=torsions_to_include,
        exclude_smarts=torsions_to_exclude,
    )

    bias_variables = []

    for torsion in torsions.values():
        # Creat a custom torsion force for each torsion\
        torsion_force = openmm.CustomTorsionForce("theta")
        torsion_force.addTorsion(*torsion, [])

        # Create a BiasVariable for this torsion
        bias_variable = openmm.app.metadynamics.BiasVariable(
            force=torsion_force,
            biasWidth=bias_width,
            minValue=-numpy.pi,  # Torsions are periodic, so -pi to pi
            maxValue=numpy.pi,
            periodic=True,
        )

        bias_variables.append(bias_variable)

    return bias_variables


@_register_sampling_fn(settings.MMMDMetadynamicsSamplingSettings)
def sample_mmmd_metadynamics(
    mol: openff.toolkit.Molecule,
    off_ff: openff.toolkit.ForceField,
    device: torch.device,
    settings: settings.MMMDMetadynamicsSamplingSettings,
    output_paths: dict[OutputType, pathlib.Path],
) -> datasets.Dataset:
    """Generate a dataset of samples from MD with the given MM force field
    with metadynamics samplings of the torsions. Each torsion is treated as an
    independent collective variable and biased independently. This function
    generates samples using the MM potential, and recalculates energies and
    forces of snapshots from the ML potential.

    Parameters
    ----------
    mol : openff.toolkit.Molecule
        The molecule to sample.
    off_ff : openff.toolkit.ForceField
        The MM force field to use for sampling.
    device : torch.device
        The device to use for any MD or ML calculations.
    settings : settings.MMMDMetadynamicsSamplingSettings
        The sampling settings to use.
    output_paths: dict[OutputType, PathLike]
        A mapping of output types to filesystem paths.

    Returns
    -------
    datasets.Dataset
        The generated dataset of samples with energies and forces.
    """
    # Make sure we have all the required output paths and no others
    if set(output_paths.keys()) != settings.output_types:
        raise ValueError(
            f"Output paths must contain exactly the keys {settings.output_types}"
        )

    mol = _copy_mol_and_add_conformers(mol, settings.n_conformers)
    interchange = openff.interchange.Interchange.from_smirnoff(
        off_ff, openff.toolkit.Topology.from_molecules(mol)
    )

    torsions = get_rot_torsions_by_rot_bond(mol)
    if not torsions:
        raise ValueError("No rotatable bonds found in the molecule.")

    # Configure metadynamics
    bias_variables = _get_torsion_bias_forces(
        mol,
        torsions_to_include=_TORSIONS_TO_INCLUDE_SMARTS,
        torsions_to_exclude=_TORSIONS_TO_EXCLUDE_SMARTS,
        bias_width=settings.bias_width,
    )

    system = interchange.to_openmm_system()

    bias_dir = output_paths[OutputType.METADYNAMICS_BIAS]
    bias_dir.mkdir()

    metad = Metadynamics(  # type: ignore[no-untyped-call]
        system=system,
        variables=bias_variables,
        temperature=settings.temperature,
        biasFactor=settings.bias_factor,
        height=settings.bias_height,
        frequency=settings.n_steps_per_bias,
        saveFrequency=settings.n_steps_per_bias_save,
        biasDir=bias_dir,
        independentCVs=True,
    )

    simulation = Simulation(
        interchange.topology.to_openmm(),
        system,
        _get_integrator(settings.temperature, settings.timestep),
    )

    step_fn = functools.partial(metad.step, simulation)

    # First, generate the MD snapshots using the MM potential
    mm_dataset = _run_md(
        mol,
        simulation,
        step_fn,
        settings.equilibration_n_steps_per_conformer,
        settings.production_n_snapshots_per_conformer,
        settings.production_n_steps_per_snapshot_per_conformer,
        str(output_paths.get(OutputType.PDB_TRAJECTORY, None)),
    )

    # Now, recalculate energies and forces using the ML potential
    ml_system = _get_ml_omm_system(mol, settings.ml_potential)
    ml_simulation = Simulation(
        interchange.topology.to_openmm(),
        ml_system,
        _get_integrator(settings.temperature, settings.timestep),
    )
    ml_dataset = recalculate_energies_and_forces(mm_dataset, ml_simulation)

    return ml_dataset
