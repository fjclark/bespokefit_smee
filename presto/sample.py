"""Functionality to obtain samples to fit the force field to."""

import copy
import functools
import pathlib
from collections.abc import Callable
from typing import Protocol, TypedDict, Unpack

import datasets
import datasets.table
import descent.targets.energy
import loguru
import mdtraj
import numpy
import numpy as np
import openff.interchange
import openff.toolkit
import openmm
import torch
from openff.units import unit as off_unit
from openmm import LangevinMiddleIntegrator
from openmm.app import PDBFile, PDBReporter, Simulation
from openmm.unit import Quantity, angstrom
from rich.progress import track

from . import mlp, settings
from .data_utils import (
    create_dataset_with_uniform_weights,
    merge_weighted_datasets,
)
from .find_torsions import (
    DEFAULT_TORSIONS_TO_EXCLUDE_SMARTS,
    DEFAULT_TORSIONS_TO_INCLUDE_SMARTS,
    get_rot_torsions_by_rot_bond,
)
from .load_molecules import load_conformers_for_molecule
from .metadynamics import Metadynamics
from .outputs import OutputType, get_mol_path
from .utils.gpu import cleanup_simulation
from .utils.register import get_registry_decorator

logger = loguru.logger

_ANGSTROM = off_unit.angstrom

_OMM_KELVIN = openmm.unit.kelvin
_OMM_PS = openmm.unit.picosecond
_OMM_ANGS = openmm.unit.angstrom
_OMM_KCAL_PER_MOL = openmm.unit.kilocalorie_per_mole
_OMM_KCAL_PER_MOL_ANGS = openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom
_OMM_KJ_PER_MOL = openmm.unit.kilojoules_per_mole
_OMM_RADIAN = openmm.unit.radian


class SampleFnArgs(TypedDict):
    """Arguments for sampling functions."""

    mols: list[openff.toolkit.Molecule]
    off_ff: openff.toolkit.ForceField
    device: torch.device
    settings: settings.SamplingSettings
    output_paths: dict[OutputType, pathlib.Path]


class SampleFn(Protocol):
    """A protocol for sampling functions."""

    def __call__(self, **kwargs: Unpack[SampleFnArgs]) -> list[datasets.Dataset]:
        """Execute sampling function."""


_SAMPLING_FNS_REGISTRY: dict[type[settings.SamplingSettings], SampleFn] = {}
"""Registry of sampling functions for different sampling settings types."""

_register_sampling_fn = get_registry_decorator(_SAMPLING_FNS_REGISTRY)

_MOL_INDEX_OFFSET: int = 0
"""Set only by a coordinator worker sampling one molecule in isolation, so that its
per-molecule output filenames keep the molecule's index in the whole workflow."""


def _copy_mol_and_add_conformers(
    mol: openff.toolkit.Molecule,
    n_conformers: int,
    starting_conformers: pathlib.Path | None = None,
) -> openff.toolkit.Molecule:
    """Copy a molecule and add starting conformers to it.

    If ``starting_conformers`` is None (default), ``n_conformers`` conformers are
    generated with ETKDG. Otherwise the conformers are loaded from the given SDF (matched
    to ``mol`` by graph and aligned to its atom ordering) and ``n_conformers`` is ignored.
    """
    mol = copy.deepcopy(mol)

    if starting_conformers is None:
        mol.generate_conformers(n_conformers=n_conformers, rms_cutoff=0.0 * _ANGSTROM)
        n_gen_conformers = len(mol.conformers)
        if n_gen_conformers < n_conformers:
            logger.warning(
                f"Only {n_gen_conformers} conformers were generated, which is less than the requested {n_conformers}."
                f" As a result, {n_gen_conformers / n_conformers * 100:.1f}% of the requested samples will be generated."
            )
        return mol

    # Drop any incidental conformer (e.g. one carried by an SDF-loaded molecule) before
    # attaching the supplied starting conformers.
    mol._conformers = []
    conformers = load_conformers_for_molecule(mol, starting_conformers)
    logger.info(
        f"Starting from {len(conformers)} supplied conformers in {starting_conformers} "
        f"(n_conformers={n_conformers} ignored)."
    )
    for conformer in conformers:
        mol.add_conformer(conformer)
    return mol


def _get_integrator(
    temp: openmm.unit.Quantity, timestep: openmm.unit.Quantity
) -> LangevinMiddleIntegrator:
    return LangevinMiddleIntegrator(temp, 1 / _OMM_PS, timestep)


def _create_simulation(
    topology: openmm.app.Topology,
    system: openmm.System,
    temperature: openmm.unit.Quantity,
    timestep: openmm.unit.Quantity,
    device: torch.device,
) -> tuple[Simulation, LangevinMiddleIntegrator]:
    """Create an OpenMM simulation with a standard Langevin integrator."""
    integrator = _get_integrator(temperature, timestep)
    platform = _get_openmm_platform(device)
    properties = None
    if device.type == "cuda" and device.index is not None:
        properties = {"DeviceIndex": str(device.index)}
    simulation = Simulation(
        topology, system, integrator, platform=platform, platformProperties=properties
    )
    return simulation, integrator


def _get_openmm_platform(device: torch.device) -> openmm.Platform:
    """Map a torch device to an OpenMM platform."""
    if device.type == "cuda":
        return openmm.Platform.getPlatformByName("CUDA")

    if device.type == "cpu":
        return openmm.Platform.getPlatformByName("CPU")

    raise ValueError(f"Unsupported device type for OpenMM platform selection: {device}")


def _build_ml_simulation(
    mol: openff.toolkit.Molecule,
    topology: openmm.app.Topology,
    mlp_settings: settings.MLPSettings,
    temperature: openmm.unit.Quantity,
    timestep: openmm.unit.Quantity,
    device: torch.device,
) -> tuple[Simulation, LangevinMiddleIntegrator]:
    """Create a simulation that uses an ML potential system."""
    ml_system = mlp.get_ml_omm_system(
        mol,
        mlp_settings,
        device,
    )

    # As a temporary hack to get around an OpeMM-ML issue, set the device to CPU for the ML simulation
    # This doesn't slow things down as the MLP is still loaded on the GPU when requested.
    return _create_simulation(
        topology, ml_system, temperature, timestep, device=torch.device("cpu")
    )


def _build_mm_simulation(
    interchange: openff.interchange.Interchange,
    temperature: openmm.unit.Quantity,
    timestep: openmm.unit.Quantity,
    device: torch.device,
) -> tuple[Simulation, LangevinMiddleIntegrator]:
    """Create a simulation that uses an MM system from an Interchange object."""
    mm_system = interchange.to_openmm_system()
    mm_topology = interchange.topology.to_openmm()
    return _create_simulation(mm_topology, mm_system, temperature, timestep, device)


def _run_md(
    mol: openff.toolkit.Molecule,
    simulation: Simulation,
    step_fn: Callable[[int], None],
    equilibration_n_steps_per_conformer: int,
    production_n_snapshots_per_conformer: int,
    production_n_steps_per_snapshot_per_conformer: int,
    pdb_reporter_path: str | None = None,
) -> datasets.Dataset:
    """Run MD on a molecule and return a dataset of the coordinates, energies, and forces of the snapshots.

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

    Returns:
    -------
    datasets.Dataset
        The dataset of snapshots with coordinates, energies, and forces.
    """
    coords, energy, forces = [], [], []
    if pdb_reporter_path is not None:
        reporter = PDBReporter(
            pdb_reporter_path, production_n_steps_per_snapshot_per_conformer
        )

    for conf_idx, initial_positions in track(
        enumerate(mol.conformers),
        description="[green]Generating Snapshots",
        total=len(mol.conformers),
    ):
        simulation.context.setPositions(initial_positions.to_openmm())

        # Equilibration
        simulation.minimizeEnergy(maxIterations=100)
        step_fn(equilibration_n_steps_per_conformer)

        # Production
        if pdb_reporter_path is not None:
            simulation.reporters.append(reporter)

        for _ in track(
            range(production_n_snapshots_per_conformer),
            transient=True,
            description=f"[red]Running MD for conformer {conf_idx + 1}",
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
    coords_out = torch.as_tensor(np.array(coords))
    energy_0 = energy[0]
    energy_out = torch.as_tensor(np.array([x - energy_0 for x in energy]))
    forces_out = torch.as_tensor(np.array(forces))

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

    for i in track(
        range(n_conf),
        description="[blue]Recalculating energies and forces",
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
                "energy": torch.tensor(np.array(recalc_energies)),
                "forces": torch.tensor(np.array(recalc_forces)),
            }
        ]
    )


@_register_sampling_fn(settings.MMMDSamplingSettings)
def sample_mmmd(
    mols: list[openff.toolkit.Molecule],
    off_ff: openff.toolkit.ForceField,
    device: torch.device,
    settings: settings.MMMDSamplingSettings,
    output_paths: dict[OutputType, pathlib.Path],
) -> list[datasets.Dataset]:
    """Generate datasets of samples from MD with the given MM force field for multiple molecules.

    Parameters
    ----------
    mols : list[openff.toolkit.Molecule]
        The molecules to sample.
    off_ff : openff.toolkit.ForceField
        The MM force field to use for sampling.
    device : torch.device
        The device to use for any MD or ML calculations.
    settings : _SamplingSettings
        The sampling settings to use.
    output_paths: dict[OutputType, PathLike]
        A mapping of output types to filesystem paths.

    Returns:
    -------
    list[datasets.Dataset]
        The generated datasets of samples with energies and forces, one per molecule.
    """
    if set(output_paths.keys()) != settings.output_types:
        raise ValueError(
            f"Output paths must contain exactly the keys {settings.output_types}"
        )

    all_datasets = []

    for mol_idx, mol in enumerate(mols, start=_MOL_INDEX_OFFSET):
        mol_with_conformers = _copy_mol_and_add_conformers(
            mol, settings.n_conformers, settings.starting_conformers
        )
        interchange = openff.interchange.Interchange.from_smirnoff(
            off_ff, openff.toolkit.Topology.from_molecules(mol_with_conformers)
        )

        simulation, integrator = _build_mm_simulation(
            interchange, settings.temperature, settings.timestep, device
        )

        # Create molecule-specific PDB path
        pdb_path = None
        if OutputType.PDB_TRAJECTORY in output_paths:
            base_path = output_paths[OutputType.PDB_TRAJECTORY]
            pdb_path = str(get_mol_path(base_path, mol_idx))

        mm_dataset = _run_md(
            mol_with_conformers,
            simulation,
            simulation.step,
            settings.equilibration_n_steps_per_conformer,
            settings.production_n_snapshots_per_conformer,
            settings.production_n_steps_per_snapshot_per_conformer,
            pdb_path,
        )

        # Clean up MM simulation to free GPU memory
        cleanup_simulation(simulation, integrator)

        # Recalculate energies and forces using the ML potential
        ml_simulation, ml_integrator = _build_ml_simulation(
            mol_with_conformers,
            interchange.topology,
            settings.mlp_settings,
            settings.temperature,
            settings.timestep,
            device,
        )
        ml_dataset = recalculate_energies_and_forces(mm_dataset, ml_simulation)

        # Clean up ML simulation to free GPU memory
        cleanup_simulation(ml_simulation, ml_integrator)

        # Convert to weighted dataset
        entry = ml_dataset[0]
        n_confs = len(entry["energy"])
        weighted_dataset = create_dataset_with_uniform_weights(
            smiles=entry["smiles"],
            coords=entry["coords"].reshape(n_confs, -1, 3),
            energy=entry["energy"],
            forces=entry["forces"].reshape(n_confs, -1, 3),
            energy_weight=settings.loss_energy_weight,
            forces_weight=settings.loss_force_weight,
        )

        all_datasets.append(weighted_dataset)

    return all_datasets


@_register_sampling_fn(settings.MLMDSamplingSettings)
def sample_mlmd(
    mols: list[openff.toolkit.Molecule],
    off_ff: openff.toolkit.ForceField,
    device: torch.device,
    settings: settings.MLMDSamplingSettings,
    output_paths: dict[OutputType, pathlib.Path],
) -> list[datasets.Dataset]:
    """Generate datasets of samples from MD with an ML potential for multiple molecules.

    Parameters
    ----------
    mols : list[openff.toolkit.Molecule]
        The molecules to sample.
    off_ff : openff.toolkit.ForceField
        The MM force field (kept for consistency).
    device : torch.device
        The device to use for any MD or ML calculations.
    settings : _SamplingSettings
        The sampling settings to use.
    output_paths: dict[OutputType, PathLike]
        A mapping of output types to filesystem paths.

    Returns:
    -------
    list[datasets.Dataset]
        The generated datasets of samples with energies and forces, one per molecule.
    """
    if set(output_paths.keys()) != settings.output_types:
        raise ValueError(
            f"Output paths must contain exactly the keys {settings.output_types}"
        )

    all_datasets = []

    for mol_idx, mol in enumerate(mols, start=_MOL_INDEX_OFFSET):
        mol_with_conformers = _copy_mol_and_add_conformers(
            mol, settings.n_conformers, settings.starting_conformers
        )
        ml_simulation, integrator = _build_ml_simulation(
            mol_with_conformers,
            mol_with_conformers.to_topology().to_openmm(),
            settings.mlp_settings,
            settings.temperature,
            settings.timestep,
            device,
        )

        # Create molecule-specific PDB path
        pdb_path = None
        if OutputType.PDB_TRAJECTORY in output_paths:
            base_path = output_paths[OutputType.PDB_TRAJECTORY]
            pdb_path = str(get_mol_path(base_path, mol_idx))

        ml_dataset = _run_md(
            mol_with_conformers,
            ml_simulation,
            ml_simulation.step,
            settings.equilibration_n_steps_per_conformer,
            settings.production_n_snapshots_per_conformer,
            settings.production_n_steps_per_snapshot_per_conformer,
            pdb_path,
        )

        # Clean up ML simulation to free GPU memory
        cleanup_simulation(ml_simulation, integrator)

        # Convert to weighted dataset
        entry = ml_dataset[0]
        n_confs = len(entry["energy"])
        weighted_dataset = create_dataset_with_uniform_weights(
            smiles=entry["smiles"],
            coords=entry["coords"].reshape(n_confs, -1, 3),
            energy=entry["energy"],
            forces=entry["forces"].reshape(n_confs, -1, 3),
            energy_weight=settings.loss_energy_weight,
            forces_weight=settings.loss_force_weight,
        )

        all_datasets.append(weighted_dataset)

    return all_datasets


def _get_torsion_bias_forces(
    mol: openff.toolkit.Molecule,
    torsions_to_include: list[str] = DEFAULT_TORSIONS_TO_INCLUDE_SMARTS,
    torsions_to_exclude: list[str] = DEFAULT_TORSIONS_TO_EXCLUDE_SMARTS,
    bias_width: float = np.pi / 10,
) -> list[openmm.app.metadynamics.BiasVariable]:
    """Find important torsions in a molecule and return a list of BiasVariable objects, one for each torsion.

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
    mols: list[openff.toolkit.Molecule],
    off_ff: openff.toolkit.ForceField,
    device: torch.device,
    settings: settings.MMMDMetadynamicsSamplingSettings,
    output_paths: dict[OutputType, pathlib.Path],
) -> list[datasets.Dataset]:
    """Generate datasets using metadynamics for multiple molecules.

    Parameters
    ----------
    mols : list[openff.toolkit.Molecule]
        The molecules to sample.
    off_ff : openff.toolkit.ForceField
        The MM force field to use.
    device : torch.device
        The device to use for any MD or ML calculations.
    settings : MMMDMetadynamicsSamplingSettings
        The sampling settings to use.
    output_paths: dict[OutputType, PathLike]
        A mapping of output types to filesystem paths.

    Returns:
    -------
    list[datasets.Dataset]
        The generated datasets of samples with energies and forces, one per molecule.
    """
    if set(output_paths.keys()) != settings.output_types:
        raise ValueError(
            f"Output paths must contain exactly the keys {settings.output_types}"
        )

    all_datasets = []

    for mol_idx, mol in enumerate(mols, start=_MOL_INDEX_OFFSET):
        mol_with_conformers = _copy_mol_and_add_conformers(
            mol, settings.n_conformers, settings.starting_conformers
        )
        interchange = openff.interchange.Interchange.from_smirnoff(
            off_ff, openff.toolkit.Topology.from_molecules(mol_with_conformers)
        )

        torsions = get_rot_torsions_by_rot_bond(
            mol_with_conformers,
            include_smarts=settings.torsions_to_include_smarts,
            exclude_smarts=settings.torsions_to_exclude_smarts,
        )
        if not torsions:
            logger.warning(
                f"No rotatable bonds found in molecule {mol_idx}. Skipping metadynamics."
            )
            # Fall back to regular MD for this molecule
            simulation, integrator = _build_mm_simulation(
                interchange, settings.temperature, settings.timestep, device
            )

            pdb_path = None
            if OutputType.PDB_TRAJECTORY in output_paths:
                base_path = output_paths[OutputType.PDB_TRAJECTORY]
                pdb_path = str(get_mol_path(base_path, mol_idx))

            mm_dataset = _run_md(
                mol_with_conformers,
                simulation,
                simulation.step,
                settings.equilibration_n_steps_per_conformer,
                settings.production_n_snapshots_per_conformer,
                settings.production_n_steps_per_snapshot_per_conformer,
                pdb_path,
            )

            # Clean up MM simulation to free GPU memory
            cleanup_simulation(simulation, integrator)
        else:
            # Setup metadynamics
            bias_variables = _get_torsion_bias_forces(
                mol_with_conformers,
                torsions_to_include=settings.torsions_to_include_smarts,
                torsions_to_exclude=settings.torsions_to_exclude_smarts,
                bias_width=settings.bias_width,
            )

            system = interchange.to_openmm_system()

            # Create molecule-specific bias directory
            base_bias_dir = output_paths[OutputType.METADYNAMICS_BIAS]
            bias_dir = get_mol_path(base_bias_dir, mol_idx)
            bias_dir.mkdir(parents=True, exist_ok=True)

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

            simulation, integrator = _create_simulation(
                interchange.topology.to_openmm(),
                system,
                settings.temperature,
                settings.timestep,
                device,
            )

            step_fn = functools.partial(metad.step, simulation)

            # Create molecule-specific PDB path
            pdb_path = None
            if OutputType.PDB_TRAJECTORY in output_paths:
                base_path = output_paths[OutputType.PDB_TRAJECTORY]
                pdb_path = str(get_mol_path(base_path, mol_idx))

            mm_dataset = _run_md(
                mol_with_conformers,
                simulation,
                step_fn,
                settings.equilibration_n_steps_per_conformer,
                settings.production_n_snapshots_per_conformer,
                settings.production_n_steps_per_snapshot_per_conformer,
                pdb_path,
            )

            # Clean up MM simulation to free GPU memory
            cleanup_simulation(simulation, integrator)

        # Recalculate with ML potential
        ml_simulation, ml_integrator = _build_ml_simulation(
            mol_with_conformers,
            interchange.topology.to_openmm(),
            settings.mlp_settings,
            settings.temperature,
            settings.timestep,
            device,
        )
        ml_dataset = recalculate_energies_and_forces(mm_dataset, ml_simulation)

        # Clean up ML simulation to free GPU memory
        cleanup_simulation(ml_simulation, ml_integrator)

        # Convert to weighted dataset
        entry = ml_dataset[0]
        n_confs = len(entry["energy"])
        weighted_dataset = create_dataset_with_uniform_weights(
            smiles=entry["smiles"],
            coords=entry["coords"].reshape(n_confs, -1, 3),
            energy=entry["energy"],
            forces=entry["forces"].reshape(n_confs, -1, 3),
            energy_weight=settings.loss_energy_weight,
            forces_weight=settings.loss_force_weight,
        )

        all_datasets.append(weighted_dataset)

    return all_datasets


def _get_molecule_from_dataset(
    dataset: datasets.Dataset,
) -> openff.toolkit.Molecule:
    """Extract molecule from dataset using SMILES.

    Parameters
    ----------
    dataset : datasets.Dataset
        Dataset containing SMILES string

    Returns:
    -------
    openff.toolkit.Molecule
        Reconstructed molecule from SMILES
    """
    assert len(dataset) == 1, "Dataset should contain exactly one entry."
    entry = dataset[0]
    smiles = entry["smiles"]
    return openff.toolkit.Molecule.from_smiles(smiles, allow_undefined_stereo=True)


def _find_available_force_group(simulation: Simulation) -> int:
    """Find an unused force group in the simulation system.

    Parameters
    ----------
    simulation : Simulation
        OpenMM simulation object

    Returns:
    -------
    int
        An available force group number (0-31)

    Raises:
    ------
    RuntimeError
        If all force groups (0-31) are in use
    """
    used_groups = set()
    for i in range(simulation.system.getNumForces()):
        force = simulation.system.getForce(i)
        used_groups.add(force.getForceGroup())

    # Find first available group
    for group in range(32):
        if group not in used_groups:
            return group

    raise RuntimeError("All force groups (0-31) are in use")


def _add_torsion_restraint_forces(
    simulation: Simulation,
    torsion_atoms_list: list[tuple[int, int, int, int]],
    force_constant: float,
    initial_angles: list[float] | None = None,
) -> tuple[list[int], int]:
    """Add torsion restraint forces to the simulation system.

    This adds CustomTorsionForce objects that can be updated later
    without reinitializing the context. All restraints are added to
    a dedicated force group.

    Parameters
    ----------
    simulation : Simulation
        OpenMM simulation object
    torsion_atoms_list : list[tuple[int, int, int, int]]
        List of torsion atom indices to freeze
    force_constant : float
        Force constant for torsion restraints (in kJ/mol/rad^2)
    initial_angles : list[float] | None, optional
        Initial target angles for each torsion (in radians).
        If None, defaults to 0.0 for all torsions.

    Returns:
    -------
    tuple[list[int], int]
        Tuple of (list of force indices that were added, force group number)
    """
    if initial_angles is None:
        initial_angles = [0.0] * len(torsion_atoms_list)

    # Find an available force group for the restraints
    restraint_force_group = _find_available_force_group(simulation)
    logger.debug(f"Adding torsion restraints to force group {restraint_force_group}")

    force_indices = []

    for torsion_atoms, target_angle in zip(
        torsion_atoms_list, initial_angles, strict=True
    ):
        restraint_force = openmm.CustomTorsionForce(
            "0.5*k*min(dtheta, 2*pi-dtheta)^2; "
            "dtheta = abs(theta-theta0); pi = 3.1415926535"
        )
        restraint_force.addPerTorsionParameter("k")
        restraint_force.addPerTorsionParameter("theta0")
        restraint_force.addTorsion(*torsion_atoms, [force_constant, target_angle])

        # Assign to dedicated force group
        restraint_force.setForceGroup(restraint_force_group)

        force_idx: int = simulation.system.addForce(restraint_force)
        force_indices.append(force_idx)

    # Only reinitialize once after adding all forces
    simulation.context.reinitialize(preserveState=True)

    return force_indices, restraint_force_group


def _update_torsion_restraints(
    simulation: Simulation,
    force_indices: list[int],
    target_angles: list[float],
    force_constant: float,
) -> None:
    """Update the target angles for torsion restraints without reinitializing.

    Parameters
    ----------
    simulation : Simulation
        OpenMM simulation object
    force_indices : list[int]
        List of force indices for the torsion restraints
    target_angles : list[float]
        New target angles (in radians) for each torsion
    force_constant : float
        Force constant for torsion restraints (in kJ/mol/rad^2)
    """
    for force_idx, target_angle in zip(force_indices, target_angles, strict=True):
        force = simulation.system.getForce(force_idx)
        p1, p2, p3, p4, _ = force.getTorsionParameters(0)
        force.setTorsionParameters(0, p1, p2, p3, p4, [force_constant, target_angle])

    # Update parameters in context without full reinitialize
    for force_idx in force_indices:
        force = simulation.system.getForce(force_idx)
        force.updateParametersInContext(simulation.context)


def _remove_torsion_restraint_forces(
    simulation: Simulation, force_indices: list[int]
) -> None:
    """Remove torsion restraint forces from the simulation.

    Parameters
    ----------
    simulation : Simulation
        OpenMM simulation object
    force_indices : list[int]
        List of force indices to remove
    """
    # Remove in reverse order to maintain correct indices
    for force_idx in sorted(force_indices, reverse=True):
        simulation.system.removeForce(force_idx)

    # Only reinitialize once after removing all forces
    simulation.context.reinitialize(preserveState=True)


def _minimize_with_frozen_torsions(
    simulation: Simulation,
    coords: numpy.ndarray,
    torsion_atoms_list: list[tuple[int, int, int, int]],
    force_indices: list[int],
    torsion_force_constant: float,
    restraint_force_group: int,
    max_iterations: int = 0,
) -> tuple[numpy.ndarray, float, numpy.ndarray]:
    """Minimize a conformation with all torsions frozen.

    Assumes torsion restraint forces have already been added to the system.
    Only updates the target angles without reinitializing.

    Parameters
    ----------
    simulation : Simulation
        OpenMM simulation object (with torsion forces already added)
    coords : numpy.ndarray
        Starting coordinates
    torsion_atoms_list : list[tuple[int, int, int, int]]
        List of torsion atom indices to freeze
    force_indices : list[int]
        Indices of the torsion restraint forces in the system
    torsion_force_constant : float
        Force constant for torsion restraints (in kJ/mol/rad^2)
    restraint_force_group : int
        Force group number for the torsion restraints
    max_iterations : int, optional
        Maximum minimization iterations (0 = until convergence)

    Returns:
    -------
    tuple[numpy.ndarray, float, numpy.ndarray]
        Minimized coordinates, energy, and forces (excluding restraint forces)
    """
    # Set initial positions
    simulation.context.setPositions(Quantity(coords, angstrom))

    # Calculate current angles and update restraint targets using mdtraj
    # Create a minimal mdtraj trajectory for angle calculation
    traj = mdtraj.Trajectory(
        xyz=coords.reshape(1, -1, 3) / 10.0,  # Convert Angstroms to nm
        topology=mdtraj.Topology.from_openmm(simulation.topology),
    )
    current_angles = mdtraj.compute_dihedrals(traj, torsion_atoms_list)[
        0
    ]  # Get angles for the single frame
    current_angles_list = current_angles.tolist()

    _update_torsion_restraints(
        simulation, force_indices, current_angles_list, torsion_force_constant
    )

    # Minimize
    simulation.minimizeEnergy(maxIterations=max_iterations)

    # Get minimized state - exclude restraint force group from energy
    # Create groups mask: all groups except the restraint group
    groups_mask = sum(
        1 << group for group in range(32) if group != restraint_force_group
    )
    state = simulation.context.getState(
        getEnergy=True, getPositions=True, getForces=True, groups=groups_mask
    )
    minimized_coords = np.array(state.getPositions().value_in_unit(_OMM_ANGS))
    energy = state.getPotentialEnergy().value_in_unit(_OMM_KCAL_PER_MOL)
    forces = np.array(
        state.getForces(asNumpy=True).value_in_unit(_OMM_KCAL_PER_MOL_ANGS)
    )

    return minimized_coords, energy, forces


def generate_torsion_minimised_dataset(
    mm_dataset: datasets.Dataset,
    ml_simulation: Simulation,
    mm_simulation: Simulation,
    torsion_restraint_force_constant: float = 1000.0,
    torsions_to_include_smarts: list[str] = DEFAULT_TORSIONS_TO_INCLUDE_SMARTS,
    torsions_to_exclude_smarts: list[str] = DEFAULT_TORSIONS_TO_EXCLUDE_SMARTS,
    ml_minimisation_steps: int = 10,
    mm_minimisation_steps: int = 10,
    ml_pdb_path: pathlib.Path | str | None = None,
    mm_pdb_path: pathlib.Path | str | None = None,
    map_ml_coords_energy_to_mm_coords_energy: bool = True,
    mm_min_energy_weight: float = 1000.0,
    mm_min_forces_weight: float = 0.1,
    ml_min_energy_weight: float = 1000.0,
    ml_min_forces_weight: float = 0.1,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Generate a dataset of torsion-restrained minimised structures.

    For each conformation in the input dataset:
    1. Restrain all rotatable torsions to their current values
    2. Perform a short MLP minimisation and save the energies
    3. From those coordinates, perform a short MM minimisation and save the coordinates
    4. Set forces to 0

    Parameters
    ----------
    mm_dataset : datasets.Dataset
        Input dataset with coordinates from MM MD sampling.
    ml_simulation : Simulation
        OpenMM simulation with ML potential.
    mm_simulation : Simulation
        OpenMM simulation with MM force field.
    torsion_restraint_force_constant : float, optional
        Force constant for torsion restraints in kJ/mol/rad^2.
    torsions_to_include_smarts : list[str] | None, optional
        List of SMARTS patterns to include for rotatable torsions. If None, include all rotatable torsions.
    torsions_to_exclude_smarts : list[str] | None, optional
        List of SMARTS patterns to exclude for rotatable torsions. If None, exclude no rotatable torsions.
    ml_minimisation_steps : int, optional
        Number of MLP minimisation steps (default: 10).
    mm_minimisation_steps : int, optional
        Number of MM minimisation steps (default: 10).
    ml_pdb_path : pathlib.Path | str | None, optional
        Path to save ML-minimised structures as a multi-model PDB file.
    mm_pdb_path : pathlib.Path | str | None, optional
        Path to save MM-minimised structures as a multi-model PDB file.
    map_ml_coords_energy_to_mm_coords_energy: bool = True,
        Whether to substitute the MLP energy for the MM-minimised coordinates with the
        MLP energy for the corresponding MLP-minimised coordinates.
    mm_min_energy_weight : float, optional
        Energy weight for MM-minimised dataset.
    mm_min_forces_weight : float, optional
        Forces weight for MM-minimised dataset.
    ml_min_energy_weight : float, optional
        Energy weight for ML-minimised dataset.
    ml_min_forces_weight : float, optional
        Forces weight for ML-minimised dataset.

    Returns:
    -------
    tuple[datasets.Dataset, datasets.Dataset]
        Tuple of (MM-minimised dataset, ML-minimised dataset).
    """
    # Extract molecule and find rotatable torsions
    mol = _get_molecule_from_dataset(mm_dataset)
    torsions_dict = get_rot_torsions_by_rot_bond(
        mol,
        include_smarts=torsions_to_include_smarts,
        exclude_smarts=torsions_to_exclude_smarts,
    )
    torsion_atoms_list = list(torsions_dict.values())

    if not torsion_atoms_list:
        logger.warning(
            "No rotatable torsions found - returning empty torsion minimised dataset"
        )
        # Return empty datasets with correct schema
        entry = mm_dataset[0]
        smiles = entry["smiles"]
        n_atoms = len(entry["coords"]) // (3 * len(entry["energy"]))
        empty_dataset = create_dataset_with_uniform_weights(
            smiles=smiles,
            coords=torch.empty(0, n_atoms, 3),
            energy=torch.empty(0),
            forces=torch.full((0, n_atoms, 3), 0.0),
            energy_weight=1.0,
            forces_weight=0.0,
        )
        return empty_dataset, empty_dataset

    # Extract coordinates from dataset
    assert len(mm_dataset) == 1, "Dataset should contain exactly one entry."
    entry = mm_dataset[0]
    n_snapshots = len(entry["energy"])
    n_atoms = len(entry["coords"]) // (3 * n_snapshots)
    coords = entry["coords"].reshape(n_snapshots, n_atoms, 3).numpy()
    smiles = entry["smiles"]

    # Add torsion restraint forces once to both simulations
    logger.debug(f"Adding {len(torsion_atoms_list)} torsion restraint forces")
    mm_force_indices, mm_restraint_group = _add_torsion_restraint_forces(
        mm_simulation, torsion_atoms_list, torsion_restraint_force_constant
    )
    ml_force_indices, ml_restraint_group = _add_torsion_restraint_forces(
        ml_simulation, torsion_atoms_list, torsion_restraint_force_constant
    )

    mm_minimised_coords = []
    mm_coords_ml_energies = []
    mm_coords_ml_forces = []

    ml_minimised_coords = []
    ml_coords_ml_energies = []
    ml_coords_ml_forces = []

    for i in track(
        range(n_snapshots),
        description="Generating torsion-minimised structures",
    ):
        # Step 1: Minimize with ML potential and frozen torsions
        ml_coords, ml_energy, ml_forces = _minimize_with_frozen_torsions(
            ml_simulation,
            coords[i],
            torsion_atoms_list,
            ml_force_indices,
            torsion_restraint_force_constant,
            ml_restraint_group,
            ml_minimisation_steps,
        )
        ml_coords_ml_energies.append(ml_energy)
        ml_coords_ml_forces.append(ml_forces)
        ml_minimised_coords.append(ml_coords)

        # Step 2: Minimize with MM potential and frozen torsions from ML coords
        mm_coords, _, _ = _minimize_with_frozen_torsions(
            mm_simulation,
            ml_coords,
            torsion_atoms_list,
            mm_force_indices,
            torsion_restraint_force_constant,
            mm_restraint_group,
            mm_minimisation_steps,
        )
        mm_minimised_coords.append(mm_coords)

        # Step 3: Now recalculate ML energy/forces at MM-minimised coords
        groups_mask = sum(
            1 << group for group in range(32) if group != ml_restraint_group
        )
        ml_simulation.context.setPositions(Quantity(mm_coords, angstrom))
        state = ml_simulation.context.getState(
            getEnergy=True, getForces=True, groups=groups_mask
        )
        ml_energy = state.getPotentialEnergy().value_in_unit(_OMM_KCAL_PER_MOL)
        ml_forces = state.getForces(asNumpy=True).value_in_unit(_OMM_KCAL_PER_MOL_ANGS)
        mm_coords_ml_energies.append(ml_energy)
        mm_coords_ml_forces.append(ml_forces)

    # Remove torsion restraint forces
    logger.debug("Removing torsion restraint forces")
    _remove_torsion_restraint_forces(mm_simulation, mm_force_indices)
    _remove_torsion_restraint_forces(ml_simulation, ml_force_indices)

    # Save ML-minimised structures to PDB if path is provided
    if ml_pdb_path is not None:
        logger.debug(f"Saving ML-minimised structures to {ml_pdb_path}")
        topology = ml_simulation.topology
        with open(ml_pdb_path, "w") as f:
            for i, ml_coords_frame in enumerate(ml_minimised_coords):
                positions = Quantity(ml_coords_frame, angstrom)
                PDBFile.writeModel(topology, positions, f, modelIndex=i)
            PDBFile.writeFooter(topology, f)

    # Save MM-minimised structures to PDB if path is provided
    if mm_pdb_path is not None:
        logger.debug(f"Saving MM-minimised structures to {mm_pdb_path}")
        topology = mm_simulation.topology
        with open(mm_pdb_path, "w") as f:
            for i, mm_coords_frame in enumerate(mm_minimised_coords):
                positions = Quantity(mm_coords_frame, angstrom)
                PDBFile.writeModel(topology, positions, f, modelIndex=i)
            PDBFile.writeFooter(topology, f)

    # Return two datasets: one for ML-minimised, one for MM-minimised
    mm_mapped_energies = (
        ml_coords_ml_energies
        if map_ml_coords_energy_to_mm_coords_energy
        else mm_coords_ml_energies
    )
    mm_min_dataset = create_dataset_with_uniform_weights(
        smiles=smiles,
        coords=torch.tensor(np.array(mm_minimised_coords)),
        energy=torch.tensor(np.array(mm_mapped_energies) - np.min(mm_mapped_energies)),
        forces=torch.tensor(np.array(mm_coords_ml_forces)),
        energy_weight=mm_min_energy_weight,
        forces_weight=mm_min_forces_weight,
    )

    ml_min_dataset = create_dataset_with_uniform_weights(
        smiles=smiles,
        coords=torch.tensor(np.array(ml_minimised_coords)),
        energy=torch.tensor(
            np.array(ml_coords_ml_energies) - np.min(ml_coords_ml_energies)
        ),
        forces=torch.tensor(np.array(ml_coords_ml_forces)),
        energy_weight=ml_min_energy_weight,
        forces_weight=ml_min_forces_weight,
    )

    return mm_min_dataset, ml_min_dataset


@_register_sampling_fn(settings.MMMDMetadynamicsTorsionMinimisationSamplingSettings)
def sample_mmmd_metadynamics_with_torsion_minimisation(
    mols: list[openff.toolkit.Molecule],
    off_ff: openff.toolkit.ForceField,
    device: torch.device,
    settings: settings.MMMDMetadynamicsTorsionMinimisationSamplingSettings,
    output_paths: dict[OutputType, pathlib.Path],
) -> list[datasets.Dataset]:
    """Generate datasets using metadynamics with additional torsion-minimised structures.

    This function extends sample_mmmd_metadynamics by:
    1. Running metadynamics to generate samples (same as sample_mmmd_metadynamics)
    2. For each sample, generating additional torsion-restrained minimised structures using both
       the ML and MM potentials.
    3. Returning all datasets with requested weights.

    Parameters
    ----------
    mols : list[openff.toolkit.Molecule]
        The molecules to sample.
    off_ff : openff.toolkit.ForceField
        The MM force field to use.
    device : torch.device
        The device to use for any MD or ML calculations.
    settings : MMMDMetadynamicsTorsionMinimisationSamplingSettings
        The sampling settings to use.
    output_paths: dict[OutputType, PathLike]
        A mapping of output types to filesystem paths.

    Returns:
    -------
    list[datasets.Dataset]
        The generated datasets with combined metadynamics and torsion-minimised samples.
    """
    if set(output_paths.keys()) != settings.output_types:
        raise ValueError(
            f"Output paths must contain exactly the keys {settings.output_types}"
        )

    all_datasets = []

    for mol_idx, mol in enumerate(mols, start=_MOL_INDEX_OFFSET):
        mol_with_conformers = _copy_mol_and_add_conformers(
            mol, settings.n_conformers, settings.starting_conformers
        )
        interchange = openff.interchange.Interchange.from_smirnoff(
            off_ff, openff.toolkit.Topology.from_molecules(mol_with_conformers)
        )

        torsions = get_rot_torsions_by_rot_bond(
            mol_with_conformers,
            include_smarts=settings.torsions_to_include_smarts,
            exclude_smarts=settings.torsions_to_exclude_smarts,
        )
        system = interchange.to_openmm_system()

        if not torsions:
            logger.warning(
                f"No rotatable bonds found in molecule {mol_idx}. "
                "Falling back to regular MD without torsion minimisation."
            )
            # Fall back to regular MD for this molecule
            simulation, integrator = _create_simulation(
                interchange.topology.to_openmm(),
                system,
                settings.temperature,
                settings.timestep,
                device,
            )

            pdb_path = None
            if OutputType.PDB_TRAJECTORY in output_paths:
                base_path = output_paths[OutputType.PDB_TRAJECTORY]
                pdb_path = str(get_mol_path(base_path, mol_idx))

            mm_dataset = _run_md(
                mol_with_conformers,
                simulation,
                simulation.step,
                settings.equilibration_n_steps_per_conformer,
                settings.production_n_snapshots_per_conformer,
                settings.production_n_steps_per_snapshot_per_conformer,
                pdb_path,
            )

            # Clean up MM simulation to free GPU memory
            cleanup_simulation(simulation, integrator)

            # Recalculate with ML potential
            ml_simulation, ml_integrator = _build_ml_simulation(
                mol_with_conformers,
                interchange.topology.to_openmm(),
                settings.mlp_settings,
                settings.temperature,
                settings.timestep,
                device,
            )
            ml_dataset = recalculate_energies_and_forces(mm_dataset, ml_simulation)

            # Clean up ML simulation to free GPU memory
            cleanup_simulation(ml_simulation, ml_integrator)

            # Convert to weighted dataset
            entry = ml_dataset[0]
            n_confs = len(entry["energy"])
            weighted_dataset = create_dataset_with_uniform_weights(
                smiles=entry["smiles"],
                coords=entry["coords"].reshape(n_confs, -1, 3),
                energy=entry["energy"],
                forces=entry["forces"].reshape(n_confs, -1, 3),
                energy_weight=settings.loss_energy_weight,
                forces_weight=settings.loss_force_weight,
            )
            all_datasets.append(weighted_dataset)
            continue

        # Setup metadynamics
        bias_variables = _get_torsion_bias_forces(
            mol_with_conformers,
            torsions_to_include=settings.torsions_to_include_smarts,
            torsions_to_exclude=settings.torsions_to_exclude_smarts,
            bias_width=settings.bias_width,
        )

        # Create molecule-specific bias directory
        base_bias_dir = output_paths[OutputType.METADYNAMICS_BIAS]
        bias_dir = get_mol_path(base_bias_dir, mol_idx)
        bias_dir.mkdir(parents=True, exist_ok=True)

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

        simulation, integrator = _create_simulation(
            interchange.topology.to_openmm(),
            system,
            settings.temperature,
            settings.timestep,
            device,
        )

        step_fn = functools.partial(metad.step, simulation)

        # Create molecule-specific PDB path
        pdb_path = None
        if OutputType.PDB_TRAJECTORY in output_paths:
            base_path = output_paths[OutputType.PDB_TRAJECTORY]
            pdb_path = str(get_mol_path(base_path, mol_idx))

        # Step 1: Generate MM metadynamics samples
        mm_dataset = _run_md(
            mol_with_conformers,
            simulation,
            step_fn,
            settings.equilibration_n_steps_per_conformer,
            settings.production_n_snapshots_per_conformer,
            settings.production_n_steps_per_snapshot_per_conformer,
            pdb_path,
        )

        # Clean up MM simulation to free GPU memory
        cleanup_simulation(simulation, integrator)

        # Create ML simulation for energy/force recalculation
        ml_simulation, ml_integrator = _build_ml_simulation(
            mol_with_conformers,
            interchange.topology.to_openmm(),
            settings.mlp_settings,
            settings.temperature,
            settings.timestep,
            device,
        )

        # Step 2: Recalculate energies and forces with ML potential
        ml_dataset = recalculate_energies_and_forces(mm_dataset, ml_simulation)

        # Clean up ML recalculation simulation to free GPU memory
        cleanup_simulation(ml_simulation, ml_integrator)

        # Convert to weighted dataset with MMMD weights
        entry = ml_dataset[0]
        n_confs = len(entry["energy"])
        mmmd_weighted_dataset = create_dataset_with_uniform_weights(
            smiles=entry["smiles"],
            coords=entry["coords"].reshape(n_confs, -1, 3),
            energy=entry["energy"],
            forces=entry["forces"].reshape(n_confs, -1, 3),
            energy_weight=settings.loss_energy_weight,
            forces_weight=settings.loss_force_weight,
        )

        # Step 3: Generate torsion-minimised structures
        # Create a fresh MM simulation for minimisation (without metadynamics biases)
        mm_min_simulation, mm_min_integrator = _build_mm_simulation(
            interchange, settings.temperature, settings.timestep, device
        )

        # Create a fresh ML simulation for minimisation
        ml_min_simulation, ml_min_integrator = _build_ml_simulation(
            mol_with_conformers,
            interchange.topology.to_openmm(),
            settings.mlp_settings,
            settings.temperature,
            settings.timestep,
            device,
        )

        # Create molecule-specific PDB paths for minimised structures
        ml_pdb_path = None
        mm_pdb_path = None
        if OutputType.ML_MINIMISED_PDB in output_paths:
            base_path = output_paths[OutputType.ML_MINIMISED_PDB]
            ml_pdb_path = get_mol_path(base_path, mol_idx)
        if OutputType.MM_MINIMISED_PDB in output_paths:
            base_path = output_paths[OutputType.MM_MINIMISED_PDB]
            mm_pdb_path = get_mol_path(base_path, mol_idx)

        torsion_mm_min_dataset, torsion_ml_min_dataset = (
            generate_torsion_minimised_dataset(
                mm_dataset,
                ml_min_simulation,
                mm_min_simulation,
                torsion_restraint_force_constant=settings.torsion_restraint_force_constant.value_in_unit(
                    _OMM_KJ_PER_MOL / _OMM_RADIAN**2
                ),
                torsions_to_include_smarts=settings.torsions_to_include_smarts,
                torsions_to_exclude_smarts=settings.torsions_to_exclude_smarts,
                ml_minimisation_steps=settings.ml_minimisation_steps,
                mm_minimisation_steps=settings.mm_minimisation_steps,
                ml_pdb_path=ml_pdb_path,
                mm_pdb_path=mm_pdb_path,
                map_ml_coords_energy_to_mm_coords_energy=settings.map_ml_coords_energy_to_mm_coords_energy,
                mm_min_energy_weight=settings.loss_energy_weight_mm_torsion_min,
                mm_min_forces_weight=settings.loss_force_weight_mm_torsion_min,
                ml_min_energy_weight=settings.loss_energy_weight_ml_torsion_min,
                ml_min_forces_weight=settings.loss_force_weight_ml_torsion_min,
            )
        )

        # Clean up minimisation simulations to free GPU memory
        cleanup_simulation(ml_min_simulation, ml_min_integrator)
        cleanup_simulation(mm_min_simulation, mm_min_integrator)

        # Merge all datasets
        combined_dataset = merge_weighted_datasets(
            [mmmd_weighted_dataset, torsion_ml_min_dataset, torsion_mm_min_dataset]
        )

        all_datasets.append(combined_dataset)

    return all_datasets


@_register_sampling_fn(settings.PreComputedDatasetSettings)
def load_precomputed_dataset(
    mols: list[openff.toolkit.Molecule],
    off_ff: openff.toolkit.ForceField,
    device: torch.device,
    settings: settings.PreComputedDatasetSettings,
    output_paths: dict[OutputType, pathlib.Path],
) -> list[datasets.Dataset]:
    """Load pre-computed dataset(s) from disk.

    For single-molecule fits, loads one dataset. For multi-molecule fits,
    loads one dataset per molecule in the order they appear in `mols`.

    Parameters
    ----------
    mols : list[openff.toolkit.Molecule]
        The molecules. The number of datasets loaded must match the number of molecules.
    off_ff : openff.toolkit.ForceField
        The force field (not used, kept for API consistency).
    device : torch.device
        The device to set the dataset format to.
    settings : PreComputedDatasetSettings
        Settings containing the path(s) to the pre-computed dataset(s).
    output_paths : dict[OutputType, pathlib.Path]
        Output paths (should be empty for this protocol).

    Returns:
    -------
    list[datasets.Dataset]
        The loaded datasets, one per molecule.

    Raises:
    ------
    ValueError
        If the number of dataset paths doesn't match the number of molecules.
    FileNotFoundError
        If any dataset path doesn't exist.
    """
    if set(output_paths.keys()) != settings.output_types:
        raise ValueError(
            f"Output paths must contain exactly the keys {settings.output_types}"
        )

    # Validate that the number of paths matches the number of molecules
    if len(settings.dataset_paths) != len(mols):
        raise ValueError(
            f"Number of dataset paths ({len(settings.dataset_paths)}) must match "
            f"number of molecules ({len(mols)})"
        )

    loaded_datasets = []

    for mol_idx, dataset_path in enumerate(settings.dataset_paths):
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path} (molecule {mol_idx})"
            )

        logger.info(
            f"Loading pre-computed dataset for molecule {mol_idx} from {dataset_path}"
        )
        loaded_dataset = datasets.load_from_disk(str(dataset_path))
        loaded_dataset.set_format("torch", device=device)
        loaded_datasets.append(loaded_dataset)

    return loaded_datasets
