"""
Functionality to obtain samples, to which the force field is fitted.
"""

import copy
import functools
import multiprocessing
from contextlib import redirect_stdout
from typing import Callable

import datasets
import datasets.table
import descent.targets.energy
import numpy
import openff.interchange
import openff.toolkit
import openmm
import torch
from openff.units import unit as off_unit
from openmm import LangevinMiddleIntegrator
from openmm.app import Simulation
from openmm.unit import Quantity, angstrom
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.ML.Cluster import Butina
from tqdm import tqdm

from . import mlp

_ANGSTROM = off_unit.angstrom

_OMM_KELVIN = openmm.unit.kelvin
_OMM_PS = openmm.unit.picosecond
_OMM_ANGS = openmm.unit.angstrom
_OMM_KCAL_PER_MOL = openmm.unit.kilocalorie_per_mole
_OMM_KCAL_PER_MOL_ANGS = openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom

GetDataFnType = Callable[
    [
        openff.toolkit.Molecule,
        openff.toolkit.ForceField,
        mlp.AvailableModels,
        float,
        float,
        int,
        int,
        int,
        int,
        float,
        float,
        float,
        int,
    ],
    datasets.Dataset,
]


def get_data_MMMD(
    mol: openff.toolkit.Molecule,
    off: openff.toolkit.ForceField,
    ML_path: mlp.AvailableModels = "mace-off23-small",
    temperature: float = 300,
    dt: float = 0.001,
    N: int = 1000,
    Nc: int = 1,
    MD_stepsize: int = 10,
    MD_startup: int = 100,
    MD_energy_upper_cutoff: float = 10.0,
    MD_energy_lower_cutoff: float = 1.0,
    cluster_tolerance: float = 0.075,
    cluster_Parallel: int = 1,
) -> datasets.Dataset:
    """generate a dataset from an openmm run using the input FF.
    Returns:
        A dataset full of MD snapshops with their energies and forces
    """
    # set up an openmm simulation
    molecule = copy.deepcopy(mol)
    molecule.generate_conformers(n_conformers=Nc, rms_cutoff=0.0 * _ANGSTROM)
    interchange = openff.interchange.Interchange.from_smirnoff(
        off, openff.toolkit.Topology.from_molecules(molecule)
    )
    integrator = LangevinMiddleIntegrator(
        temperature * _OMM_KELVIN, 1 / _OMM_PS, dt * _OMM_PS
    )
    simulation_ff = interchange.to_openmm_simulation(integrator)
    # minimize the system energy and take a snapshot for the ground-state reference
    coords, energy, forces, weight = [], [], [], []
    for conformer in tqdm(
        molecule.conformers,
        leave=False,
        colour="green",
        desc="Generating Snapshots",
    ):
        interchange.positions = conformer
        simulation_ff.context.setPositions(interchange.positions.to_openmm())
        simulation_ff.minimizeEnergy(maxIterations=100)
        coords.append(
            simulation_ff.context.getState(getPositions=True)
            .getPositions()
            .value_in_unit(_OMM_ANGS)
        )
        simulation_ff.step(MD_startup)
        # run the MD and take snapshots
        for _ in tqdm(range(N - 1), leave=False, colour="red", desc="Running MD"):
            simulation_ff.step(MD_stepsize)
            coords.append(
                simulation_ff.context.getState(getPositions=True)
                .getPositions()
                .value_in_unit(_OMM_ANGS)
            )
    coords_out = torch.tensor(coords)
    # Generate energy and force for the snapshots using a ML potential
    potential = mlp.get_mlp(ML_path)
    with open("/dev/null", "w") as f:
        with redirect_stdout(f):
            system = potential.createSystem(
                interchange.to_openmm_topology(),
                charge=mol.total_charge.m_as(off_unit.e),
            )
    integrator = copy.copy(integrator)
    simulation_ml = Simulation(interchange.topology, system, integrator)
    for i in tqdm(
        range(N * Nc),
        leave=False,
        colour="green",
        desc="Calculating Energies and Forces",
    ):
        my_pos = Quantity(numpy.array(coords_out[i]), angstrom)
        simulation_ml.context.setPositions(my_pos)
        state = simulation_ml.context.getState(getEnergy=True, getForces=True)
        energy.append(state.getPotentialEnergy().value_in_unit(_OMM_KCAL_PER_MOL))
        forces.append(
            state.getForces(asNumpy=True).value_in_unit(_OMM_KCAL_PER_MOL_ANGS)
        )
        weight.append(1.0)
        # delE = energy[i] - energy[0]
        # if delE < MD_energy_lower_cutoff:
        #     weight.append(1.0)
        # elif delE > MD_energy_upper_cutoff:
        #     weight.append(0.0)
        # else:
        #     weight.append(1.0 / math.sqrt(1.0 + (delE - 1.0) ** 2))
    smiles = mol.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)
    energy_0 = energy[0]
    energy_out = torch.tensor([x - energy_0 for x in energy])
    forces_out = torch.tensor(forces)

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


def get_data_MLMD(
    mol: openff.toolkit.Molecule,
    off: openff.toolkit.ForceField,
    ML_path: mlp.AvailableModels = "mace-off23-small",
    temperature: float = 300,
    dt: float = 0.001,
    N: int = 1000,
    Nc: int = 1,
    MD_stepsize: int = 10,
    MD_startup: int = 100,
    MD_energy_upper_cutoff: float = 10.0,
    MD_energy_lower_cutoff: float = 1.0,
    cluster_tolerance: float = 0.075,
    cluster_Parallel: int = 1,
) -> datasets.Dataset:
    """generate a dataset from an openmm run using the input ML Potential.
    Returns:
        A dataset full of MD snapshops with their energies and forces
    """
    # set up an openmm simulation
    molecule = copy.deepcopy(mol)
    molecule.generate_conformers(n_conformers=Nc, rms_cutoff=0.0 * _ANGSTROM)
    force_field = copy.deepcopy(off)
    interchange = openff.interchange.Interchange.from_smirnoff(
        force_field, openff.toolkit.Topology.from_molecules(molecule)
    )
    integrator = LangevinMiddleIntegrator(
        temperature * _OMM_KELVIN, 1 / _OMM_PS, dt * _OMM_PS
    )
    potential = mlp.get_mlp(ML_path)
    with open("/dev/null", "w") as f:
        with redirect_stdout(f):
            system = potential.createSystem(
                interchange.to_openmm_topology(),
                charge=mol.total_charge.m_as(off_unit.e),
            )
    simulation = Simulation(interchange.topology, system, integrator)

    coords, energy, forces, weight = [], [], [], []
    for conformer in tqdm(
        list(molecule.conformers),
        leave=False,
        colour="green",
        desc="Generating Dataset",
    ):
        interchange.positions = conformer
        position = interchange.positions.to_openmm()
        simulation.context.setPositions(position)
        simulation.minimizeEnergy(maxIterations=100)
        state = simulation.context.getState(
            getPositions=True, getEnergy=True, getForces=True
        )
        coords.append(state.getPositions().value_in_unit(_OMM_ANGS))
        energy.append(state.getPotentialEnergy().value_in_unit(_OMM_KCAL_PER_MOL))
        forces.append(
            state.getForces(asNumpy=True).value_in_unit(_OMM_KCAL_PER_MOL_ANGS)
        )
        weight.append(1.0)
        simulation.step(MD_startup)
        for _ in tqdm(range(N - 1), leave=False, colour="red"):
            simulation.step(MD_stepsize)
            state = simulation.context.getState(
                getPositions=True, getEnergy=True, getForces=True
            )
            coords.append(state.getPositions().value_in_unit(_OMM_ANGS))
            energy.append(state.getPotentialEnergy().value_in_unit(_OMM_KCAL_PER_MOL))
            forces.append(
                state.getForces(asNumpy=True).value_in_unit(_OMM_KCAL_PER_MOL_ANGS)
            )
            weight.append(1.0)
            # delE = energy[i] - energy[0]
            # if delE < MD_energy_lower_cutoff:
            #     weight.append(1.0)
            # elif delE > MD_energy_upper_cutoff:
            #     weight.append(0.0)
            # else:
            #     weight.append(1.0 / math.sqrt(1.0 + (delE - 1.0) ** 2))
    energy_0 = energy[0]
    energy_out = torch.tensor([x - energy_0 for x in energy])
    forces_out = torch.tensor(forces)
    coords_out = torch.tensor(coords)
    smiles = mol.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)
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


def get_data_cMMMD(
    mol: openff.toolkit.Molecule,
    off: openff.toolkit.ForceField,
    ML_path: mlp.AvailableModels = "mace-off23-small",
    temperature: float = 300,
    dt: float = 0.001,
    N: int = 1000,
    Nc: int = 1,
    MD_stepsize: int = 10,
    MD_startup: int = 100,
    MD_energy_upper_cutoff: float = 10.0,
    MD_energy_lower_cutoff: float = 1.0,
    cluster_tolerance: float = 0.075,
    cluster_Parallel: int = 1,
) -> datasets.Dataset:
    """generate a dataset from an openmm run using the input FF
    and cluster the data by rmsd and include counts in the weights
    Returns:
        A dataset full of MD snapshops with their energies and forces
    """
    # set up an openmm simulation
    molecule = copy.deepcopy(mol)
    molecule.generate_conformers(n_conformers=Nc, rms_cutoff=0.0 * _ANGSTROM)
    interchange = openff.interchange.Interchange.from_smirnoff(
        off, openff.toolkit.Topology.from_molecules(molecule)
    )
    integrator = LangevinMiddleIntegrator(
        temperature * _OMM_KELVIN, 1 / _OMM_PS, dt * _OMM_PS
    )
    simulation_ff = interchange.to_openmm_simulation(integrator)
    # minimize the system energy and take a snapshot for the ground-state reference
    coords: list[float] = []
    energy: list[float] = []
    forces: list[float] = []
    weight: list[float] = []
    for conformer in tqdm(
        molecule.conformers,
        leave=False,
        colour="green",
        desc="Generating Snapshots",
    ):
        interchange.positions = conformer
        simulation_ff.context.setPositions(interchange.positions.to_openmm())
        simulation_ff.minimizeEnergy(maxIterations=100)
        coords.append(
            simulation_ff.context.getState(getPositions=True)
            .getPositions()
            .value_in_unit(_OMM_ANGS)
        )
        simulation_ff.step(MD_startup)
        # run the MD and take snapshots
        for _ in tqdm(range(N - 1), leave=False, colour="red", desc="Running MD"):
            simulation_ff.step(MD_stepsize)
            coords.append(
                simulation_ff.context.getState(getPositions=True)
                .getPositions()
                .value_in_unit(_OMM_ANGS)
            )
    coords_out = torch.tensor(coords)
    # Cluster the coordinates based on rmsd
    coords_clstr = coords_out.reshape(N * Nc, -1, 3).tolist()
    coords_clstr = [c * _ANGSTROM for c in coords_clstr]
    mol_clstr = copy.deepcopy(mol)
    mol_clstr._conformers = coords_clstr
    mol_rdkit: Chem.Mol = Chem.RemoveHs(mol_clstr.to_rdkit())
    conf_ids = [conf.GetId() for conf in mol_rdkit.GetConformers()]
    conf_pairs = [(i, j) for i in range(len(conf_ids)) for j in range(i)]
    conf_pairs_np = numpy.array_split(numpy.array(conf_pairs), cluster_Parallel)
    rms_fn = functools.partial(compute_best_rms, mol=mol_rdkit)
    with multiprocessing.Pool(cluster_Parallel) as pool:
        dists = list(
            tqdm(
                pool.imap(rms_fn, conf_pairs_np),
                total=len(conf_pairs_np),
                leave=False,
                colour="green",
                desc="Clustering the Conformers",
            )
        )
    dists_flat = [d for dist in dists for d in dist]
    clusters = Butina.ClusterData(  # type: ignore[no-untyped-call]
        dists_flat, len(conf_ids), cluster_tolerance, isDistData=True, reordering=True
    )
    cluster_ids = [cluster[0] for cluster in clusters]
    cluster_len = [len(cluster) for cluster in clusters]
    tqdm.write(f"Clustering Summary: {len(conf_ids)} -> {len(cluster_ids)}")
    coords_use = coords_out[cluster_ids, :, :]
    # Generate energy and force for the snapshots using a ML potential
    potential = mlp.get_mlp(ML_path)
    with open("/dev/null", "w") as f:
        with redirect_stdout(f):
            system = potential.createSystem(
                interchange.to_openmm_topology(),
                charge=mol.total_charge.m_as(off_unit.e),
            )
    integrator = copy.copy(integrator)
    simulation_ml = Simulation(interchange.topology, system, integrator)
    for i in tqdm(
        range(len(cluster_ids)),
        leave=False,
        colour="green",
        desc="Calculating Energies and Forces",
    ):
        my_pos = Quantity(numpy.array(coords_use[i]), angstrom)
        simulation_ml.context.setPositions(my_pos)
        state = simulation_ml.context.getState(getEnergy=True, getForces=True)
        energy.append(state.getPotentialEnergy().value_in_unit(_OMM_KCAL_PER_MOL))
        forces.append(
            state.getForces(asNumpy=True).value_in_unit(_OMM_KCAL_PER_MOL_ANGS)
        )
        weight.append(cluster_len[i])
        # delE = energy[i] - energy[0]
        # if delE < MD_energy_lower_cutoff:
        #     weight.append(cluster_len[i])
        # elif delE > MD_energy_upper_cutoff:
        #     weight.append(0.0)
        # else:
        #     weight.append(1.0 / math.sqrt(1.0 + (delE - 1.0) ** 2))
    energy_0 = energy[0]
    energy_out = torch.tensor([x - energy_0 for x in energy])
    forces_out = torch.tensor(forces)
    smiles = mol.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)
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


def compute_best_rms(pairs: list[tuple[int, int]], mol: Chem.Mol) -> list[float]:
    atom_map = [(i, i) for i in range(mol.GetNumAtoms())]
    return [
        rdMolAlign.AlignMol(
            Chem.Mol(mol), Chem.Mol(mol), int(i), int(j), atomMap=atom_map
        )
        for i, j in pairs
    ]
