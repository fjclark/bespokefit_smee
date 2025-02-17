"""
DATA_MAKER:

Dataset generation functions for run-fit
"""
###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################
from contextlib import redirect_stdout
import smee
from tqdm import tqdm
import multiprocessing
import torch
import numpy
import math
import datasets
import datasets.table
import pyarrow
import typing
import copy
import openff.interchange
import openff.toolkit
from openmm.app import *
from openmm import *
from openmm.unit import *
from openmmml import MLPotential
from openff.units import unit as off_unit
import descent.targets.energy
from parameterizer import convert_to_smirnoff
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.ML.Cluster import Butina
###############################################################################
############################### FUNCTIONS #####################################
###############################################################################
N_WORKERS = 32
DATA_SCHEMA = pyarrow.schema(
    [
        ("coords", pyarrow.list_(pyarrow.float64())),
        ("energy", pyarrow.list_(pyarrow.float64())),
        ("forces", pyarrow.list_(pyarrow.float64())),
        ("weight", pyarrow.list_(pyarrow.float64())),
    ]
)
class Entry(typing.TypedDict):
    """Contains:
    - The coordinates [Å] of the conformers
    - The reference energies [kcal/mol] with ``shape=(n_confs,)``
    - The reference forces [kcal/mol/Å] with ``shape=(n_confs, n_particles, 3)``
    - The reference loss weights with ``shape=(n_confs,)``
    """
    coords: torch.Tensor
    energy: torch.Tensor
    forces: torch.Tensor
    weight: torch.Tensor

def create_dataset(entries: list[Entry]) -> datasets.Dataset:
    """Create a dataset from a list of existing entries.
    Args:
        entries: The entries to create the dataset from.
    Returns:
        The created dataset.
    """
    table = pyarrow.Table.from_pylist(
        [
            {
                "coords": torch.tensor(entry["coords"]).flatten().tolist(),
                "energy": torch.tensor(entry["energy"]).flatten().tolist(),
                "forces": torch.tensor(entry["forces"]).flatten().tolist(),
                "weight": torch.tensor(entry["weight"]).flatten().tolist(),
            }
            for entry in entries
        ],
        schema=DATA_SCHEMA,
    )
    dataset = datasets.Dataset(datasets.table.InMemoryTable(table))
    dataset.set_format("torch")
    return dataset

def get_data_MMMD(
    mol: openff.toolkit.Molecule,
    off: openff.toolkit.ForceField,
    ML_path: str = "mace-off23-small",
    temperature: float = 300,
    dt: float = 0.001,
    N: int = 1000,
    Nc: int = 1,
    Ng: int = 10,
    Ni: int = 100
) -> datasets.Dataset:
    """generate a dataset from an openmm run using the input FF.

    Args:
        mol: molecule object
        force_field: The force field to use
        ML_path: ML potential used to calculate energies and forces
        temperature: Temperature of the MD run
        dt: Time step in the MD run
        N: Number of samples to take
        Nc: Number of conformers to generate at start
        Ng: Number of steps to take between snapshops
        Ni: Number of steps to take before begining to take snapshots

    Returns:
        A dataset full of MD snapshops with their energies and forces
    """
# set up an openmm simulation
    molecule        = copy.deepcopy(mol)
    molecule.generate_conformers(n_conformers=Nc, rms_cutoff=0.0 * off_unit.angstrom)
    off_topology    = openff.toolkit.Topology.from_molecules(molecule)
    interchange     = openff.interchange.Interchange.from_smirnoff(off,off_topology)
    integrator      = LangevinMiddleIntegrator(temperature*kelvin,1/picosecond,dt*picoseconds)
    simulation_ff   = interchange.to_openmm_simulation(integrator)
# minimize the system energy and take a snapshot for the ground-state reference
    coords, energy, forces, weight = [], [], [], []
    for i, conformer in tqdm(list(enumerate(molecule.conformers)),leave=False,colour='green',desc="Generating Snapshots"):
        interchange.positions = conformer
        position              = interchange.positions.to_openmm()
        simulation_ff.context.setPositions(interchange.positions.to_openmm())
        simulation_ff.minimizeEnergy(maxIterations=100)
        state                 = simulation_ff.context.getState(getPositions=True)
        coords.append(state.getPositions().value_in_unit(openmm.unit.angstrom))
        simulation_ff.step(Ni)
# run the MD and take snapshots
        for _ in tqdm(range(N-1),leave=False,colour='red',desc="Running MD"):
            simulation_ff.step(Ng)
            state  = simulation_ff.context.getState(getPositions=True)
            coords.append(state.getPositions().value_in_unit(openmm.unit.angstrom))
    coords_out    = torch.tensor(coords) 
# Generate energy and force for the snapshots using a ML potential
    potential     = MLPotential(*ML_path)
    with open("/dev/null", 'w') as f:
        with redirect_stdout(f):
            system = potential.createSystem(interchange.to_openmm_topology())   
    integrator     = copy.copy(integrator)
    simulation_ml  = Simulation(interchange.topology, system, integrator)        
    for i in tqdm(range(N * Nc),leave=False,colour='green',desc="Calculating Energies and Forces"):
        my_pos =  Quantity(numpy.array(coords_out[i]), angstrom)
        simulation_ml.context.setPositions(my_pos)
        state  = simulation_ml.context.getState(getEnergy=True, getForces=True)
        energy.append(state.getPotentialEnergy().value_in_unit(openmm.unit.kilocalorie_per_mole))
        forces.append(state.getForces(asNumpy=True).value_in_unit(openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom))
        delE = energy[i] - energy[0]
        if delE < 1.0:
            weight.append(1.0)
        elif delE > 30.0:
            weight.append(0.0)
        else:
            weight.append(1.0 / math.sqrt(1.0 + (delE - 1.0) ** 2))       
    energy_0   = energy[0]
    energy_out = torch.tensor([x - energy_0 for x in energy])
    forces_out = torch.tensor(forces)
    weight_out = torch.tensor(weight)

    return create_dataset( [{"coords": coords_out, "energy": energy_out, "forces": forces_out, "weight": weight_out}]) 

def get_data_MLMD(
    mol: openff.toolkit.Molecule,
    off: openff.toolkit.ForceField,
    ML_path: str = "mace-off23-small",
    temperature: float = 300,
    dt: float = 0.001,
    N: int = 1000,
    Nc: int = 1,
    Ng: int = 10,
    Ni: int = 100
) -> datasets.Dataset:
    """generate a dataset from an openmm run using the input ML Potential.

    Args:
        mol: molecule object
        force_field: Forcefield to setup up the md. This is ignored in the actual calculation
        ML_path: ML potential used
        temperature: Temperature of the MD run
        dt: Time step in the MD run
        N: Number of samples to take
        Nc: Number of conformers to generate at start
        Ng: Number of steps to take between snapshops
        Ni: Number of steps to take before begining to take snapshots

    Returns:
        A dataset full of MD snapshops with their energies and forces
    """
# set up an openmm simulation
    molecule        = copy.deepcopy(mol)
    molecule.generate_conformers(n_conformers=Nc, rms_cutoff=0.0 * off_unit.angstrom)
    off_topology    = openff.toolkit.Topology.from_molecules(molecule)
    force_field     = copy.deepcopy(off)
    interchange     = openff.interchange.Interchange.from_smirnoff(force_field,off_topology)
    integrator      = LangevinMiddleIntegrator(temperature*kelvin,1/picosecond,dt*picoseconds)
    potential       = MLPotential(*ML_path)
    with open("/dev/null", 'w') as f:
        with redirect_stdout(f):
            system  = potential.createSystem(interchange.to_openmm_topology())
    simulation      = Simulation(interchange.topology, system, integrator)

    coords, energy, forces, weight = [], [], [], []
    for i, conformer in tqdm(list(enumerate(molecule.conformers)),leave=False,colour='green',desc="Generating Dataset"):
        interchange.positions = conformer
        position              = interchange.positions.to_openmm()
        simulation.context.setPositions(position)
        simulation.minimizeEnergy(maxIterations=100)
        state  = simulation.context.getState(getPositions=True, getEnergy=True, getForces=True)
        coords.append(state.getPositions().value_in_unit(openmm.unit.angstrom))
        energy.append(state.getPotentialEnergy().value_in_unit(openmm.unit.kilocalorie_per_mole))
        forces.append(state.getForces(asNumpy=True).value_in_unit(openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom))
        weight.append(1.0)
        simulation.step(Ni)
        for _ in tqdm(range(N-1),leave=False,colour='red'):
            simulation.step(Ng)
            state = simulation.context.getState(getPositions=True, getEnergy=True, getForces=True)
            coords.append(state.getPositions().value_in_unit(openmm.unit.angstrom))
            energy.append(state.getPotentialEnergy().value_in_unit(openmm.unit.kilocalorie_per_mole))
            forces.append(state.getForces(asNumpy=True).value_in_unit(openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom))
            delE = energy[i] - energy[0]
            if delE < 1.0:
                weight.append(1.0)
            elif delE > 10.0:
                weight.append(0.0)
            else:
                weight.append(1.0 / math.sqrt(1.0 + (delE - 1.0) ** 2))        
    energy_0   = energy[0]
    energy_out = torch.tensor([x - energy_0 for x in energy])
    forces_out = torch.tensor(forces)
    coords_out = torch.tensor(coords) 
    weight_out = torch.tensor(weight)
    return create_dataset( [{"coords": coords_out, "energy": energy_out, "forces": forces_out, "weight": weight_out}]) 

def get_data_cMMMD(
    mol: openff.toolkit.Molecule,
    off: openff.toolkit.ForceField,
    ML_path: str = "mace-off23-small",
    temperature: float = 300,
    dt: float = 0.001,
    N: int = 1000,
    Nc: int = 1,
    Ng: int = 10,
    Ni: int = 100
) -> datasets.Dataset:
    """generate a dataset from an openmm run using the input FF.

    Args:
        mol: molecule object
        force_field: The force field to use
        ML_path: ML potential used to calculate energies and forces
        temperature: Temperature of the MD run
        dt: Time step in the MD run
        N: Number of samples to take
        Nc: Number of conformers to generate at start
        Ng: Number of steps to take between snapshops
        Ni: Number of steps to take before begining to take snapshots

    Returns:
        A dataset full of MD snapshops with their energies and forces
    """
# set up an openmm simulation
    molecule        = copy.deepcopy(mol)
    molecule.generate_conformers(n_conformers=Nc, rms_cutoff=0.0 * off_unit.angstrom)
    off_topology    = openff.toolkit.Topology.from_molecules(molecule)
    interchange     = openff.interchange.Interchange.from_smirnoff(off,off_topology)
    integrator      = LangevinMiddleIntegrator(temperature*kelvin,1/picosecond,dt*picoseconds)
    simulation_ff   = interchange.to_openmm_simulation(integrator)
# minimize the system energy and take a snapshot for the ground-state reference
    coords, energy, forces, weight = [], [], [], []
    for i, conformer in tqdm(list(enumerate(molecule.conformers)),leave=False,colour='green',desc="Generating Snapshots"):
        interchange.positions = conformer
        position              = interchange.positions.to_openmm()
        simulation_ff.context.setPositions(interchange.positions.to_openmm())
        simulation_ff.minimizeEnergy(maxIterations=100)
        state                 = simulation_ff.context.getState(getPositions=True)
        coords.append(state.getPositions().value_in_unit(openmm.unit.angstrom))
        simulation_ff.step(Ni)
# run the MD and take snapshots
        for _ in tqdm(range(N-1),leave=False,colour='red',desc="Running MD"):
            simulation_ff.step(Ng)
            state  = simulation_ff.context.getState(getPositions=True)
            coords.append(state.getPositions().value_in_unit(openmm.unit.angstrom))
    coords_out = torch.tensor(coords) 
# Cluster the coordinates based on rmsd
    coords_clstr = coords_out.reshape(N * Nc, -1, 3).tolist()
    coords_clstr = [c * off_unit.angstrom for c in coords_clstr]
    mol_clstr    = copy.deepcopy(mol)
    mol_clstr._conformers = coords_clstr
    mol_rdkit: Chem.Mol   = Chem.RemoveHs(mol_clstr.to_rdkit())
    conf_ids     = [conf.GetId() for conf in mol_rdkit.GetConformers()]
    conf_pairs   = [(i, j) for i in range(len(conf_ids)) for j in range(i)]
    conf_pairs   = numpy.array_split(numpy.array(conf_pairs), N_WORKERS)
    rms_fn       = functools.partial(compute_best_rms, mol=mol_rdkit)
    with multiprocessing.Pool(N_WORKERS) as pool:
        dists   = list(tqdm(pool.imap(rms_fn, conf_pairs), total=len(conf_pairs),leave=False,colour='green',desc="Clustering the Conformers"))
    dists       = [d for dist in dists for d in dist]
    clusters    = Butina.ClusterData(dists, len(conf_ids), 0.075, isDistData=True, reordering=True)
#    clusters    = Butina.ClusterData(dists, len(conf_ids), 0.1, isDistData=True, reordering=True)
    cluster_ids = [cluster[0] for cluster in clusters]
    cluster_len = [len(cluster) for cluster in clusters]
    tqdm.write(f"{len(conf_ids)} -> {len(cluster_ids)}")
    coords_use = coords_out[cluster_ids, :, :]
# Generate energy and force for the snapshots using a ML potential
    potential     = MLPotential(*ML_path)
    with open("/dev/null", 'w') as f:
        with redirect_stdout(f):
            system = potential.createSystem(interchange.to_openmm_topology())   
    integrator     = copy.copy(integrator)
    simulation_ml  = Simulation(interchange.topology, system, integrator)        
    for i in tqdm(range(len(cluster_ids)),leave=False,colour='green',desc="Calculating Energies and Forces"):
        my_pos =  Quantity(numpy.array(coords_use[i]), angstrom)
        simulation_ml.context.setPositions(my_pos)
        state  = simulation_ml.context.getState(getEnergy=True, getForces=True)
        energy.append(state.getPotentialEnergy().value_in_unit(openmm.unit.kilocalorie_per_mole))
        forces.append(state.getForces(asNumpy=True).value_in_unit(openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom))
        delE = energy[i] - energy[0]
        if delE < 1.0:
            weight.append(cluster_len[i])
        elif delE > 10.0:
            weight.append(0.0)
        else:
            weight.append(1.0 / math.sqrt(1.0 + (delE - 1.0) ** 2))      
    energy_0   = energy[0]
    energy_out = torch.tensor([x - energy_0 for x in energy])
    forces_out = torch.tensor(forces)
    weight_out = torch.tensor(weight)
    return create_dataset( [{"coords": coords_use, "energy": energy_out, "forces": forces_out, "weight": weight_out}]) 

def compute_best_rms(pairs: list[tuple[int, int]], mol: Chem.Mol) -> list[float]:
    # return rdMolAlign.GetBestRMS(Chem.Mol(mol), Chem.Mol(mol), *pair)
    atom_map = [(i, i) for i in range(mol.GetNumAtoms())]

    return [
        rdMolAlign.AlignMol(
            Chem.Mol(mol), Chem.Mol(mol), int(i), int(j), atomMap=atom_map
        )
        for i, j in pairs
    ]
