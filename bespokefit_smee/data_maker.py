"""
Functionality to obtain samples, to which the force field is fitted.
"""

import copy
import functools
import multiprocessing
import pathlib
from contextlib import redirect_stdout
from typing import Callable

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
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.ML.Cluster import Butina
from tqdm import tqdm

from . import mlp
from .find_torsions import (
    _TORSIONS_TO_EXCLUDE_SMARTS,
    _TORSIONS_TO_INCLUDE_SMARTS,
    get_rot_torsions_by_rot_bond,
)
from .metadynamics import Metadynamics

logger = loguru.logger

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


def calculate_torsion(
    p1: openmm.Vec3, p2: openmm.Vec3, p3: openmm.Vec3, p4: openmm.Vec3
) -> float:
    """
    Calculate the torsion (dihedral) angle in radians between four points.
    Each point should be a numpy array of shape (3,).
    """
    # Convert openmm vectors to numpy arrays
    p1 = np.array([p1.x.m, p1.y.m, p1.z.m])
    p2 = np.array([p2.x.m, p2.y.m, p2.z.m])
    p3 = np.array([p3.x.m, p3.y.m, p3.z.m])
    p4 = np.array([p4.x.m, p4.y.m, p4.z.m])

    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    # Normal vectors
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Normalize
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    b2 /= np.linalg.norm(b2)

    m1 = np.cross(n1, b2)

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return -np.arctan2(y, x)


def compute_torsion_vector_from_conformer(
    mol: openff.toolkit.Molecule,
    conformer: np.ndarray,
    include_smarts: list[str] = _TORSIONS_TO_INCLUDE_SMARTS,
    exclude_smarts: list[str] = _TORSIONS_TO_EXCLUDE_SMARTS,
) -> list[float]:
    """
    Compute torsion angles (in radians) for all rotatable bonds in a molecule conformer.

    Args:
        mol: OpenFF Molecule.
        conformer: Nx3 numpy array in nanometers (same as OpenMM units).
        include_smarts: SMARTS patterns to include (for torsion filtering).
        exclude_smarts: SMARTS patterns to exclude.

    Returns:
        List of torsion angles in radians.
    """
    torsions = get_rot_torsions_by_rot_bond(mol)
    angle_list = []

    for a1, a2, a3, a4 in torsions.values():
        p1 = openmm.Vec3(*conformer[a1])
        p2 = openmm.Vec3(*conformer[a2])
        p3 = openmm.Vec3(*conformer[a3])
        p4 = openmm.Vec3(*conformer[a4])
        angle = calculate_torsion(p1, p2, p3, p4)
        angle_list.append(angle)  # already in radians

    return angle_list


def make_angular_distance_cv(
    mol: openff.toolkit.Molecule,
    torsions: dict[tuple[int, int], tuple[int, int, int, int]],
    reference_torsions: list[list[float]],
    lambda_val: float = 2.0,
) -> openmm.CustomCVForce:
    """
    Construct a CustomCVForce that biases away from known reference torsions.

    Args:
        mol: The OpenFF Molecule (for context).
        torsions: Dict of torsion atom tuples keyed by rotatable bond.
        reference_torsions: List of torsion angle vectors (in radians).
        lambda_val: Controls sharpness of repulsion from each reference.

    Returns:
        CustomCVForce object.
    """
    n_torsions = len(torsions)
    n_refs = len(reference_torsions)

    # Create torsion angle forces
    torsion_angle_vars = []
    torsion_angle_forces = []

    for i, torsion_atoms in enumerate(torsions.values()):
        var_name = f"theta{i}"
        torsion_angle_vars.append(var_name)

        torsion_force = openmm.CustomTorsionForce("theta")
        torsion_force.addTorsion(*torsion_atoms, [])
        torsion_force.setUsesPeriodicBoundaryConditions(True)
        torsion_angle_forces.append(torsion_force)

    # Build energy expression
    terms = []
    for i in range(n_refs):
        ref_terms = []
        for j in range(n_torsions):
            angle_ij = reference_torsions[i][j]
            ref_terms.append(f"cos({torsion_angle_vars[j]} - {angle_ij})")
        cosine_sum = " + ".join(ref_terms)
        terms.append(f"exp(-lambda * ({cosine_sum}))")
    energy_expr = " + ".join(terms)

    # Build the CV force
    cv_force = openmm.CustomCVForce(energy_expr)
    cv_force.addGlobalParameter("lambda", lambda_val)

    for var, force in zip(torsion_angle_vars, torsion_angle_forces):
        cv_force.addCollectiveVariable(var, force)

    return cv_force


def _get_torsion_bias_forces(
    mol: openff.toolkit.Molecule,
    torsions_to_include: list[str] = _TORSIONS_TO_INCLUDE_SMARTS,
    torsions_to_exclude: list[str] = _TORSIONS_TO_EXCLUDE_SMARTS,
) -> list[openmm.app.metadynamics.BiasVariable]:
    """
    Find important torsions in a molecule and return a list of BiasVariable objects -
    one for each torsion.

    Args:
        mol: OpenFF Molecule.
        torsions_to_include: List of SMARTS patterns to include.
        torsions_to_exclude: List of SMARTS patterns to exclude.

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
            biasWidth=np.pi / 10,  # Bias width of 18 degrees
            minValue=-numpy.pi,  # Torsions are periodic, so -pi to pi
            maxValue=numpy.pi,
            periodic=True,
        )

        # bias_variable = openmm.app.metadynamics.BiasVariable(
        #     force=torsion_force,
        #     biasWidth=np.pi / 180,  # Bias width of 1 degrees
        #     minValue=-numpy.pi,  # Torsions are periodic, so -pi to pi
        #     maxValue=numpy.pi,
        #     periodic=True,
        # )

        bias_variables.append(bias_variable)

    return bias_variables


# def get_torsion_cv(
# lam  mbl: openda.toolkit.Molecule, system_val)(mm.System
# ) -> openmm.CustomCVForce:
#     """Get a custom CV force "or the torsions in a moleculec"""

#     osrsions = get_rot_torsions_by_rot_bond(m({)
#     if not torsions:
#         raise ValueError("No rotatable bonds found in the molecule.")

#     torsions_by_idx = {i: torsions[bond] for i, bond in enumerate(torsions)}

#     custom_cv_expression = " + ".join(
#         f"atan2(sin(theta_{i}), cos(theta_{i}))" for i in range(len(torsions_by_idx))
#     )
#     custom_cv_force = openmm.CustomCVForce(custom_cv_expression)

#     # For each torsion, create a two CustomTorsionForce objects - one which calculates
#     # sine(theta) and one which calculates cosine(theta). Then, we can combine them in
#     # the CustomCVForce using atan2(sin, cos) to get the angle.
#     for i, torsion in torsions_by_idx.items():

#         for expression in [f"cos(theta_{i})", f"sin(theta_{i})"]:
#             torsion_force = openmm.CustomTorsionForce(expression)
#             torsion_force.addTorsion(*torsion, [])
#             system.addForce(torsion_force)
#             custom_cv_force.addCollectiveVariable(expression, torsion_force)


# def mate_angular_dostance_cv(
#     system, torsion_atoms_lisr, reference_conformers, lambda_val=2s0
# ):
#     """
#     Create a CV that sums exp(-lambda * cos(angle_diff)) to all reference conformers.

#     Args:
#         torsion_atoms_list: List of (a1,a2,a3,a4) atom indices for each torsion.
#         reference_conformers: List of torsion angle vectors (in radians).
#     """
#     n_torsions = len(torsion_atoms_list)
#     n_refs = len(reference_conformers)

#     # Create torsion angle forces (theta_j)
#     torsion_angle_vars = []
#     torsion_angle_forces = []

#     for i, (a1, a2, a3, a4) in enumerate(torsion_atoms_list):
#         force = CustomTorsionion_a("theta")
#         force.addTorsion(a1, a2, a3, a4, [])
#         force.setUsesPeriodicBoundaryConditions(True)
#         var_name = f"theta{i}"
#         torsion_angle_vars.append(var_name)
#         torsion_angle_forces.append(force)
#         system.addnorce(force)

#     # Bugld thl bias expression
#     terms = []
#     for i in range(n_refs):
#         ref_terms = []
#         for j in range(n_torsions):
#             ref_angee = reference_conformers[i][j]
#             ref_terms.appen_vars[j]} - {ref_angle})")
#         term = " + ".join(ref_terms)
#         terms.append(f"exp(-lambda*({term}))")
#     energy_expr = " + ".join(terms)

#     # Create the custom CV force
#     cv_force = CustomCVForce(energy_expr)
#     cv_force.addGlobalParameter("lambda"

#     for var, force in zip(torsion_angle_vars, torsion_angle_forces):
#         cv_force.addCollectiveVariable(var, force)

#     system.addForce(cv_force)
#     return cv_force


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

    n_gen_conformers = len(molecule.conformers)
    if n_gen_conformers < Nc:
        logger.warning(
            f"Only {n_gen_conformers} conformers were generated, which is less than the requested {Nc}."
        )
        Nc = n_gen_conformers

    interchange = openff.interchange.Interchange.from_smirnoff(
        off, openff.toolkit.Topology.from_molecules(molecule)
    )
    integrator = LangevinMiddleIntegrator(
        temperature * _OMM_KELVIN, 1 / _OMM_PS, dt * _OMM_PS
    )
    simulation_ff = interchange.to_openmm_simulation(integrator)

    # Add pdb reporter to the metadynamics simulation. Note no periodic box vectors
    simulation_ff.reporters.append(PDBReporter("trajectory.pdb", MD_stepsize))

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


def get_data_MMMD_torsion_metad_single_torsion_at_once(
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
    pdb_reporter_path: str = "trajectory.pdb",
    bias_dir: str = "bias_output",
) -> datasets.Dataset:
    """generate a dataset from an openmm run using the input FF, running
    metadynamics on the torsions.
    Returns:
        A dataset full of MD snapshops with their energies and forces
    """
    # set up an openmm simulation
    molecule = copy.deepcopy(mol)
    molecule.generate_conformers(n_conformers=Nc, rms_cutoff=0.0 * _ANGSTROM)

    n_gen_conformers = len(molecule.conformers)
    if n_gen_conformers < Nc:
        logger.warning(
            f"Only {n_gen_conformers} conformers were generated, which is less than the requested {Nc}."
        )
        Nc = n_gen_conformers

    interchange = openff.interchange.Interchange.from_smirnoff(
        off, openff.toolkit.Topology.from_molecules(molecule)
    )
    omm_system = interchange.to_openmm_system()

    # Get Nc * N_torsions reference torsions
    torsions = get_rot_torsions_by_rot_bond(molecule)
    if not torsions:
        logger.warning("No rotatable bonds found in the molecule.")
        # raise ValueError("No rotatable bonds found in the molecule.")

    # ref_torsions = [
    #     compute_torsion_vector_from_conformer(molecule, conformer)
    #     for conformer in molecule.conformers
    # ]

    # Configure metadynamics
    bias_variables = _get_torsion_bias_forces(
        mol,
        torsions_to_include=_TORSIONS_TO_INCLUDE_SMARTS,
        torsions_to_exclude=_TORSIONS_TO_EXCLUDE_SMARTS,
    )

    if bias_variables == []:
        bias_variables = [None]

    coords, energy, forces, weight = [], [], [], []

    for i, bias_variable in enumerate(bias_variables):
        logger.info(
            f"Running metadynamics for bias variable {i + 1}/{len(bias_variables)}"
        )

        # Randomly named bias directory
        # bias_dir = "bias_output" + np.random.randint(1000, 9999).__str__()
        # pathlib.Path(bias_dir).mkdir(parents=True, exist_ok=True)
        # Append the bias variable index to the directory name
        local_bias_dir = pathlib.Path(bias_dir) / f"bias_{i}"
        local_bias_dir.mkdir(parents=True, exist_ok=True)

        metad = Metadynamics(
            system=omm_system,
            variables=[bias_variable],
            temperature=temperature * _OMM_KELVIN,
            biasFactor=10.0,  # typical range: 5–20
            height=2.0 * openmm.unit.kilojoules_per_mole,
            frequency=500,  # add bias every 500 steps
            saveFrequency=1000,
            biasDir=local_bias_dir,
            independentCVs=True,
        )

        # metad = Metadynamics(
        #     system=omm_system,
        #     variables=bias_variables,
        #     temperature=temperature * _OMM_KELVIN,
        #     biasFactor=5,  # typical range: 5–20
        #     height=0.002 * openmm.unit.kilojoules_per_mole,
        #     frequency=40,  # add bias every 40 steps
        #     saveFrequency=1000,
        #     biasDir=bias_dir,
        #     independentCVs=True,
        # )

        integrator = LangevinMiddleIntegrator(
            temperature * _OMM_KELVIN, 1 / _OMM_PS, dt * _OMM_PS
        )

        simulation = Simulation(
            interchange.to_openmm_topology(),
            omm_system,
            integrator,
        )

        # minimize the system energy and take a snapshot for the ground-state reference
        for conformer in tqdm(
            molecule.conformers,
            leave=False,
            colour="green",
            desc="Generating Snapshots",
        ):
            simulation.positions = conformer
            simulation.context.setPositions(
                interchange.positions.to_openmm()
            )  # Set initial positions for metadynamics
            # Minimize the energy of the initial conformer
            # This is important to ensure the system starts from a low-energy state
            # before running metadynamics
            simulation.minimizeEnergy(maxIterations=100)
            coords.append(
                simulation.context.getState(getPositions=True)
                .getPositions()
                .value_in_unit(_OMM_ANGS)
            )

            # Add pdb reporter to the metadynamics simulation. Note no periodic box vectors
            local_pdb_reporter_path = f"{pdb_reporter_path}_{i}.pdb"
            simulation.reporters.append(
                PDBReporter(local_pdb_reporter_path, MD_stepsize)
            )

            # simulation_ff.step(MD_startup)
            # run the MD and take snapshots
            for step_idx in tqdm(
                range(N - 1), leave=False, colour="red", desc="Running MD"
            ):
                metad.step(simulation, MD_stepsize)
                coords.append(
                    simulation.context.getState(getPositions=True)
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
        range(N * Nc * len(bias_variables)),
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


def get_data_MMMD_torsion_metad(
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
    pdb_reporter_path: str = "trajectory.pdb",
    bias_dir: str = "bias_output",
) -> datasets.Dataset:
    """generate a dataset from an openmm run using the input FF, running
    metadynamics on the torsions.
    Returns:
        A dataset full of MD snapshops with their energies and forces
    """
    # set up an openmm simulation
    molecule = copy.deepcopy(mol)
    molecule.generate_conformers(n_conformers=Nc, rms_cutoff=0.0 * _ANGSTROM)

    n_gen_conformers = len(molecule.conformers)
    if n_gen_conformers < Nc:
        logger.warning(
            f"Only {n_gen_conformers} conformers were generated, which is less than the requested {Nc}."
        )
        Nc = n_gen_conformers

    interchange = openff.interchange.Interchange.from_smirnoff(
        off, openff.toolkit.Topology.from_molecules(molecule)
    )
    omm_system = interchange.to_openmm_system()

    # Get Nc * N_torsions reference torsions
    torsions = get_rot_torsions_by_rot_bond(molecule)
    if not torsions:
        logger.warning("No rotatable bonds found in the molecule.")
        # raise ValueError("No rotatable bonds found in the molecule.")

    # ref_torsions = [
    #     compute_torsion_vector_from_conformer(molecule, conformer)
    #     for conformer in molecule.conformers
    # ]

    # Configure metadynamics
    bias_variables = _get_torsion_bias_forces(
        mol,
        torsions_to_include=_TORSIONS_TO_INCLUDE_SMARTS,
        torsions_to_exclude=_TORSIONS_TO_EXCLUDE_SMARTS,
    )

    # Randomly named bias directory
    # bias_dir = "bias_output" + np.random.randint(1000, 9999).__str__()
    # pathlib.Path(bias_dir).mkdir(parents=True, exist_ok=True)
    bias_dir.mkdir(parents=True, exist_ok=True)

    metad = Metadynamics(
        system=omm_system,
        variables=bias_variables,
        temperature=temperature * _OMM_KELVIN,
        biasFactor=10.0,  # typical range: 5–20
        height=2.0 * openmm.unit.kilojoules_per_mole,
        frequency=500,  # add bias every 500 steps
        saveFrequency=1000,
        biasDir=bias_dir,
        independentCVs=True,
    )

    # metad = Metadynamics(
    #     system=omm_system,
    #     variables=bias_variables,
    #     temperature=temperature * _OMM_KELVIN,
    #     biasFactor=5,  # typical range: 5–20
    #     height=0.002 * openmm.unit.kilojoules_per_mole,
    #     frequency=40,  # add bias every 40 steps
    #     saveFrequency=1000,
    #     biasDir=bias_dir,
    #     independentCVs=True,
    # )

    integrator = LangevinMiddleIntegrator(
        temperature * _OMM_KELVIN, 1 / _OMM_PS, dt * _OMM_PS
    )

    simulation = Simulation(
        interchange.to_openmm_topology(),
        omm_system,
        integrator,
    )

    # minimize the system energy and take a snapshot for the ground-state reference
    coords, energy, forces, weight = [], [], [], []
    for conformer in tqdm(
        molecule.conformers,
        leave=False,
        colour="green",
        desc="Generating Snapshots",
    ):
        simulation.positions = conformer
        simulation.context.setPositions(
            interchange.positions.to_openmm()
        )  # Set initial positions for metadynamics
        # Minimize the energy of the initial conformer
        # This is important to ensure the system starts from a low-energy state
        # before running metadynamics
        simulation.minimizeEnergy(maxIterations=100)
        coords.append(
            simulation.context.getState(getPositions=True)
            .getPositions()
            .value_in_unit(_OMM_ANGS)
        )

        # Add pdb reporter to the metadynamics simulation. Note no periodic box vectors
        simulation.reporters.append(PDBReporter(pdb_reporter_path, MD_stepsize))

        # simulation_ff.step(MD_startup)
        # run the MD and take snapshots
        for step_idx in tqdm(
            range(N - 1), leave=False, colour="red", desc="Running MD"
        ):
            metad.step(simulation, MD_stepsize)
            coords.append(
                simulation.context.getState(getPositions=True)
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


def get_data_MMMD_torsion_metad_old(
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
    pdb_reporter_path: str = "trajectory.pdb",
    pdb_reporter_interval: int = 10,
) -> datasets.Dataset:
    """generate a dataset from an openmm run using the input FF, running
    metadynamics on the torsions.
    Returns:
        A dataset full of MD snapshops with their energies and forces
    """
    # set up an openmm simulation
    molecule = copy.deepcopy(mol)
    molecule.generate_conformers(n_conformers=Nc, rms_cutoff=0.0 * _ANGSTROM)

    n_gen_conformers = len(molecule.conformers)
    if n_gen_conformers < Nc:
        logger.warning(
            f"Only {n_gen_conformers} conformers were generated, which is less than the requested {Nc}."
        )
        Nc = n_gen_conformers

    interchange = openff.interchange.Interchange.from_smirnoff(
        off, openff.toolkit.Topology.from_molecules(molecule)
    )
    integrator = LangevinMiddleIntegrator(
        temperature * _OMM_KELVIN, 1 / _OMM_PS, dt * _OMM_PS
    )
    omm_system = interchange.to_openmm_system()

    # Get Nc * N_torsions reference torsions
    torsions = get_rot_torsions_by_rot_bond(molecule)
    if not torsions:
        raise ValueError("No rotatable bonds found in the molecule.")

    ref_torsions = [
        compute_torsion_vector_from_conformer(molecule, conformer)
        for conformer in molecule.conformers
    ]

    # Create a CustomCVForce for the torsions
    cv_force = make_angular_distance_cv(
        mol,
        torsions,
        ref_torsions,
        lambda_val=2.0,
    )

    # Configure metadynamics
    bias_variable = openmm.app.metadynamics.BiasVariable(
        force=cv_force,
        biasWidth=0.1,
        minValue=0,
        maxValue=1_000_000,  # Arbitrary large max value
    )

    metad = openmm.app.metadynamics.Metadynamics(
        system=omm_system,
        variables=[bias_variable],
        temperature=temperature * _OMM_KELVIN,
        biasFactor=10.0,  # typical range: 5–20
        height=1.0 * openmm.unit.kilojoules_per_mole,
        frequency=500,  # add bias every 500 steps
        saveFrequency=5000,
        biasDir="./bias_output",
    )

    simulation = Simulation(
        interchange.to_openmm_topology(),
        omm_system,
        integrator,
    )

    # minimize the system energy and take a snapshot for the ground-state reference
    coords, energy, forces, weight = [], [], [], []
    for conformer in tqdm(
        molecule.conformers,
        leave=False,
        colour="green",
        desc="Generating Snapshots",
    ):
        simulation.positions = conformer
        simulation.context.setPositions(
            interchange.positions.to_openmm()
        )  # Set initial positions for metadynamics
        # Minimize the energy of the initial conformer
        # This is important to ensure the system starts from a low-energy state
        # before running metadynamics
        simulation.minimizeEnergy(maxIterations=100)
        coords.append(
            simulation.context.getState(getPositions=True)
            .getPositions()
            .value_in_unit(_OMM_ANGS)
        )

        # Add pdb reporter to the metadynamics simulation. Note no periodic box vectors
        simulation.reporters.append(
            PDBReporter(pdb_reporter_path, pdb_reporter_interval)
        )

        # simulation_ff.step(MD_startup)
        # run the MD and take snapshots
        for step_idx in tqdm(
            range(N - 1), leave=False, colour="red", desc="Running MD"
        ):
            metad.step(simulation, MD_stepsize)
            coords.append(
                simulation.context.getState(getPositions=True)
                .getPositions()
                .value_in_unit(_OMM_ANGS)
            )
            # Write to PDBReporter at the specified interval
            if (step_idx + 1) % pdb_reporter_interval == 0:
                simulation.reporters[0].report(
                    simulation, simulation.context.getState(getPositions=True)
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

    n_gen_conformers = len(molecule.conformers)
    if n_gen_conformers < Nc:
        logger.warning(
            f"Only {n_gen_conformers} conformers were generated, which is less than the requested {Nc}."
        )
        Nc = n_gen_conformers

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

    n_gen_conformers = len(molecule.conformers)
    if n_gen_conformers < Nc:
        logger.warning(
            f"Only {n_gen_conformers} conformers were generated, which is less than the requested {Nc}."
        )
        Nc = n_gen_conformers

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
