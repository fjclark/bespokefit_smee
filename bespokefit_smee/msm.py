"""Functionality for applying the modified Seminario method."""

import copy
from contextlib import redirect_stdout
from typing import cast

import loguru
import numpy as np
import openff.interchange
import openff.toolkit
import openmm.unit
import smee
import torch
from numpy import typing as npt
from openff.units import unit as off_unit
from tqdm import tqdm

from . import mlp
from .settings import MSMSettings

logger = loguru.logger

_OMM_KELVIN = openmm.unit.kelvin
_OMM_PS = openmm.unit.picosecond
_OMM_NM = openmm.unit.nanometer
_OMM_ANGS = openmm.unit.angstrom
_OMM_KCAL_PER_MOL_ANGS = openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom


def apply_msm(
    mol: openff.toolkit.Molecule,
    off_ff: openff.toolkit.ForceField,
    tensor_top: smee.TensorTopology,
    tensor_ff: smee.TensorForceField,
    device: torch.device,
    settings: MSMSettings,
) -> smee.TensorForceField:
    """Generate modified Seminario parameters for the bond and angle terms in the
    force-field and apply these to the tensor ff. see doi: 10.1021/acs.jctc.7b00785
    """
    from openmm import LangevinMiddleIntegrator
    from openmm.app.simulation import Simulation

    from .writers import get_potential_comparison

    #   set up an MD sim with the ML potential
    molecule = copy.deepcopy(mol)
    molecule.generate_conformers(n_conformers=1)
    interchange = openff.interchange.Interchange.from_smirnoff(
        off_ff, openff.toolkit.Topology.from_molecules(molecule)
    )
    integrator = LangevinMiddleIntegrator(0 * _OMM_KELVIN, 1 / _OMM_PS, 0.01 * _OMM_PS)
    potential = mlp.get_mlp(settings.ml_potential)
    with open("/dev/null", "w") as f:
        with redirect_stdout(f):
            system = potential.createSystem(
                interchange.to_openmm_topology(),
                charge=mol.total_charge.m_as(off_unit.e),
                device=device,
            )
    simulation = Simulation(interchange.topology, system, integrator)
    #   calculate the ground-state geometry and energy
    interchange.positions = molecule.conformers[0]
    simulation.context.setPositions(interchange.positions.to_openmm())
    simulation.minimizeEnergy(maxIterations=0, tolerance=settings.tolerance)
    position = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
    crd0 = position.value_in_unit(_OMM_NM).reshape(3 * molecule.n_atoms)
    #   extract bond info from the smee tensor
    bonds_obj = cast(
        smee.ValenceParameterMap, copy.deepcopy(tensor_top.parameters["Bonds"])
    )
    n_bonds = len(bonds_obj.assignment_matrix.indices()[0].detach().flatten().tolist())
    n_bond_types = (
        max(bonds_obj.assignment_matrix.indices()[-1].detach().flatten().tolist()) + 1
    )
    bond_types = [
        [
            i
            for i, x in enumerate(bonds_obj.assignment_matrix.indices()[-1].tolist())
            if x == j
        ]
        for j in range(n_bond_types)
    ]
    bond_indxs = bonds_obj.particle_idxs.tolist()
    #   extract angle info from the smee tensor
    angles_obj = cast(
        smee.ValenceParameterMap, copy.deepcopy(tensor_top.parameters["Angles"])
    )
    n_angles = len(
        angles_obj.assignment_matrix.indices()[0].detach().flatten().tolist()
    )
    n_angle_types = (
        max(angles_obj.assignment_matrix.indices()[-1].detach().flatten().tolist()) + 1
    )
    angle_types = [
        [
            i
            for i, x in enumerate(angles_obj.assignment_matrix.indices()[-1].tolist())
            if x == j
        ]
        for j in range(n_angle_types)
    ]
    angle_indxs = angles_obj.particle_idxs.tolist()
    #   calculate hessian elements with finite difference, ignoring the diagonal and all below
    hessian = np.zeros((3 * molecule.n_atoms, 3 * molecule.n_atoms))
    for i in tqdm(
        range(n_bonds), leave=False, colour="green", desc="Generating Hessian Fragments"
    ):
        i1, i2 = bond_indxs[i][0] * 3, bond_indxs[i][1] * 3
        for j1 in range(i1, i1 + 3):
            crd = crd0
            crd[j1] += settings.finite_step
            simulation.context.setPositions(
                crd.reshape(molecule.n_atoms, 3)
            )  # coords +
            f1 = (
                simulation.context.getState(getForces=True)
                .getForces(asNumpy=True)
                .value_in_unit(_OMM_KCAL_PER_MOL_ANGS)
            )
            dEp = -f1[i2 // 3]
            crd[j1] -= 2 * settings.finite_step
            simulation.context.setPositions(
                crd.reshape(molecule.n_atoms, 3)
            )  # coords -
            f1 = (
                simulation.context.getState(getForces=True)
                .getForces(asNumpy=True)
                .value_in_unit(_OMM_KCAL_PER_MOL_ANGS)
            )
            dEm = -f1[i2 // 3]
            hessian[j1, range(i2, i2 + 3)] = (dEp - dEm) / (2 * settings.finite_step)
    #   calculate mod-seminario force constants along the bonds and group by bond-type, as given in the smee tensors
    bond_k, bond_l = [], []
    for j in range(n_bond_types):
        k_sum, l_sum = 0.0, 0.0
        for i in bond_types[j]:
            iA, iB = bond_indxs[i][0], bond_indxs[i][1]
            jA, jB = iA * 3, iB * 3
            b = (
                position.value_in_unit(_OMM_ANGS)[iA]
                - position.value_in_unit(_OMM_ANGS)[iB]
            )
            norm_b = np.linalg.norm(b)
            k_sum += modSem_projection(-hessian[jA : jA + 3, jB : jB + 3], b / norm_b)
            l_sum += float(norm_b)
        bond_k.append(k_sum * settings.vib_scaling**2 * 0.1 / len(bond_types[j]))
        bond_l.append(l_sum / len(bond_types[j]))
    #   calculate mod-seminario force constants along around the angles and group by angle-type, as given in the smee tensors
    angle_k, angle_t = [], []
    for j in range(n_angle_types):
        k_sum, t_sum = 0.0, 0.0
        for i in angle_types[j]:
            iA, iB, iC = angle_indxs[i][0], angle_indxs[i][1], angle_indxs[i][2]
            jA, jB, jC = iA * 3, iB * 3, iC * 3
            bAB = (
                position.value_in_unit(_OMM_ANGS)[iA]
                - position.value_in_unit(_OMM_ANGS)[iB]
            )
            bCB = (
                position.value_in_unit(_OMM_ANGS)[iC]
                - position.value_in_unit(_OMM_ANGS)[iB]
            )
            if iA > iB:
                HAB = -hessian[jB : jB + 3, jA : jA + 3]
            else:
                HAB = -hessian[jA : jA + 3, jB : jB + 3]
            if iC > iB:
                HCB = -hessian[jB : jB + 3, jC : jC + 3]
            else:
                HCB = -hessian[jC : jC + 3, jB : jB + 3]
            lAB, lCB = np.linalg.norm(bAB), np.linalg.norm(bCB)
            uAB, uCB = bAB / lAB, bCB / lCB
            uN = unit_normal_vector(uAB, uCB)
            uPA, uPC = unit_normal_vector(uN, uAB), unit_normal_vector(uCB, uN)
            kPA, kPC = modSem_projection(HAB, uPA), modSem_projection(HCB, uPC)
            fixA, fixC = 0.0, 0.0
            NfA, NfC = 0.0, 0.0
            for jj in range(n_angles):
                iiA, iiB, iiC = (
                    angle_indxs[jj][0],
                    angle_indxs[jj][1],
                    angle_indxs[jj][2],
                )
                if iiB == iB & jj != i:
                    if iiA == iA:
                        bCBp = (
                            position.value_in_unit(_OMM_ANGS)[iiC]
                            - position.value_in_unit(_OMM_ANGS)[iiB]
                        )
                        uPAp = unit_normal_vector(
                            unit_normal_vector(uAB, bCBp / np.linalg.norm(bCBp)), uAB
                        )
                        fixA += np.dot(uPA, uPAp) ** 2
                        NfA += 1
                    elif iiC == iC:
                        bABp = (
                            position.value_in_unit(_OMM_ANGS)[iiA]
                            - position.value_in_unit(_OMM_ANGS)[iiB]
                        )
                        uPCp = unit_normal_vector(
                            unit_normal_vector(uCB, bABp / np.linalg.norm(bABp)), uCB
                        )
                        fixC += np.dot(uPC, uPCp) ** 2
                        NfC += 1
            if NfA > 0:
                fixA = fixA / NfA
            if NfC > 0:
                fixC = fixC / NfC
            k_sum += float(
                1 / (((1 + fixA) / (lAB**2 * kPA)) + ((1 + fixC) / (lCB**2 * kPC)))
            )
            t_sum += np.arccos(np.dot(uAB, uCB))
        angle_k.append(k_sum * settings.vib_scaling**2 * 0.1 / len(angle_types[j]))
        angle_t.append(t_sum / len(angle_types[j]))
    #   put the new constants into the force-field object and report!
    tensor_ff_out = copy.deepcopy(tensor_ff)
    tensor_ff_out.potentials_by_type["Bonds"].parameters = torch.tensor(
        [[bond_k[j], bond_l[j]] for j in range(n_bond_types)]
    )
    tensor_ff_out.potentials_by_type["Angles"].parameters = torch.tensor(
        [[angle_k[j], angle_t[j]] for j in range(n_angle_types)]
    )
    bond_potential_comparison = get_potential_comparison(
        tensor_ff.potentials_by_type["Bonds"],
        tensor_ff_out.potentials_by_type["Bonds"],
    )
    angle_potential_comparison = get_potential_comparison(
        tensor_ff.potentials_by_type["Angles"],
        tensor_ff_out.potentials_by_type["Angles"],
    )

    logger.info(
        "Modified Seminario Summary:"
        f"{bond_potential_comparison}"
        f"{angle_potential_comparison}"
    )

    return tensor_ff_out


def unit_normal_vector(
    u1: npt.NDArray[np.float64], u2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return a unit vector perpendicular to the two input vectors"""
    cross = np.cross(u1, u2)
    return cross / np.linalg.norm(cross)


def modSem_projection(
    parhess: npt.NDArray[np.float64], unit_vector: npt.NDArray[np.float64]
) -> float:
    """Return a spring constant projected out of a partial hessian onto a unit vector"""
    vals, vecs = np.linalg.eig(parhess)
    kab1 = sum(abs(np.dot(unit_vector, vecs[:, i])) * vals[i] for i in range(3)).real
    kba1 = sum(
        abs(np.dot(unit_vector[::-1], vecs[:, i])) * vals[i] for i in range(3)
    ).real
    vals, vecs = np.linalg.eig(parhess.transpose())
    kab2 = sum(abs(np.dot(unit_vector, vecs[:, i])) * vals[i] for i in range(3)).real
    kba2 = sum(
        abs(np.dot(unit_vector[::-1], vecs[:, i])) * vals[i] for i in range(3)
    ).real
    return float(0.25 * (kab1 + kba1 + kab2 + kba2))
