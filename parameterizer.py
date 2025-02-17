"""
PARAMETERIZE:

force-field parameterization functions
"""
###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################
from contextlib import redirect_stdout
import sys
import copy
import math
import numpy as np
import collections
import openff.interchange
import openff.toolkit
from openff.units import unit
import smee
import smee.converters
import torch
import torch.distributed
import pydantic
from rdkit import Chem
from tqdm import tqdm
###############################################################################
############################### FUNCTIONS #####################################
###############################################################################

_UNITLESS           = unit.dimensionless
_ANGSTROM           = unit.angstrom
_RADIANS            = unit.radians
_KCAL_PER_MOL       = unit.kilocalories / unit.mole
_KCAL_PER_MOL_ANGSQ = unit.kilocalories / unit.mole / unit.angstrom ** 2
_KCAL_PER_MOL_RADSQ = unit.kilocalories / unit.mole / unit.radians ** 2

def _create_smarts(mol: openff.toolkit.Molecule, idxs: torch.Tensor) -> str:
    """Create a mapped SMARTS representation of a molecule."""
    from rdkit import Chem

    mol_rdkit = mol.to_rdkit()

    for i, idx in enumerate(idxs):
        atom = mol_rdkit.GetAtomWithIdx(int(idx))
        atom.SetAtomMapNum(i + 1)

    smarts = Chem.MolToSmarts(mol_rdkit)
    return smarts

def _prepare_potential(
    mol: openff.toolkit.Molecule,
    symmetries: list[int],
    potential: smee.TensorPotential,
    parameter_map: smee.ParameterMap,
) -> None:
    """Prepare a potential to use bespoke parameters for each 'slot'."""

    is_indexed = any(key.mult is not None for key in potential.parameter_keys)

    ids_to_parameter_idxs = collections.defaultdict(set)
    ids_to_particle_idxs = collections.defaultdict(set)

    ids_to_smarts = {}

    for particle_idxs, assignment_row in zip(
        parameter_map.particle_idxs,
        parameter_map.assignment_matrix.to_dense(),
        strict=True,
    ):
        particle_idxs = tuple(int(idx) for idx in particle_idxs)
        particle_ids = tuple(symmetries[idx] for idx in particle_idxs)

        if potential.type != "ImproperTorsions" and particle_ids[-1] < particle_ids[0]:
            particle_ids = particle_ids[::-1]

        parameter_idxs = [
            parameter_idx
            for parameter_idx, value in enumerate(assignment_row)
            if int(value) != 0
        ]
        assert len(parameter_idxs) == 1

        ids_to_parameter_idxs[particle_ids].add(parameter_idxs[0])
        ids_to_particle_idxs[particle_ids].add(particle_idxs)

        if potential.type == "ImproperTorsions":
            particle_idxs = (
                particle_idxs[1],
                particle_idxs[0],
                particle_idxs[2],
                particle_idxs[3],
            )

        ids_to_smarts[particle_ids] = _create_smarts(mol, particle_idxs)

    ids_to_parameter_idxs = {
        particle_ids: sorted(parameter_idxs)
        for particle_ids, parameter_idxs in ids_to_parameter_idxs.items()
    }

    parameter_ids = [
        (particle_ids, parameter_idx)
        for particle_ids, parameter_idxs in ids_to_parameter_idxs.items()
        for parameter_idx in parameter_idxs
    ]
    potential.parameters = potential.parameters[
        [parameter_idx for _, parameter_idx in parameter_ids]
    ]
    potential.parameter_keys = [
        openff.interchange.models.PotentialKey(
            id=ids_to_smarts[particle_ids],
            mult=ids_to_parameter_idxs[particle_ids].index(parameter_idx)
            if is_indexed
            else None,
            associated_handler=potential.type,
        )
        for particle_ids, parameter_idx in parameter_ids
    ]

    assignment_matrix = smee.utils.zeros_like(
        (len(parameter_map.particle_idxs), len(potential.parameters)),
        parameter_map.assignment_matrix,
    )
    particle_idxs_updated = []

    for particle_ids, particle_idxs in ids_to_particle_idxs.items():
        for particle_idx in particle_idxs:
            for parameter_idx in ids_to_parameter_idxs[particle_ids]:
                j = parameter_ids.index((particle_ids, parameter_idx))

                assignment_matrix[len(particle_idxs_updated), j] = 1
                particle_idxs_updated.append(particle_idx)

    parameter_map.particle_idxs = smee.utils.tensor_like(
        particle_idxs_updated, parameter_map.particle_idxs
    )
    parameter_map.assignment_matrix = assignment_matrix.to_sparse()
    
def apply_parameters(
    mol: openff.toolkit.Molecule,
    off: openff.toolkit.ForceField,
    ML_path: str,
) -> tuple[smee.TensorForceField, smee.TensorTopology]:
    """Prepare a tensor force field that contains unique parameters for each
    topologically symmetric term of a molecule.

    Args:
        smiles: The molecule to prepare bespoke parameters for.
        force_field_paths: The base force field to copy the parameters from.

    Returns:
        The prepared tensor force field and topology ready for fitting.
    """
    force_field, [topology] = smee.converters.convert_interchange(openff.interchange.Interchange.from_smirnoff(off,mol.to_topology()))
    symmetries = list(Chem.CanonicalRankAtoms(mol.to_rdkit(), breakTies=False))
    if topology.n_v_sites != 0:
        raise NotImplementedError("virtual sites are not supported yet.")
    for potential in force_field.potentials:
        parameter_map = topology.parameters[potential.type]
        if isinstance(parameter_map, smee.NonbondedParameterMap):
            continue
        _prepare_potential(mol, symmetries, potential, parameter_map)
#    modSeminario(mol,off,ML_path,force_field)
    return force_field, topology    

def expand_torsions(ff: openff.toolkit.ForceField) -> openff.toolkit.ForceField:
    """Expand the torsion potential to include K0-4 for proper torsions"""
    ff_copy         = copy.deepcopy(ff)
    torsion_handler = ff_copy.get_parameter_handler("ProperTorsions")
    for parameter in torsion_handler:
        # set the defaults
        parameter.idivf = [1.0] * 4
        default_k       = [0 * unit.kilocalories_per_mole] * 4
        default_phase   = [0 * unit.degree ] * 4
        default_p       = [1, 2, 3, 4]
        # update the existing k values for the correct phase and p
        for i, p in enumerate(parameter.periodicity):
            try:
                default_k[p - 1]     = parameter.k[i]
                default_phase[p - 1] = parameter.phase[i]
            except IndexError:
                continue
        # update with new parameters
        parameter.k           = default_k
        parameter.phase       = default_phase
        parameter.periodicity = default_p
    return ff_copy

def modSeminario(
    mol: openff.toolkit.Molecule,
    off: openff.toolkit.ForceField,
    ML_path: str,
    sff: smee.TensorForceField,
    finite_step: float = 0.005291772,
):
    from openmm.app.simulation import Simulation
    from openmm import LangevinMiddleIntegrator
    import openmm.unit
    from openmmml import MLPotential
    molecule        = copy.deepcopy(mol)
    molecule.generate_conformers(n_conformers=1)
    off_topology    = openff.toolkit.Topology.from_molecules(molecule)
    force_field     = copy.deepcopy(off)
    interchange     = openff.interchange.Interchange.from_smirnoff(force_field,off_topology)
    print(interchange.collections,flush=True)
    integrator      = LangevinMiddleIntegrator(300,1,0.01)
    potential       = MLPotential(ML_path)
    with open("/dev/null", 'w') as f:
        with redirect_stdout(f):
            system  = potential.createSystem(interchange.to_openmm_topology())
    simulation            = Simulation(interchange.topology, system, integrator)
    interchange.positions = molecule.conformers[0]
    position              = interchange.positions.to_openmm()
    simulation.context.setPositions(position)
    simulation.minimizeEnergy(maxIterations=0,tolerance=0.1)
    state   = simulation.context.getState(getEnergy=True)
    e0      = state.getPotentialEnergy().value_in_unit(openmm.unit.kilocalorie_per_mole)
    hessian = np.zeros((3 * molecule.n_atoms, 3 * molecule.n_atoms))
    coords0 = position.value_in_unit(openmm.unit.nanometer).reshape(3 * molecule.n_atoms)
    for i in tqdm(range(3 * molecule.n_atoms),leave=False,colour='green',desc="Generating Hessian Fragments"):
        for j in range(i + 1, 3 * molecule.n_atoms): # ignore diagonal and everything underneath
            coords = coords0
            coords[i] += finite_step
            coords[j] += finite_step
            simulation.context.setPositions(coords.reshape(molecule.n_atoms,3)) # coords ++
            state      = simulation.context.getState(getEnergy=True)
            e1         = state.getPotentialEnergy().value_in_unit(openmm.unit.kilocalorie_per_mole)
            coords[j]  = coords0[j] - finite_step
            simulation.context.setPositions(coords.reshape(molecule.n_atoms,3)) # coords +-
            state      = simulation.context.getState(getEnergy=True)
            e3         = state.getPotentialEnergy().value_in_unit(openmm.unit.kilocalorie_per_mole)
            coords[i]  = coords0[i] - finite_step
            simulation.context.setPositions(coords.reshape(molecule.n_atoms,3)) # coords --
            state      = simulation.context.getState(getEnergy=True)
            e2         = state.getPotentialEnergy().value_in_unit(openmm.unit.kilocalorie_per_mole)
            coords[j]  = coords0[j] + finite_step
            simulation.context.setPositions(coords.reshape(molecule.n_atoms,3)) # coords -+
            state      = simulation.context.getState(getEnergy=True)
            e4         = state.getPotentialEnergy().value_in_unit(openmm.unit.kilocalorie_per_mole)
          # Hess = [E(dx + dy) + E(-dx - dy) - E(dx - dy) - E(-dx + dy)] / 4 dx dy
          # dx and dy multiplied by 10 to turn nm into ang
            hessian[i,j] = (e1 + e2 - e3 - e4) / ( 4 * (10 * finite_step) ** 2 * np.sqrt(molecule.atom(i // 3).mass.m_as(unit.amu) * molecule.atom(j // 3).mass.m_as(unit.amu)))
    vecs = np.empty((3, 3), dtype=complex)
    vals = np.empty((3), dtype=complex)
    for i,single_bond in enumerate(molecule.bonds):
        i1              = single_bond.atom1.molecule_atom_index * 3
        i2              = single_bond.atom2.molecule_atom_index * 3
        b               = position.value_in_unit(openmm.unit.angstrom)[single_bond.atom1.molecule_atom_index] - position.value_in_unit(openmm.unit.angstrom)[single_bond.atom2.molecule_atom_index]
        length          = np.linalg.norm(b)
        b               = b / length
        partial_hessian = -hessian[i1:i1+3,i2:i2+3]
        vals, vecs      = np.linalg.eig(partial_hessian)
      # take average of (+)ve and (-)ve bond vectors. Also, multiply by two to account for the factor of 1/2 in the FF equation
        k               = abs(sum((np.dot(vecs[:,j],b) * vals[j]).real for j in range(3))) + abs(sum((np.dot(vecs[:,j],b[::-1]) * vals[j]).real for j in range(3)))
        print("B",i,":",single_bond.atom1.molecule_atom_index,single_bond.atom2.molecule_atom_index,":",k,length,flush=True)

    for i,single_angle in enumerate(molecule.angles):
        print("A",i,":",single_angle[0].molecule_atom_index,single_angle[1].molecule_atom_index,single_angle[2].molecule_atom_index)

class ParameterConfig(pydantic.BaseModel):
    """Configuration for how a potential's parameters should be trained."""

    cols: list[str] = pydantic.Field(
        description="The parameters to train, e.g. 'k', 'length', 'epsilon'."
    )

    scales: dict[str, float] = pydantic.Field(
        {},
        description="The scales to apply to each parameter, e.g. 'k': 1.0, "
        "'length': 1.0, 'epsilon': 1.0.",
    )
    constraints: dict[str, tuple[float | None, float | None]] = pydantic.Field(
        {},
        description="The min and max values to clamp each parameter within, e.g. "
        "'k': (0.0, None), 'angle': (0.0, pi), 'epsilon': (0.0, None), where "
        "none indicates no constraint.",
    )

def convert_to_smirnoff(
    ff: smee.TensorForceField, base: openff.toolkit.ForceField | None = None
) -> openff.toolkit.ForceField:
    """Convert a tensor force field that *contains bespoke valence parameters* to
    SMIRNOFF format.

    Notes:
        * Currently only the valence terms are converted into SMIRNOFF format.
        * Currently only bond, angle, torsion and improper potentials are supported.

    See Also:
        `befit.ff.prepare`

    Args:
        ff: The force field containing the bespoke valence terms.
        base: The (optional) original SMIRNOFF force field to add the bespoke
            parameters to. If no specified, a force field containing only the bespoke
            parameters will be returned.

    Returns:
        A SMIRNOFF force field containing the valence terms of the input force field.
    """
    ff_smirnoff = openff.toolkit.ForceField() if base is None else copy.deepcopy(base)
            
    for potential in ff.potentials:
        if potential.type in {"Bonds", "Angles", "ProperTorsions", "ImproperTorsions"}:
            assert potential.attribute_cols is None
            parameters_by_smarts = collections.defaultdict(dict)
            for parameter, parameter_key in zip(
                potential.parameters, potential.parameter_keys, strict=True
            ):
                assert parameter_key.mult not in parameters_by_smarts[parameter_key.id]
                parameters_by_smarts[parameter_key.id][parameter_key.mult] = parameter
            handler = ff_smirnoff.get_parameter_handler(potential.type)
            for smarts, parameters_by_mult in parameters_by_smarts.items():
                mults = {*parameters_by_mult}
                if None in mults and len(mults) > 1:
                    raise NotImplementedError("unexpected parameters found")
                if None not in mults and mults != {*range(len(mults))}:
                    raise NotImplementedError("unexpected parameters found")
                counter = len(handler.parameters) + 1
                parameter_id = f"{potential.type[0].lower()}-bespoke-{counter}"
                parameter_dict = {"smirks": smarts, "id": parameter_id}
                parameter_dict.update(
                    {
                        (col if mult is None else f"{col}{mult + 1}"): float(
                            parameter[col_idx]
                        )
                        * potential.parameter_units[col_idx]
                        for mult, parameter in parameters_by_mult.items()
                        for col_idx, col in enumerate(potential.parameter_cols)
                    }
                )
                handler.add_parameter(parameter_dict)
        elif potential.type=="LBonds":
            assert potential.attribute_cols is None
            parameters_by_smarts = collections.defaultdict(dict)
            new_params = []
            for param in potential.parameters:
                k1  = param[0].item()
                k2  = param[1].item()
                b1  = param[2].item()
                b2  = param[3].item()
                k   = k1 + k2
                b   = (k1 * b1 + k2 * b2) / k
                dt = param.dtype
                new_params.append([k,b])
            reconstructed_param = torch.tensor(new_params,dtype=dt)
            reconstructed_units = (_KCAL_PER_MOL_ANGSQ, _ANGSTROM)
            reconstructed_cols  = ("k", "length")          
            for parameter, parameter_key in zip(
                reconstructed_param, potential.parameter_keys, strict=True
            ):
                assert parameter_key.mult not in parameters_by_smarts[parameter_key.id]
                parameters_by_smarts[parameter_key.id][parameter_key.mult] = parameter
            handler = ff_smirnoff.get_parameter_handler("Bonds")
            for smarts, parameters_by_mult in parameters_by_smarts.items():
                mults = {*parameters_by_mult}
                if None in mults and len(mults) > 1:
                    raise NotImplementedError("unexpected parameters found")
                if None not in mults and mults != {*range(len(mults))}:
                    raise NotImplementedError("unexpected parameters found")
                counter = len(handler.parameters) + 1
                parameter_id = f"{potential.type[0].lower()}-bespoke-{counter}"
                parameter_dict = {"smirks": smarts, "id": parameter_id}
                parameter_dict.update(
                    {
                        (col if mult is None else f"{col}{mult + 1}"): float(parameter[col_idx]) * reconstructed_units[col_idx]
                        for mult, parameter in parameters_by_mult.items()
                        for col_idx, col in enumerate(reconstructed_cols)
                    }
                )
                handler.add_parameter(parameter_dict)
        elif potential.type=="LAngles":
            assert potential.attribute_cols is None
            parameters_by_smarts = collections.defaultdict(dict)
            new_params = []
            for param in potential.parameters:
                k1  = param[0].item()
                k2  = param[1].item()
                a1  = param[2].item()
                a2  = param[3].item()
                k   = k1 + k2
                a   = (k1 * a1 + k2 * a2) / k
                dt = param.dtype
                new_params.append([k,a])
            reconstructed_param = torch.tensor(new_params,dtype=dt)
            reconstructed_units = (_KCAL_PER_MOL_RADSQ, _RADIANS)
            reconstructed_cols  = ("k", "angle")          
            for parameter, parameter_key in zip(reconstructed_param, potential.parameter_keys, strict=True):
                assert parameter_key.mult not in parameters_by_smarts[parameter_key.id]
                parameters_by_smarts[parameter_key.id][parameter_key.mult] = parameter
            handler = ff_smirnoff.get_parameter_handler("Angles")
            for smarts, parameters_by_mult in parameters_by_smarts.items():
                mults = {*parameters_by_mult}
                if None in mults and len(mults) > 1:
                    raise NotImplementedError("unexpected parameters found")
                if None not in mults and mults != {*range(len(mults))}:
                    raise NotImplementedError("unexpected parameters found")
                counter = len(handler.parameters) + 1
                parameter_id = f"{potential.type[0].lower()}-bespoke-{counter}"
                parameter_dict = {"smirks": smarts, "id": parameter_id}
                parameter_dict.update(
                    {
                        (col if mult is None else f"{col}{mult + 1}"): float(parameter[col_idx]) * reconstructed_units[col_idx]
                        for mult, parameter in parameters_by_mult.items()
                        for col_idx, col in enumerate(reconstructed_cols)
                    }
                )
                handler.add_parameter(parameter_dict)
        elif potential.type=="LinearProperTorsions":
            assert potential.attribute_cols is None
            parameters_by_smarts = collections.defaultdict(dict)
            new_params = []
            for param in potential.parameters:
                k1           = param[0].item()
                k2           = param[1].item()
                periodicity  = param[2].item()
                phase1       = param[3].item()
                phase2       = param[4].item()
                idivf        = param[5].item()
                k     = k1 + k2
                if k == 0.0: 
                    phase = 0.0
                else:
                    phase = math.acos((k1 - k2) / k)
#                    phase = math.acos((k1 * math.cos(phase1) + k2 * math.cos(phase2))/k)
                dt = param.dtype
                new_params.append([k,periodicity,phase,idivf])
            reconstructed_param = torch.tensor(new_params,dtype=dt)
            reconstructed_units = (_KCAL_PER_MOL, _UNITLESS, _RADIANS, _UNITLESS)
            reconstructed_cols  = ("k", "periodicity", "phase", "idivf")
            for parameter, parameter_key in zip(reconstructed_param, potential.parameter_keys, strict=True):
                assert parameter_key.mult not in parameters_by_smarts[parameter_key.id]
                parameters_by_smarts[parameter_key.id][parameter_key.mult] = parameter          
            handler = ff_smirnoff.get_parameter_handler("ProperTorsions")
            for smarts, parameters_by_mult in parameters_by_smarts.items():
                mults = {*parameters_by_mult}
                if None in mults and len(mults) > 1:
                    raise NotImplementedError("unexpected parameters found")
                if None not in mults and mults != {*range(len(mults))}:
                    raise NotImplementedError("unexpected parameters found")
                counter = len(handler.parameters) + 1
                parameter_id = f"{potential.type[0].lower()}-bespoke-{counter}"
                parameter_dict = {"smirks": smarts, "id": parameter_id}
                parameter_dict.update(
                    {
                        (col if mult is None else f"{col}{mult + 1}"): float(parameter[col_idx]) * reconstructed_units[col_idx]
                        for mult, parameter in parameters_by_mult.items()
                        for col_idx, col in enumerate(reconstructed_cols)
                    }
                )
                handler.add_parameter(parameter_dict)
        elif potential.type=="LinearImproperTorsions":
            assert potential.attribute_cols is None
            parameters_by_smarts = collections.defaultdict(dict)
            new_params = []
            for param in potential.parameters:
                k1           = param[0].item()
                k2           = param[1].item()
                periodicity  = param[2].item()
                phase1       = param[3].item()
                phase2       = param[4].item()
                idivf        = param[5].item()
                k     = k1 + k2
                if k == 0.0: 
                    phase = 0.0
                else:
                    phase = math.acos((k1 - k2) / k)
#                    phase = math.acos((k1 * math.cos(phase1) + k2 * math.cos(phase2))/k)
                dt = param.dtype
                new_params.append([k,periodicity,phase,idivf])
            reconstructed_param = torch.tensor(new_params,dtype=dt)
            reconstructed_units = (_KCAL_PER_MOL, _UNITLESS, _RADIANS, _UNITLESS)
            reconstructed_cols  = ("k", "periodicity", "phase", "idivf")
            for parameter, parameter_key in zip(reconstructed_param, potential.parameter_keys, strict=True):
                assert parameter_key.mult not in parameters_by_smarts[parameter_key.id]
                parameters_by_smarts[parameter_key.id][parameter_key.mult] = parameter          
            handler = ff_smirnoff.get_parameter_handler("ImproperTorsions")
            for smarts, parameters_by_mult in parameters_by_smarts.items():
                mults = {*parameters_by_mult}
                if None in mults and len(mults) > 1:
                    raise NotImplementedError("unexpected parameters found")
                if None not in mults and mults != {*range(len(mults))}:
                    raise NotImplementedError("unexpected parameters found")
                counter = len(handler.parameters) + 1
                parameter_id = f"{potential.type[0].lower()}-bespoke-{counter}"
                parameter_dict = {"smirks": smarts, "id": parameter_id}
                parameter_dict.update(
                    {
                        (col if mult is None else f"{col}{mult + 1}"): float(parameter[col_idx]) * reconstructed_units[col_idx]
                        for mult, parameter in parameters_by_mult.items()
                        for col_idx, col in enumerate(reconstructed_cols)
                    }
                )
                handler.add_parameter(parameter_dict)
        
    return ff_smirnoff

class TrainableParameters:
    """A wrapper around a SMEE force field that handles zeroing out gradients of
    fixed parameters and applying parameter constraints."""

    def __init__(
        self,
        force_field: smee.TensorForceField,
        parameters: dict[str, ParameterConfig],
    ):
        self.potential_types = [*parameters]
        self._force_field = force_field

        potentials = [
            force_field.potentials_by_type[potential_type]
            for potential_type in self.potential_types
        ]

        self._frozen_cols = [
            [
                i
                for i, col in enumerate(potential.parameter_cols)
                if col not in parameters[potential_type].cols
            ]
            for potential_type, potential in zip(self.potential_types, potentials)
        ]

        self._scales = [
            torch.tensor(
                [
                    parameters[potential_type].scales.get(col, 1.0)
                    for col in potential.parameter_cols
                ]
            ).reshape(1, -1)
            for potential_type, potential in zip(self.potential_types, potentials)
        ]
        self._constraints = [
            {
                i: parameters[potential_type].constraints[col]
                for i, col in enumerate(potential.parameter_cols)
                if col in parameters[potential_type].constraints
            }
            for potential_type, potential in zip(self.potential_types, potentials)
        ]

        self.parameters = [
            (potential.parameters.detach().clone() * scale).requires_grad_()
            for potential, scale in zip(potentials, self._scales)
        ]

    @property
    def force_field(self) -> smee.TensorForceField:
        for potential_type, parameter, scale in zip(
            self.potential_types, self.parameters, self._scales
        ):
            potential = self._force_field.potentials_by_type[potential_type]
            potential.parameters = parameter / scale

        return self._force_field

    @torch.no_grad()
    def clamp(self):
        for parameter, constraints in zip(self.parameters, self._constraints):
            for i, (min_value, max_value) in constraints.items():
                if min_value is not None:
                    parameter[:, i].clamp_(min=min_value)
                if max_value is not None:
                    parameter[:, i].clamp_(max=max_value)

    @torch.no_grad()
    def freeze_grad(self):
        for parameter, col_idxs in zip(self.parameters, self._frozen_cols):
            parameter.grad[:, col_idxs] = 0.0          

def linearize_bonds(ff: smee.TensorForceField) -> smee.TensorForceField:         
    """Linearize the bond parameters in the forcefield for more robust optimization"""
    ff_copy = copy.deepcopy(ff)
    ff_copy.potentials = []
    for potential in ff.potentials:
        if potential.type in {"Bonds"}:
            new_potential = copy.deepcopy(potential)
            new_potential.type = "LBonds"
            new_potential.fn = "k1/2*(r-b1)**2+k2/2*(r-b2)**2"
            new_potential.parameter_cols = ("k1", "k2", "b1", "b2")
            new_params = []
            for param in potential.parameters:
                k  = param[0].item()
                b  = param[1].item()
                dt = param.dtype
                b1 = b * 0.9
                b2 = b * 1.1
                d  = b2 - b1
                k1 = k * (b2 - b) / d
                k2 = k * (b - b1) / d
                new_params.append([k1,k2,b1,b2])
            new_potential.parameters = torch.tensor(new_params,dtype=dt)
            new_potential.parameter_units = (
                _KCAL_PER_MOL_ANGSQ, 
                _KCAL_PER_MOL_ANGSQ, 
                _ANGSTROM, 
                _ANGSTROM
            )
            ff_copy.potentials.append(new_potential)
        else:
            ff_copy.potentials.append(potential)
    return ff_copy
           
def linearize_angles(ff: smee.TensorForceField) -> smee.TensorForceField:         
    """Linearize the bond parameters in the forcefield for more robust optimization"""
    ff_copy = copy.deepcopy(ff)
    ff_copy.potentials = []
    for potential in ff.potentials:
        if potential.type in {"Angles"}:
            new_potential = copy.deepcopy(potential)
            new_potential.type = "LAngles"
            new_potential.fn = "k1/2*(theta-angle1)**2+k2/2*(theta-angle2)**2"
            new_potential.parameter_cols = ("k1", "k2", "angle1", "angle2")
            new_params = []
            for param in potential.parameters:
                k  = param[0].item()
                a  = param[1].item()
                dt = param.dtype
                a1 = a * 0.9
                a2 = a * 1.1
                d  = a2 - a1
                k1 = k * (a2 - a) / d
                k2 = k * (a - a1) / d
                new_params.append([k1,k2,a1,a2])
            new_potential.parameters = torch.tensor(new_params,dtype=dt)
            new_potential.parameter_units = (
                _KCAL_PER_MOL_RADSQ, 
                _KCAL_PER_MOL_RADSQ, 
                _RADIANS, 
                _RADIANS
            )
            ff_copy.potentials.append(new_potential)
        else:
            ff_copy.potentials.append(potential)
    return ff_copy

def linearize_torsions_1(ff: smee.TensorForceField) -> smee.TensorForceField:         
    """Linearize the bond parameters in the forcefield for more robust optimization"""
    ff_copy = copy.deepcopy(ff)
    ff_copy.potentials = []
    for potential in ff.potentials:
        if potential.type in {"ProperTorsions"}:
            new_potential = copy.deepcopy(potential)
            new_potential.type = "LinearProperTorsions"
            new_potential.fn = "k1*(1+cos(periodicity*theta-phase1))+k2*(1+cos(periodicity*theta-phase2))"
            new_potential.parameter_cols = ("k1", "k2", "periodicity", "phase1", "phase2", "idivf")
            new_params = []
            for param in potential.parameters:
                k            = param[0].item()
                periodicity  = param[1].item()
                phase        = param[2].item()
                idivf        = param[3].item()
                dt = param.dtype
                k1 = abs(k * 0.5 * (1 + math.cos(phase)))
                k2 = abs(k * 0.5 * (1 - math.cos(phase)))
                new_params.append([k1,k2,periodicity,0.0,math.pi,idivf])
            new_potential.parameters = torch.tensor(new_params,dtype=dt)
            new_potential.parameter_units = (_KCAL_PER_MOL,_KCAL_PER_MOL,_UNITLESS,_RADIANS,_RADIANS,_UNITLESS)
            ff_copy.potentials.append(new_potential)
        else:
            ff_copy.potentials.append(potential)
    return ff_copy

def linearize_torsions_2(ff: smee.TensorForceField) -> smee.TensorForceField:         
    """Linearize the bond parameters in the forcefield for more robust optimization"""
    ff_copy = copy.deepcopy(ff)
    ff_copy.potentials = []
    for potential in ff.potentials:
        if potential.type in {"ProperTorsions"}:
            new_potential = copy.deepcopy(potential)
            new_potential.type = "LinearProperTorsions"
            new_potential.fn = "k1*(1+cos(periodicity*theta-phase1))+k2*(1+cos(periodicity*theta-phase2))"
            new_potential.parameter_cols = ("k1", "k2", "periodicity", "phase1", "phase2", "idivf")
            new_params = []
            for param in potential.parameters:
                k            = param[0].item()
                periodicity  = param[1].item()
                phase        = param[2].item()
                idivf        = param[3].item()
                dt = param.dtype
                k1 = abs(k * 0.5 * (1 + math.cos(phase)))
                k2 = abs(k * 0.5 * (1 - math.cos(phase)))
                new_params.append([k1,k2,periodicity,0.0,math.pi,idivf])
            new_potential.parameters = torch.tensor(new_params,dtype=dt)
            new_potential.parameter_units = (_KCAL_PER_MOL,_KCAL_PER_MOL,_UNITLESS,_RADIANS,_RADIANS,_UNITLESS)
            ff_copy.potentials.append(new_potential)
        elif potential.type in {"ImproperTorsions"}:
            new_potential = copy.deepcopy(potential)
            new_potential.type = "LinearImproperTorsions"
            new_potential.fn = "k1*(1+cos(periodicity*theta-phase1))+k2*(1+cos(periodicity*theta-phase2))"
            new_potential.parameter_cols = ("k1", "k2", "periodicity", "phase1", "phase2", "idivf")
            new_params = []
            for param in potential.parameters:
                k            = param[0].item()
                periodicity  = param[1].item()
                phase        = param[2].item()
                idivf        = param[3].item()
                dt = param.dtype
                k1 = abs(k * 0.5 * (1 + math.cos(phase)))
                k2 = abs(k * 0.5 * (1 - math.cos(phase)))
                new_params.append([k1,k2,periodicity,0.0,math.pi,idivf])
            new_potential.parameters = torch.tensor(new_params,dtype=dt)
            new_potential.parameter_units = (_KCAL_PER_MOL,_KCAL_PER_MOL,_UNITLESS,_RADIANS,_RADIANS,_UNITLESS)
            ff_copy.potentials.append(new_potential)
        else:
            ff_copy.potentials.append(potential)
    return ff_copy

def linearize_topologies_1(topology: smee.TensorTopology) -> smee.TensorTopology:         
    """Linearize the param section of the topology object"""
    topology.parameters["LBonds"]                 = topology.parameters["Bonds"]
    topology.parameters["LAngles"]                = topology.parameters["Angles"]
    topology.parameters["LinearProperTorsions"]   = topology.parameters["ProperTorsions"]
    return topology

def linearize_topologies_2(topology: smee.TensorTopology) -> smee.TensorTopology:         
    """Linearize the param section of the topology object"""
    topology.parameters["LBonds"]                 = topology.parameters["Bonds"]
    topology.parameters["LAngles"]                = topology.parameters["Angles"]
    topology.parameters["LinearProperTorsions"]   = topology.parameters["ProperTorsions"]
    topology.parameters["LinearImproperTorsions"] = topology.parameters["ImproperTorsions"]
    return topology
       
def parameter_builder(params: str, ff: smee.TensorForceField) -> TrainableParameters:
    """Generate a parameter set to train based on user input"""
    if params == "BATI":
        return TrainableParameters(
            ff,
            {
                "Bonds": ParameterConfig(
                    cols=["k", "length"],
                    scales={"k": 1.0, "length": 1.0},
                    constraints={"k": (0.0, None), "length": (0.0, None)},
                ),
                "Angles": ParameterConfig(
                    cols=["k", "angle"],
                    scales={"k": 1.0, "angle": 1.0},
                    constraints={"k": (0.0, None), "angle": (0.0, math.pi)},
                ),
                "ProperTorsions": ParameterConfig(
                    cols=["k"],
                    scales={"k": 100.0, },
                    constraints={"k": (None, None)},
                ),
                "ImproperTorsions": ParameterConfig(
                    cols=["k"],
                    scales={"k": 100.0},
                    constraints={"k": (None, None)},
                ),
            }
        )
    elif params == "LBATI":
        return TrainableParameters(
            linearize_angles(linearize_bonds(ff)),
            {
                "LBonds": ParameterConfig(
                    cols=["k1", "k2"],
                    scales={"k1": 1.0, "k2": 1.0},
                    constraints={"k1": (None, None), "k2": (None, None)},
                ),
                "LAngles": ParameterConfig(
                    cols=["k1", "k2"],
                    scales={"k1": 10.0, "k2": 10.0},
                    constraints={"k1": (None, None), "k2": (None, None)},
                ),
                "ProperTorsions": ParameterConfig(
                    cols=["k"],
                    scales={"k": 100.0},
                    constraints={"k": (None, None)},
                ),
                "ImproperTorsions": ParameterConfig(
                    cols=["k"],
                    scales={"k": 100.0},
                    constraints={"k": (None, None)},
                ),
            }
        )
    elif params == "LLBATI":
        return TrainableParameters(
            linearize_torsions_2(linearize_angles(linearize_bonds(ff))),
            {
                "LBonds": ParameterConfig(
                    cols=["k1", "k2"],
                    scales={"k1": 1.0, "k2": 1.0},
                    constraints={"k1": (None, None), "k2": (None, None)},
                ),
                "LAngles": ParameterConfig(
                    cols=["k1", "k2"],
                    scales={"k1": 10.0, "k2": 10.0},
                    constraints={"k1": (None, None), "k2": (None, None)},
                ),
                "LinearProperTorsions": ParameterConfig(
                    cols=["k1", "k2"],
                    scales={"k1": 100.0, "k2": 100.0},
                    constraints={"k1": (0, None),"k2": (0, None)},
                ),
                "LinearImproperTorsions": ParameterConfig(
                    cols=["k1", "k2"],
                    scales={"k1": 100.0, "k2": 100.0},
                    constraints={"k1": (0, None),"k2": (0, None)},
                ),
            }
        )
    elif params == "BAT":
        return TrainableParameters(
            ff,
            {
                "Bonds": ParameterConfig(
                    cols=["k", "length"],
                    scales={"k": 10.0, "length": 1.0},
                    constraints={"k": (0.0, None), "length": (0.0, None)},
                ),
                "Angles": ParameterConfig(
                    cols=["k", "angle"],
                    scales={"k": 10.0, "angle": 1.0},
                    constraints={"k": (0.0, None), "angle": (0.0, math.pi)},
                ),
                "ProperTorsions": ParameterConfig(
                    cols=["k"],
                    scales={"k": 100.0},
                    constraints={"k": (None, None),},
                ),
            }
        )
    elif params == "LBAT":
        return TrainableParameters(
            linearize_angles(linearize_bonds(ff)),
            {
                "LBonds": ParameterConfig(
                    cols=["k1", "k2"],
                    scales={"k1": 1.0, "k2": 1.0},
                    constraints={"k1": (None, None), "k2": (None, None)},
                ),
                "LAngles": ParameterConfig(
                    cols=["k1", "k2"],
                    scales={"k1": 10.0, "k2": 10.0},
                    constraints={"k1": (None, None), "k2": (None, None)},
                ),
                "ProperTorsions": ParameterConfig(
                    cols=["k"],
                    scales={"k": 100.0},
                    constraints={"k": (None, None)},
                ),
            }
        )
    elif params == "LLBAT":
        return TrainableParameters(
            linearize_torsions_1(linearize_angles(linearize_bonds(ff))),
            {
                "LBonds": ParameterConfig(
                    cols=["k1", "k2"],
                    scales={"k1": 1.0, "k2": 1.0},
                    constraints={"k1": (None, None), "k2": (None, None)},
                ),
                "LAngles": ParameterConfig(
                    cols=["k1", "k2"],
                    scales={"k1": 10.0, "k2": 10.0},
                    constraints={"k1": (None, None), "k2": (None, None)},
                ),
                "LinearProperTorsions": ParameterConfig(
                    cols=["k1", "k2"],
                    scales={"k1": 100.0, "k2": 100.0},
                    constraints={"k1": (0, None),"k2": (0, None)},
                ),
            }
        )
    elif params == "BA":
        return TrainableParameters(
            ff,
            {
                "Bonds": ParameterConfig(
                    cols=["k", "length"],
                    scales={"k": 10.0, "length": 1.0},
                    constraints={"k": (0.0, None), "length": (0.0, None)},
                ),
                "Angles": ParameterConfig(
                    cols=["k", "angle"],
                    scales={"k": 10.0, "angle": 1.0},
                    constraints={"k": (0.0, None), "angle": (0.0, math.pi)},
                ),
            }
        )
    elif params == "LBA":
        return TrainableParameters(
            linearize_angles(linearize_bonds(ff)),
            {
                "LBonds": ParameterConfig(
                    cols=["k1", "k2"],
                    scales={"k1": 1.0, "k2": 1.0},
                    constraints={"k1": (None, None), "k2": (None, None)},
                ),
                "LAngles": ParameterConfig(
                    cols=["k1", "k2"],
                    scales={"k1": 1.0, "k2": 1.0},
                    constraints={"k1": (None, None), "k2": (None, None)},
                ),
            }
        )
    elif params == "T":
        return TrainableParameters(
            ff,
            {
                "ProperTorsions": ParameterConfig(
                    cols=["k"],
                    scales={"k": 100.0},
                    constraints={"k": (None, None),},
                ),
            }
        )
            
#            "ImproperTorsions": ParameterConfig(
#                cols=["k", "phase", "angle"],
#                scales={"k": 1.0 / 100.0, "phase": 1.0, "angle": 1.0},
#                constraints={"k": (None, None), "phase": (0.0, math.pi), "angle": (0.0, math.pi)},
#            ),
        
