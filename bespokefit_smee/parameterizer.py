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
from openff.units import unit as off_unit
import openmm.unit
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

_UNITLESS           = off_unit.dimensionless
_ANGSTROM           = off_unit.angstrom
_RADIANS            = off_unit.radians
_KCAL_PER_MOL       = off_unit.kilocalories_per_mole
_KCAL_PER_MOL_ANGSQ = off_unit.kilocalories_per_mole / off_unit.angstrom ** 2
_KCAL_PER_MOL_RADSQ = off_unit.kilocalories_per_mole / off_unit.radians ** 2

_OMM_KELVIN            = openmm.unit.kelvin
_OMM_PS                = openmm.unit.picosecond 
_OMM_NM                = openmm.unit.nanometer
_OMM_ANGS              = openmm.unit.angstrom
_OMM_KCAL_PER_MOL      = openmm.unit.kilocalorie_per_mole
_OMM_KCAL_PER_MOL_ANGS = openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom

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
        elif potential.type=="LinearBonds":
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
        elif potential.type=="LinearAngles":
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

def build_parameters(
    mol: openff.toolkit.Molecule,
    off: openff.toolkit.ForceField,
    ML_path: str,
    linear_harmonics: bool,
    linear_torsions: bool,
    modSem: bool,
    modSem_finite_step: float,
    modSem_vib_scaling: float,
    modSem_tolerance: float
) -> tuple[smee.TensorForceField, TrainableParameters, smee.TensorTopology]:
    """Prepare a Trainable object that contains  a force field with 
    unique parameters for each topologically symmetric term of a molecule.
    Args:
        mol: The molecule to prepare bespoke parameters for.
        off: The base force field to copy the parameters from.
        ML_path: Path to the MLMD potential used to evalutate the hessian - if used
        linear_harmonics: boolean indicating whether to use linearized harmonic potentials
        linear_torsions: boolean indicating whether to use linearized torsion potentials
        modSem: boolean indicating whether to use the moedified Seminario method to initialize the force-field
        modSem_finite_step: finite step used in evaluating the hessian - if used
        modSem_vib_scaling: scaling parameter for the modSem parameters
        modSem_tolerance: Tolerance for the geometric minimization before the hessian evaluation - if used

    Returns:
        The prepared Traninable object with a smee force_field and topology ready for fitting.
    """
    force_field, [topology] = smee.converters.convert_interchange(openff.interchange.Interchange.from_smirnoff(expand_torsions(off),mol.to_topology()))
    topology.constraints = None
    symmetries = list(Chem.CanonicalRankAtoms(mol.to_rdkit(), breakTies=False))
    if topology.n_v_sites != 0:
        raise NotImplementedError("virtual sites are not supported yet.")
    for potential in force_field.potentials:
        parameter_map = topology.parameters[potential.type]
        if isinstance(parameter_map, smee.NonbondedParameterMap):
            continue
        _prepare_potential(mol, symmetries, potential, parameter_map) ### ??? is it re-ordering the atoms and bonds?
    if modSem:
        force_field = modSeminario(mol,topology,off,ML_path,force_field,modSem_finite_step,modSem_vib_scaling,modSem_tolerance)
    improper_torsion_flag = False
    if(linear_harmonics):
        topology.parameters["LinearBonds"]  = copy.deepcopy(topology.parameters["Bonds"])
        topology.parameters["LinearAngles"] = copy.deepcopy(topology.parameters["Angles"])
        force_field = linearize_harmonics(force_field)
        parameter_list = {
            "LinearBonds": ParameterConfig(
                cols=["k1", "k2"],
                scales={"k1": 1.0, "k2": 1.0},
                constraints={"k1": (None, None), "k2": (None, None)},
            ),
            "LinearAngles": ParameterConfig(
                cols=["k1", "k2"],
                scales={"k1": 10.0, "k2": 10.0},
                constraints={"k1": (None, None), "k2": (None, None)},
            )
        }
    else:
        parameter_list = {
            "Bonds": ParameterConfig(
                cols=["k", "length"],
                scales={"k": 1.0, "length": 1.0},
                constraints={"k": (0.0, None), "length": (0.0, None)},
            ),
            "Angles": ParameterConfig(
                cols=["k", "angle"],
                scales={"k": 1.0, "angle": 1.0},
                constraints={"k": (0.0, None), "angle": (0.0, math.pi)},
            )
        }
    for potential in force_field.potentials:
        if(potential.type == "ProperTorsions"):
            if(linear_torsions): 
                topology.parameters["LinearProperTorsions"] = copy.deepcopy(topology.parameters["ProperTorsions"])
                force_field = linearize_propertorsions(force_field)
                parameter_list.update(
                    {
                        "LinearProperTorsions": ParameterConfig(
                            cols=["k1", "k2"],
                            scales={"k1": 100.0, "k2": 100.0},
                            constraints={"k1": (0, None),"k2": (0, None)},
                        )
                    }
                )  
            else:
                parameter_list.update(
                    {
                        "ProperTorsions": ParameterConfig(
                            cols=["k"],
                            scales={"k": 100.0, },
                            constraints={"k": (None, None)},
                        ),
                    }
                )
        elif(potential.type == "ImproperTorsions"):
            improper_torsion_flag = True
            if(linear_torsions):
                topology.parameters["LinearImproperTorsions"] = copy.deepcopy(topology.parameters["ImproperTorsions"])
                force_field = linearize_impropertorsions(force_field)
                parameter_list.update(
                    {
                        "LinearImproperTorsions": ParameterConfig(
                            cols=["k1", "k2"],
                            scales={"k1": 100.0, "k2": 100.0},
                            constraints={"k1": (0, None),"k2": (0, None)},
                        ),
                    }
                )  
            else:
                parameter_list.update(
                    {
                        "ImproperTorsions": ParameterConfig(
                            cols=["k"],
                            scales={"k": 100.0, },
                            constraints={"k": (None, None)},
                        ),
                    }
                )
    return copy.deepcopy(force_field), TrainableParameters(force_field,parameter_list), topology

def expand_torsions(ff: openff.toolkit.ForceField) -> openff.toolkit.ForceField:
    """Expand the torsion potential to include K0-4 for proper torsions"""
    ff_copy         = copy.deepcopy(ff)
    torsion_handler = ff_copy.get_parameter_handler("ProperTorsions")
    for parameter in torsion_handler:
        # set the defaults
        parameter.idivf = [1.0] * 4
        default_k       = [0 * _KCAL_PER_MOL] * 4
        default_phase   = [0 * _RADIANS] * 4
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
    top: smee.TensorTopology,
    off: openff.toolkit.ForceField,
    ML_path: str,
    sff: smee.TensorForceField,
    finite_step: float,
    vib_scaling: float,
    minimize_tol: float
) -> smee.TensorForceField:
    """Generate modified Seminario parameters for the bond and angle terms in the 
    force-field. see doi: 10.1021/acs.jctc.7b00785 
    """
    from openmm.app.simulation import Simulation
    from openmm import LangevinMiddleIntegrator
    import openmm.unit
    from openmmml import MLPotential
    from writers  import write_potential_comparison
#   set up an MD sim with the ML potential
    molecule        = copy.deepcopy(mol)
    molecule.generate_conformers(n_conformers=1)
    interchange     = openff.interchange.Interchange.from_smirnoff(off,openff.toolkit.Topology.from_molecules(molecule))
    integrator      = LangevinMiddleIntegrator(0*_OMM_KELVIN,1/_OMM_PS,0.01*_OMM_PS)
    potential       = MLPotential(ML_path)
    with open("/dev/null", 'w') as f:
        with redirect_stdout(f):
            system  = potential.createSystem(interchange.to_openmm_topology())
    simulation            = Simulation(interchange.topology, system, integrator)
#   calculate the ground-state geometry and energy
    interchange.positions = molecule.conformers[0]
    simulation.context.setPositions(interchange.positions.to_openmm())
    simulation.minimizeEnergy(maxIterations=0,tolerance=minimize_tol*_OMM_KCAL_PER_MOL_ANGS)
    position = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
    crd0     = position.value_in_unit(_OMM_NM).reshape(3 * molecule.n_atoms)
#   extract bond info from the smee tensor
    bonds_obj    = copy.deepcopy(top.parameters["Bonds"])
    n_bonds      = len(bonds_obj.assignment_matrix.indices()[0].detach().flatten().tolist())
    n_bond_types = max(bonds_obj.assignment_matrix.indices()[-1].detach().flatten().tolist()) + 1
    bond_types   = [[i for i, x in enumerate(bonds_obj.assignment_matrix.indices()[-1].tolist()) if x == j] for j in range(n_bond_types)]
    bond_indxs   = bonds_obj.particle_idxs.tolist()
#   extract angle info from the smee tensor
    angles_obj    = copy.deepcopy(top.parameters["Angles"])
    n_angles      = len(angles_obj.assignment_matrix.indices()[0].detach().flatten().tolist())
    n_angle_types = max(angles_obj.assignment_matrix.indices()[-1].detach().flatten().tolist()) + 1
    angle_types   = [[i for i, x in enumerate(angles_obj.assignment_matrix.indices()[-1].tolist()) if x == j] for j in range(n_angle_types)]
    angle_indxs   = angles_obj.particle_idxs.tolist()
#   calculate hessian elements with finite difference, ignoring the diagonal and all below
    hessian = np.zeros((3 * molecule.n_atoms, 3 * molecule.n_atoms))
    for i in tqdm(range(n_bonds),leave=False,colour='green',desc="Generating Hessian Fragments"):
        i1, i2  = bond_indxs[i][0] * 3, bond_indxs[i][1] * 3
        for j1 in range(i1,i1+3):
            crd = crd0
            crd[j1] += finite_step
            simulation.context.setPositions(crd.reshape(molecule.n_atoms,3)) # coords +
            f1      = simulation.context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(_OMM_KCAL_PER_MOL_ANGS)
            dEp     = -f1[i2 // 3]
            crd[j1] -= 2 * finite_step
            simulation.context.setPositions(crd.reshape(molecule.n_atoms,3)) # coords -
            f1      = simulation.context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(_OMM_KCAL_PER_MOL_ANGS)
            dEm     = -f1[i2 // 3]
            hessian[j1,range(i2,i2+3)] = (dEp - dEm) / (2 * finite_step)
#   calculate mod-seminario force constants along the bonds and group by bond-type, as given in the smee tensors
    bond_k, bond_l = [], []
    for j in range(n_bond_types):
        k_sum, l_sum = 0.0, 0.0
        for i in bond_types[j]:
            iA, iB = bond_indxs[i][0], bond_indxs[i][1]
            jA, jB = iA * 3, iB * 3
            b      = position.value_in_unit(_OMM_ANGS)[iA] - position.value_in_unit(_OMM_ANGS)[iB]
            l      = np.linalg.norm(b)
            k_sum += modSem_projection(-hessian[jA:jA+3,jB:jB+3],b / l)
            l_sum += l
        bond_k.append(k_sum * vib_scaling ** 2 * .1 / len(bond_types[j]))
        bond_l.append(l_sum / len(bond_types[j]))
#   calculate mod-seminario force constants along around the angles and group by angle-type, as given in the smee tensors
    angle_k, angle_t = [], []
    for j in range(n_angle_types):
        k_sum, t_sum = 0.0, 0.0
        for i in angle_types[j]:
            iA, iB, iC = angle_indxs[i][0], angle_indxs[i][1], angle_indxs[i][2]
            jA, jB, jC = iA * 3, iB * 3, iC * 3
            bAB = position.value_in_unit(_OMM_ANGS)[iA] - position.value_in_unit(_OMM_ANGS)[iB]
            bCB = position.value_in_unit(_OMM_ANGS)[iC] - position.value_in_unit(_OMM_ANGS)[iB]
            if iA > iB:
                HAB = -hessian[jB:jB+3,jA:jA+3]
            else:
                HAB = -hessian[jA:jA+3,jB:jB+3]
            if iC > iB:
                HCB = -hessian[jB:jB+3,jC:jC+3]
            else:
                HCB = -hessian[jC:jC+3,jB:jB+3]
            lAB, lCB   = np.linalg.norm(bAB), np.linalg.norm(bCB)
            uAB, uCB   = bAB / lAB, bCB / lCB
            uN         = unit_normal_vector(uAB,uCB)
            uPA, uPC   = unit_normal_vector(uN,uAB), unit_normal_vector(uCB,uN)
            kPA, kPC   = modSem_projection(HAB,uPA), modSem_projection(HCB,uPC)
            fixA, fixC = 0, 0
            NfA, NfC   = 0, 0
            for jj in range(n_angles):
                iiA, iiB, iiC = angle_indxs[jj][0], angle_indxs[jj][1], angle_indxs[jj][2]
                if iiB == iB & jj != i:
                    if iiA == iA:
                        bCBp  = position.value_in_unit(_OMM_ANGS)[iiC] - position.value_in_unit(_OMM_ANGS)[iiB]
                        uPAp  = unit_normal_vector(unit_normal_vector(uAB,bCBp / np.linalg.norm(bCBp)),uAB)
                        fixA += np.dot(uPA,uPAp) ** 2
                        NfA  += 1
                    elif iiC == iC:
                        bABp  = position.value_in_unit(_OMM_ANGS)[iiA] - position.value_in_unit(_OMM_ANGS)[iiB]
                        uPCp  = unit_normal_vector(unit_normal_vector(uCB,bABp / np.linalg.norm(bABp)),uCB)
                        fixC += np.dot(uPC,uPCp) ** 2
                        NfC  += 1
            if NfA > 0:
                fixA = fixA / NfA
            if NfC > 0:
                fixC = fixC / NfC
            k_sum   += 1 / (((1 + fixA) / (lAB ** 2 * kPA)) + ((1 + fixC) / (lCB ** 2 * kPC)))
            t_sum   += np.arccos(np.dot(uAB,uCB))
        angle_k.append(k_sum * vib_scaling ** 2 * .1 / len(angle_types[j]))
        angle_t.append(t_sum / len(angle_types[j]))
#   put the new constants into the force-field object and report!
    sff_out = copy.deepcopy(sff)
    sff_out.potentials_by_type["Bonds"].parameters  = torch.tensor([[bond_k[j], bond_l[j]] for j in range(n_bond_types)])
    sff_out.potentials_by_type["Angles"].parameters = torch.tensor([[angle_k[j], angle_t[j]] for j in range(n_angle_types)])
    print("Modified Seminario Summary:")
    write_potential_comparison(sff.potentials_by_type["Bonds"],sff_out.potentials_by_type["Bonds"])
    write_potential_comparison(sff.potentials_by_type["Angles"],sff_out.potentials_by_type["Angles"])
    return sff_out          

def unit_normal_vector(u1,u2):
    """ Return a unit vector perpendicular to the two input vectors """
    cross = np.cross(u1,u2)
    return cross / np.linalg.norm(cross)

def modSem_projection(parhess,unit_vector):
    """ Return a spring constant projected out of a partial hessian onto a unit vector """
    vals, vecs = np.linalg.eig(parhess)
    kab1 = sum(abs(np.dot(unit_vector,vecs[:,i])) * vals[i] for i in range(3)).real
    kba1 = sum(abs(np.dot(unit_vector[::-1],vecs[:,i])) * vals[i] for i in range(3)).real
    vals, vecs = np.linalg.eig(parhess.transpose())
    kab2 = sum(abs(np.dot(unit_vector,vecs[:,i])) * vals[i] for i in range(3)).real
    kba2 = sum(abs(np.dot(unit_vector[::-1],vecs[:,i])) * vals[i] for i in range(3)).real
    return 0.25 * (kab1 + kba1 + kab2 + kba2)

def linearize_harmonics(ff: smee.TensorForceField) -> smee.TensorForceField:         
    """Linearize the harmonic potential parameters in the forcefield for more robust optimization"""
    ff_copy = copy.deepcopy(ff)
    ff_copy.potentials = []
    for potential in ff.potentials:
        if potential.type in {"Bonds"}:
            new_potential = copy.deepcopy(potential)
            new_potential.type = "LinearBonds"
            new_potential.fn = "(k1+k2)/2*(r-(k1*length1+k2*length2)/(k1+k2))**2"
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
            new_potential.parameters = torch.tensor(new_params, dtype=dt, requires_grad=False)
            new_potential.parameter_units = (
                _KCAL_PER_MOL_ANGSQ, 
                _KCAL_PER_MOL_ANGSQ, 
                _ANGSTROM, 
                _ANGSTROM
            )
            ff_copy.potentials.append(new_potential)
        elif potential.type in {"Angles"}:
            new_potential = copy.deepcopy(potential)
            new_potential.type = "LinearAngles"
            new_potential.fn = "(k1+k2)/2*(r-(k1*angle1+k2*angle2)/(k1+k2))**2"
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
            new_potential.parameters = torch.tensor(new_params, dtype=dt, requires_grad=False)
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

def linearize_propertorsions(ff: smee.TensorForceField) -> smee.TensorForceField:         
    """Linearize the proper torsion parameters in the forcefield for more robust optimization"""
    ff_copy = copy.deepcopy(ff)
    ff_copy.potentials = []
    for potential in ff.potentials:
        if potential.type in {"ProperTorsions"}:
            new_potential = copy.deepcopy(potential)
            new_potential.type = "LinearProperTorsions"
            new_potential.fn = "(k1+k2)*(1+cos(periodicity*theta-acos((k1-k2)/(k1+k2))))"
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
            new_potential.parameters = torch.tensor(new_params, dtype=dt, requires_grad=True)
            new_potential.parameter_units = (_KCAL_PER_MOL,_KCAL_PER_MOL,_UNITLESS,_RADIANS,_RADIANS,_UNITLESS)
            ff_copy.potentials.append(new_potential)
        else:
            ff_copy.potentials.append(potential)
    return ff_copy

def linearize_impropertorsions(ff: smee.TensorForceField) -> smee.TensorForceField:         
    """Linearize the improper torsion parameters in the forcefield for more robust optimization"""
    ff_copy = copy.deepcopy(ff)
    ff_copy.potentials = []
    for potential in ff.potentials:
        if potential.type in {"ImproperTorsions"}:
            new_potential = copy.deepcopy(potential)
            new_potential.type = "LinearImproperTorsions"
            new_potential.fn = "(k1+k2)*(1+cos(periodicity*theta-acos((k1-k2)/(k1+k2))))"
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
