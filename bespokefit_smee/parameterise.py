"""Functionality for generating the initial parameterisation."""

import collections
import copy
import math

import loguru
import openff.interchange
import openff.toolkit
import smee
import smee.converters
import torch
from descent.train import ParameterConfig, Trainable
from openff.units import Quantity
from openff.units import unit as off_unit
from rdkit import Chem

from .msm import apply_msm
from .settings import ParameterisationSettings
from .utils.typing import TorchDevice

logger = loguru.logger

_UNITLESS = off_unit.dimensionless
_ANGSTROM = off_unit.angstrom
_RADIANS = off_unit.radians
_KCAL_PER_MOL = off_unit.kilocalories_per_mole
_KCAL_PER_MOL_ANGSQ = off_unit.kilocalories_per_mole / off_unit.angstrom**2
_KCAL_PER_MOL_RADSQ = off_unit.kilocalories_per_mole / off_unit.radians**2


def _reflect_angle(angle: float) -> float:
    """Reflect an angle (in radians) to be in the range [0, pi)."""
    return math.pi - abs((angle % (2 * math.pi)) - math.pi)


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
            parameters_by_smarts: dict[str, dict[int | None, torch.Tensor]] = (
                collections.defaultdict(dict)
            )
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
                parameter_dict: dict[str, str | Quantity] = {
                    "smirks": smarts,
                    "id": parameter_id,
                }
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
        elif potential.type == "LinearBonds":
            assert potential.attribute_cols is None
            parameters_by_smarts = collections.defaultdict(dict)
            new_params = []
            for param in potential.parameters:
                k1 = param[0].item()
                k2 = param[1].item()
                b1 = param[2].item()
                b2 = param[3].item()
                k = k1 + k2
                b = (k1 * b1 + k2 * b2) / k
                dt = param.dtype
                new_params.append([k, b])
            reconstructed_param = torch.tensor(new_params, dtype=dt)
            reconstructed_units = (_KCAL_PER_MOL_ANGSQ, _ANGSTROM)
            reconstructed_cols = ("k", "length")
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
                        (col if mult is None else f"{col}{mult + 1}"): float(
                            parameter[col_idx]
                        )
                        * reconstructed_units[col_idx]
                        for mult, parameter in parameters_by_mult.items()
                        for col_idx, col in enumerate(reconstructed_cols)
                    }
                )
                handler.add_parameter(parameter_dict)
        elif potential.type == "LinearAngles":
            assert potential.attribute_cols is None
            parameters_by_smarts = collections.defaultdict(dict)
            new_params = []
            for param in potential.parameters:
                k1 = param[0].item()
                k2 = param[1].item()
                a1 = param[2].item()
                a2 = param[3].item()
                k = k1 + k2
                # Set k and angle to 0 if very close
                a = (k1 * a1 + k2 * a2) / k
                # Ensure that the angle is in the range [0, pi)
                a = _reflect_angle(a)
                dt = param.dtype
                new_params.append([k, a])
            reconstructed_param = torch.tensor(new_params, dtype=dt)
            reconstructed_units = (_KCAL_PER_MOL_RADSQ, _RADIANS)
            reconstructed_cols = ("k", "angle")
            for parameter, parameter_key in zip(
                reconstructed_param, potential.parameter_keys, strict=True
            ):
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
                        (col if mult is None else f"{col}{mult + 1}"): float(
                            parameter[col_idx]
                        )
                        * reconstructed_units[col_idx]
                        for mult, parameter in parameters_by_mult.items()
                        for col_idx, col in enumerate(reconstructed_cols)
                    }
                )
                handler.add_parameter(parameter_dict)
        elif potential.type == "LinearProperTorsions":
            assert potential.attribute_cols is None
            parameters_by_smarts = collections.defaultdict(dict)
            new_params = []
            for param in potential.parameters:
                k1 = param[0].item()
                k2 = param[1].item()
                periodicity = param[2].item()
                # Params 3 and 4 are phase1 and phase2
                idivf = param[5].item()
                k = k1 + k2
                if k == 0.0:
                    phase = 0.0
                else:
                    phase = math.acos((k1 - k2) / k)
                dt = param.dtype
                new_params.append([k, periodicity, phase, idivf])
            reconstructed_param = torch.tensor(new_params, dtype=dt)
            reconstructed_torsion_units = (
                _KCAL_PER_MOL,
                _UNITLESS,
                _RADIANS,
                _UNITLESS,
            )
            reconstructed_torsion_cols = ("k", "periodicity", "phase", "idivf")
            for parameter, parameter_key in zip(
                reconstructed_param, potential.parameter_keys, strict=True
            ):
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
                        (col if mult is None else f"{col}{mult + 1}"): float(
                            parameter[col_idx]
                        )
                        * reconstructed_torsion_units[col_idx]
                        for mult, parameter in parameters_by_mult.items()
                        for col_idx, col in enumerate(reconstructed_torsion_cols)
                    }
                )
                handler.add_parameter(parameter_dict)
        elif potential.type == "LinearImproperTorsions":
            assert potential.attribute_cols is None
            parameters_by_smarts = collections.defaultdict(dict)
            new_params = []
            for param in potential.parameters:
                k1 = param[0].item()
                k2 = param[1].item()
                periodicity = param[2].item()
                # Params 3 and 4 are phase1 and phase2
                idivf = param[5].item()
                k = k1 + k2
                if k == 0.0:
                    phase = 0.0
                else:
                    phase = math.acos((k1 - k2) / k)
                #                    phase = math.acos((k1 * math.cos(phase1) + k2 * math.cos(phase2))/k)
                dt = param.dtype
                new_params.append([k, periodicity, phase, idivf])
            reconstructed_param = torch.tensor(new_params, dtype=dt)
            reconstructed_torsion_units = (
                _KCAL_PER_MOL,
                _UNITLESS,
                _RADIANS,
                _UNITLESS,
            )
            reconstructed_torsion_cols = ("k", "periodicity", "phase", "idivf")
            for parameter, parameter_key in zip(
                reconstructed_param, potential.parameter_keys, strict=True
            ):
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
                        (col if mult is None else f"{col}{mult + 1}"): float(
                            parameter[col_idx]
                        )
                        * reconstructed_torsion_units[col_idx]
                        for mult, parameter in parameters_by_mult.items()
                        for col_idx, col in enumerate(reconstructed_torsion_cols)
                    }
                )
                handler.add_parameter(parameter_dict)

    return ff_smirnoff


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
    parameter_map: smee.ValenceParameterMap,
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

    sorted_ids_to_parameter_idxs = {
        particle_ids: sorted(parameter_idxs)
        for particle_ids, parameter_idxs in ids_to_parameter_idxs.items()
    }

    parameter_ids = [
        (particle_ids, parameter_idx)
        for particle_ids, parameter_idxs in sorted_ids_to_parameter_idxs.items()
        for parameter_idx in parameter_idxs
    ]
    potential.parameters = potential.parameters[
        [parameter_idx for _, parameter_idx in parameter_ids]
    ]
    potential.parameter_keys = [
        openff.interchange.models.PotentialKey(
            id=ids_to_smarts[particle_ids],
            mult=(
                sorted_ids_to_parameter_idxs[particle_ids].index(parameter_idx)
                if is_indexed
                else None
            ),
            associated_handler=potential.type,
            bond_order=None,
            virtual_site_type=None,
            cosmetic_attributes={},
        )
        for particle_ids, parameter_idx in parameter_ids
    ]

    assignment_matrix = smee.utils.zeros_like(
        (len(parameter_map.particle_idxs), len(potential.parameters)),
        parameter_map.assignment_matrix,
    )

    particle_idxs_updated: list[tuple[int, ...]] = []

    for particle_ids, particle_idxs in ids_to_particle_idxs.items():
        for particle_idx in particle_idxs:
            for parameter_idx in sorted_ids_to_parameter_idxs[particle_ids]:
                j = parameter_ids.index((particle_ids, parameter_idx))

                assignment_matrix[len(particle_idxs_updated), j] = 1
                particle_idxs_updated.append(particle_idx)

    parameter_map.particle_idxs = smee.utils.tensor_like(
        particle_idxs_updated, parameter_map.particle_idxs
    )
    parameter_map.assignment_matrix = assignment_matrix.to_sparse()


# TODO: Break this up into smaller functions
def parameterise(
    settings: ParameterisationSettings,
    device: TorchDevice = "cuda",
) -> tuple[
    openff.toolkit.Molecule,
    openff.toolkit.ForceField,
    smee.TensorTopology,
    smee.TensorForceField,
    Trainable,
]:
    """Prepare a Trainable object that contains a force field with
    unique parameters for each topologically symmetric term of a molecule.

    Parameters
    ----------
    settings: ParameterisationSettings
        The settings for the parameterisation.

    device: TorchDevice, default "cuda"
        The device to use for the force field and topology.

    Returns
    -------
    mol: openff.toolkit.Molecule
        The molecule that has been parameterised.
    off_ff: openff.toolkit.ForceField
        The original force field, used as a base for the bespoke force field.
    tensor_top: smee.TensorTopology
        The topology of the molecule.
    tensor_ff: smee.TensorForceField
        The force field with unique parameters for each topologically symmetric term.
    trainable: descent.train.Trainable
        The trainable object that contains the force field and parameter configuration.
    """
    mol = openff.toolkit.Molecule.from_smiles(
        settings.smiles, allow_undefined_stereo=True, hydrogens_are_explicit=False
    )
    off_ff = openff.toolkit.ForceField(settings.initial_force_field)

    if "[#1:1]-[*:2]" in off_ff["Constraints"].parameters:
        logger.warning(
            "The force field contains a constraint for [#1:1]-[*:2] which is not supported. "
            "Removing this constraint."
        )
        del off_ff["Constraints"].parameters["[#1:1]-[*:2]"]

    if settings.expand_torsions:
        off_ff = _expand_torsions(off_ff)

    force_field, [topology] = smee.converters.convert_interchange(
        openff.interchange.Interchange.from_smirnoff(off_ff, mol.to_topology())
    )

    # Move the force field and topology to the requested device
    force_field = force_field.to(device)
    topology = topology.to(device)

    symmetries = list(Chem.CanonicalRankAtoms(mol.to_rdkit(), breakTies=False))
    if topology.n_v_sites != 0:
        raise NotImplementedError("virtual sites are not supported yet.")
    for potential in force_field.potentials:
        parameter_map = topology.parameters[potential.type]
        if isinstance(parameter_map, smee.NonbondedParameterMap):
            continue
        # TODO: Check Tom's comment below
        _prepare_potential(
            mol, symmetries, potential, parameter_map
        )  ### ??? is it re-ordering the atoms and bonds?

    if settings.msm_settings is not None:
        raise NotImplementedError("MSM is not supported yet.")

        force_field = apply_msm(
            mol=mol,
            off_ff=off_ff,
            tensor_top=topology,
            tensor_ff=force_field,
            device=device,
            settings=settings.msm_settings,
        )

    # Parameter scales obtained from trained force field - but only for linearised bonds and
    # angles and unlinearised harmonics.
    if settings.linear_harmonics:
        topology.parameters["LinearBonds"] = copy.deepcopy(topology.parameters["Bonds"])
        topology.parameters["LinearAngles"] = copy.deepcopy(
            topology.parameters["Angles"]
        )
        force_field = linearize_harmonics(force_field, device)
        parameter_list = {
            "LinearBonds": ParameterConfig(
                cols=["k1", "k2"],
                scales={"k1": 0.0024, "k2": 0.0024},
                limits={"k1": (None, None), "k2": (None, None)},
            ),
            "LinearAngles": ParameterConfig(
                cols=["k1", "k2"],
                scales={"k1": 0.0207, "k2": 0.0207},
                limits={"k1": (None, None), "k2": (None, None)},
            ),
        }
    else:
        parameter_list = {
            "Bonds": ParameterConfig(
                cols=["k", "length"],
                scales={"k": 1.0, "length": 1.0},
                limits={"k": (0.0, None), "length": (0.0, None)},
            ),
            "Angles": ParameterConfig(
                cols=["k", "angle"],
                scales={"k": 1.0, "angle": 1.0},
                limits={"k": (0.0, None), "angle": (0.0, math.pi)},
            ),
        }
    for potential in force_field.potentials:
        if potential.type == "ProperTorsions":
            if settings.linear_torsions:
                topology.parameters["LinearProperTorsions"] = copy.deepcopy(
                    topology.parameters["ProperTorsions"]
                )
                force_field = linearize_propertorsions(force_field, device)
                parameter_list.update(
                    {
                        "LinearProperTorsions": ParameterConfig(
                            cols=["k1", "k2"],
                            scales={"k1": 100.0, "k2": 100.0},
                            limits={"k1": (0, None), "k2": (0, None)},
                        )
                    }
                )
            else:
                parameter_list.update(
                    {
                        "ProperTorsions": ParameterConfig(
                            cols=["k"],
                            scales={
                                "k": 0.3252,
                            },
                            limits={"k": (None, None)},
                        ),
                    }
                )
        elif potential.type == "ImproperTorsions":
            if settings.linear_torsions:
                topology.parameters["LinearImproperTorsions"] = copy.deepcopy(
                    topology.parameters["ImproperTorsions"]
                )
                force_field = linearize_impropertorsions(force_field, device)
                parameter_list.update(
                    {
                        "LinearImproperTorsions": ParameterConfig(
                            cols=["k1", "k2"],
                            scales={"k1": 100.0, "k2": 100.0},
                            limits={"k1": (0, None), "k2": (0, None)},
                        ),
                    }
                )
            else:
                parameter_list.update(
                    {
                        "ImproperTorsions": ParameterConfig(
                            cols=["k"],
                            scales={
                                "k": 0.1647,
                            },
                            limits={"k": (None, None)},
                        ),
                    }
                )

    return (
        copy.deepcopy(mol),
        copy.deepcopy(off_ff),
        topology,
        force_field,
        Trainable(force_field, parameter_list, {}),
    )


def _expand_torsions(ff: openff.toolkit.ForceField) -> openff.toolkit.ForceField:
    """Expand the torsion potential to include K0-4 for proper torsions"""
    ff_copy = copy.deepcopy(ff)
    torsion_handler = ff_copy.get_parameter_handler("ProperTorsions")
    for parameter in torsion_handler:
        # set the defaults
        parameter.idivf = [1.0] * 4
        default_k = [0 * _KCAL_PER_MOL] * 4
        default_phase = [0 * _RADIANS] * 4
        default_p = [1, 2, 3, 4]
        # update the existing k values for the correct phase and p
        for i, p in enumerate(parameter.periodicity):
            try:
                default_k[p - 1] = parameter.k[i]
                default_phase[p - 1] = parameter.phase[i]
            except IndexError:
                continue
        # update with new parameters
        parameter.k = default_k
        parameter.phase = default_phase
        parameter.periodicity = default_p
    return ff_copy


def linearize_harmonics(
    ff: smee.TensorForceField, device_type: str
) -> smee.TensorForceField:
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
                k = param[0].item()
                b = param[1].item()
                dt = param.dtype
                b1 = b * 0.9
                b2 = b * 1.1
                d = b2 - b1
                k1 = k * (b2 - b) / d
                k2 = k * (b - b1) / d
                new_params.append([k1, k2, b1, b2])
            new_potential.parameters = torch.tensor(
                new_params, dtype=dt, requires_grad=False, device=device_type
            )
            new_potential.parameter_units = (
                _KCAL_PER_MOL_ANGSQ,
                _KCAL_PER_MOL_ANGSQ,
                _ANGSTROM,
                _ANGSTROM,
            )
            ff_copy.potentials.append(new_potential)
        elif potential.type in {"Angles"}:
            new_potential = copy.deepcopy(potential)
            new_potential.type = "LinearAngles"
            new_potential.fn = "(k1+k2)/2*(r-(k1*angle1+k2*angle2)/(k1+k2))**2"
            new_potential.parameter_cols = ("k1", "k2", "angle1", "angle2")
            new_params = []
            for param in potential.parameters:
                k = param[0].item()
                a = param[1].item()
                dt = param.dtype
                a1 = a * 0.9
                a2 = a * 1.1
                d = a2 - a1
                k1 = k * (a2 - a) / d
                k2 = k * (a - a1) / d
                new_params.append([k1, k2, a1, a2])
            new_potential.parameters = torch.tensor(
                new_params, dtype=dt, requires_grad=False, device=device_type
            )
            new_potential.parameter_units = (
                _KCAL_PER_MOL_RADSQ,
                _KCAL_PER_MOL_RADSQ,
                _RADIANS,
                _RADIANS,
            )
            ff_copy.potentials.append(new_potential)
        else:
            ff_copy.potentials.append(potential)
    return ff_copy


def linearize_propertorsions(
    ff: smee.TensorForceField, device_type: str
) -> smee.TensorForceField:
    """Linearize the proper torsion parameters in the forcefield for more robust optimization"""
    ff_copy = copy.deepcopy(ff)
    ff_copy.potentials = []
    for potential in ff.potentials:
        if potential.type in {"ProperTorsions"}:
            new_potential = copy.deepcopy(potential)
            new_potential.type = "LinearProperTorsions"
            new_potential.fn = (
                "(k1+k2)*(1+cos(periodicity*theta-acos((k1-k2)/(k1+k2))))"
            )
            new_potential.parameter_cols = (
                "k1",
                "k2",
                "periodicity",
                "phase1",
                "phase2",
                "idivf",
            )
            new_params = []
            for param in potential.parameters:
                k = param[0].item()
                periodicity = param[1].item()
                phase = param[2].item()
                idivf = param[3].item()
                dt = param.dtype
                k1 = abs(k * 0.5 * (1 + math.cos(phase)))
                k2 = abs(k * 0.5 * (1 - math.cos(phase)))
                new_params.append([k1, k2, periodicity, 0.0, math.pi, idivf])
            new_potential.parameters = torch.tensor(
                new_params, dtype=dt, requires_grad=True, device=device_type
            )
            new_potential.parameter_units = (
                _KCAL_PER_MOL,
                _KCAL_PER_MOL,
                _UNITLESS,
                _RADIANS,
                _RADIANS,
                _UNITLESS,
            )
            ff_copy.potentials.append(new_potential)
        else:
            ff_copy.potentials.append(potential)
    return ff_copy


def linearize_impropertorsions(
    ff: smee.TensorForceField, device_type: str
) -> smee.TensorForceField:
    """Linearize the improper torsion parameters in the forcefield for more robust optimization"""
    ff_copy = copy.deepcopy(ff)
    ff_copy.potentials = []
    for potential in ff.potentials:
        if potential.type in {"ImproperTorsions"}:
            new_potential = copy.deepcopy(potential)
            new_potential.type = "LinearImproperTorsions"
            new_potential.fn = (
                "(k1+k2)*(1+cos(periodicity*theta-acos((k1-k2)/(k1+k2))))"
            )
            new_potential.parameter_cols = (
                "k1",
                "k2",
                "periodicity",
                "phase1",
                "phase2",
                "idivf",
            )
            new_params = []
            for param in potential.parameters:
                k = param[0].item()
                periodicity = param[1].item()
                phase = param[2].item()
                idivf = param[3].item()
                dt = param.dtype
                k1 = abs(k * 0.5 * (1 + math.cos(phase)))
                k2 = abs(k * 0.5 * (1 - math.cos(phase)))
                new_params.append([k1, k2, periodicity, 0.0, math.pi, idivf])
            new_potential.parameters = torch.tensor(
                new_params, dtype=dt, device=device_type
            )
            new_potential.parameter_units = (
                _KCAL_PER_MOL,
                _KCAL_PER_MOL,
                _UNITLESS,
                _RADIANS,
                _RADIANS,
                _UNITLESS,
            )
            ff_copy.potentials.append(new_potential)
        else:
            ff_copy.potentials.append(potential)
    return ff_copy
