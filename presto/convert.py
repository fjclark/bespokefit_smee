"""Convert OpenFF ForceField <--> smee TensorForceField."""

import collections
import copy
import math
from collections.abc import Callable

import loguru
import openff.interchange
import openff.toolkit
import smee
import smee.converters
import torch
from openff.units import Quantity
from openff.units import unit as off_unit

from ._exceptions import MoleculeParameterisationError
from .create_types import (
    _add_parameter_with_overwrite,
    add_types_to_forcefield,
)
from .load_molecules import _molecule_description
from .msm import apply_msm_to_molecules
from .settings import ParamSettings
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
    """Convert a tensor force field that *contains bespoke valence parameters* to.

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
                _add_parameter_with_overwrite(handler, parameter_dict)

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
                _add_parameter_with_overwrite(handler, parameter_dict)
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
                _add_parameter_with_overwrite(handler, parameter_dict)

    return ff_smirnoff


def parameterise(
    settings: ParamSettings,
    device: TorchDevice = "cuda",
) -> tuple[
    list[openff.toolkit.Molecule],
    openff.toolkit.ForceField,
    list[smee.TensorTopology],
    smee.TensorForceField,
]:
    """Prepare a Trainable object that contains a force field with.

    unique parameters for each topologically symmetric term across multiple molecules.

    Parameters
    ----------
    settings: ParamSettings
        The settings for the parameterisation.

    device: TorchDevice, default "cuda"
        The device to use for the force field and topology.

    Returns:
    -------
    mols: list[openff.toolkit.Molecule]
        The molecules that have been parameterised.
    off_ff: openff.toolkit.ForceField
        The original force field, used as a base for the bespoke force field.
    tensor_tops: list[smee.TensorTopology]
        The topologies of the molecules.
    tensor_ff: smee.TensorForceField
        The force field with unique parameters for each topologically
        symmetric term.

    Raises:
    ------
    MoleculeParameterisationError
        If any molecule could not be parameterised. Every molecule is attempted, so a
        single error reports all of the failures.
    """
    # Create molecules from SMILES
    mols = settings.openff_molecules

    off_ff = openff.toolkit.ForceField(settings.initial_force_field)

    # First check required as Parsely does not contain constraints
    if "Constraints" in off_ff.registered_parameter_handlers:
        if "[#1:1]-[*:2]" in off_ff["Constraints"].parameters:
            logger.warning(
                "The force field contains a constraint for [#1:1]-[*:2] which "
                "is not supported. Removing this constraint."
            )
            del off_ff["Constraints"].parameters["[#1:1]-[*:2]"]

    if settings.expand_torsions:
        torsion_generation_settings = settings.type_generation_settings.get(
            "ProperTorsions"
        )
        excluded_smirks = (
            torsion_generation_settings.exclude
            if torsion_generation_settings is not None
            else None
        )
        off_ff = _expand_torsions(off_ff, excluded_smirks=excluded_smirks)

    # Add bespoke parameters to the force field (deduplicated across all molecules)
    bespoke_ff = add_types_to_forcefield(
        mols,
        off_ff,
        settings.type_generation_settings,
    )

    # Molecules which cannot be parameterised are collected rather than raised on
    # immediately, so that a single run reports every offending molecule and the user
    # only has to queue one more job. `bespoke_ff` is incomplete if anything failed, but
    # we always raise below before it is used.
    failures: dict[int, list[str]] = collections.defaultdict(list)

    if settings.msm_settings is not None:
        bespoke_ff, msm_failures = apply_msm_to_molecules(
            mols=mols,
            off_ff=bespoke_ff,
            settings=settings.msm_settings,
            device=torch.device(device),
        )
        for index, message in msm_failures.items():
            failures[index].append(message)

    # Create separate Interchange objects for each molecule. This is also where OpenFF
    # applies its configured charge model. Molecules which already failed the modified
    # Seminario method are still attempted here, so that a molecule with problems at
    # both stages reports both at once, but their interchange is discarded.
    interchanges = []
    for index, mol in enumerate(mols):
        try:
            interchange = openff.interchange.Interchange.from_smirnoff(
                bespoke_ff, mol.to_topology()
            )
        except Exception as exc:
            logger.opt(exception=True).error(
                f"Parameterisation failed for molecule {index}: {exc}"
            )
            failures[index].append(f"{type(exc).__name__}: {exc}")
            continue

        if index not in failures:
            interchanges.append(interchange)

    if failures:
        report = "\n".join(
            f"  - {_molecule_description(mols[index], index)}: "
            + "; ".join(failures[index])
            for index in sorted(failures)
        )
        raise MoleculeParameterisationError(
            f"Parameterisation failed for {len(failures)} of {len(mols)} molecules:\n"
            f"{report}"
        )

    # Convert all interchanges at once to get a shared force field
    # and separate topologies that all reference the same parameters
    force_field, tensor_tops = smee.converters.convert_interchange(interchanges)
    force_field = force_field.to(device)

    # Move each topology to the device
    tensor_tops = [top.to(device) for top in tensor_tops]

    if settings.linearise_harmonics:
        force_field = linearise_harmonics_force_field(force_field, device)
        for i in range(len(tensor_tops)):
            tensor_tops[i] = linearise_harmonics_topology(tensor_tops[i], device)

    return (
        mols,
        bespoke_ff,
        tensor_tops,
        force_field,
    )


def _expand_torsions(
    ff: openff.toolkit.ForceField, excluded_smirks: list[str] | None = None
) -> openff.toolkit.ForceField:
    """Expand the torsion potential to include K0-4 for proper torsions."""
    excluded_smirks = excluded_smirks or []
    ff_copy = copy.deepcopy(ff)
    torsion_handler = ff_copy.get_parameter_handler("ProperTorsions")
    for parameter in torsion_handler:
        # Avoid expanding any excluded smarts
        if parameter.smirks in excluded_smirks:
            continue
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


def _add_angle_within_range(initial_angle: float, diff: float) -> float:
    """Add a difference to an angle cap to be within [0, pi]."""
    new_angle = initial_angle + diff
    if diff > 0:
        return min(new_angle, math.pi)
    else:
        return max(new_angle, 0.0)


def _compute_linear_harmonic_params(
    k: float,
    eq_value: float,
    compute_lower_bound: Callable[[float], float],
    compute_upper_bound: Callable[[float], float],
) -> tuple[float, float, float, float]:
    """Compute linearized harmonic parameters from standard parameters.

    This generic function distributes a force constant across two bounds,
    inversely proportional to the distance from each bound.

    Args:
        k: Force constant (e.g., kcal/mol/Å² or kcal/mol/rad²)
        eq_value: Equilibrium value (e.g., bond length or angle)
        compute_lower_bound: Function that takes eq_value and returns
            lower bound
        compute_upper_bound: Function that takes eq_value and returns
            upper bound

    Returns:
        Tuple of (k1, k2, eq1, eq2) where:
        - k1, k2: Distributed force constants
        - eq1, eq2: Lower and upper equilibrium value bounds
    """
    eq1 = compute_lower_bound(eq_value)
    eq2 = compute_upper_bound(eq_value)
    d = eq2 - eq1
    # Distribute force constant inversely proportional to distance from bounds
    k1 = k * (eq2 - eq_value) / d
    k2 = k * (eq_value - eq1) / d
    return k1, k2, eq1, eq2


def _linearize_bond_parameters(
    potential: smee.TensorPotential, device_type: str
) -> smee.TensorPotential:
    """Linearize bond potential parameters.

    Converts standard harmonic bond parameters (k, length) to linearized
    form (k1, k2, b1, b2) where the equilibrium bond length range is
    [0.5*length, 1.5*length].
    """
    new_potential = copy.deepcopy(potential)
    new_potential.type = "LinearBonds"
    new_potential.fn = "(k1+k2)/2*(r-(k1*length1+k2*length2)/(k1+k2))**2"
    new_potential.parameter_cols = ("k1", "k2", "b1", "b2")

    # Get dtype from the first parameter
    dtype = potential.parameters.dtype

    new_params = [
        _compute_linear_harmonic_params(
            param[0].item(),
            param[1].item(),
            lambda b: b - 0.4,  # Lower bound: current length - 0.4 Å
            lambda b: b + 0.4,  # Upper bound: current length + 0.4 Å
        )
        for param in potential.parameters
    ]

    new_potential.parameters = torch.tensor(
        new_params, dtype=dtype, requires_grad=False, device=device_type
    )
    new_potential.parameter_units = (
        _KCAL_PER_MOL_ANGSQ,
        _KCAL_PER_MOL_ANGSQ,
        _ANGSTROM,
        _ANGSTROM,
    )
    return new_potential


def _linearize_angle_parameters(
    potential: smee.TensorPotential, device_type: str
) -> smee.TensorPotential:
    """Linearize angle potential parameters.

    Converts standard harmonic angle parameters (k, angle) to linearized form
    (k1, k2, angle1, angle2) where the equilibrium angle range is [0, π].
    """
    new_potential = copy.deepcopy(potential)
    new_potential.type = "LinearAngles"
    new_potential.fn = "(k1+k2)/2*(r-(k1*angle1+k2*angle2)/(k1+k2))**2"
    new_potential.parameter_cols = ("k1", "k2", "angle1", "angle2")

    # Get dtype from the first parameter
    dtype = potential.parameters.dtype

    new_params = [
        _compute_linear_harmonic_params(
            param[0].item(),
            param[1].item(),
            lambda a: max(0.0, a - math.pi / 3),  # Lower bound: max(0, angle - π/3)
            lambda a: min(math.pi, a + math.pi / 3),  # Upper bound: min(π, angle + π/3)
        )
        for param in potential.parameters
    ]

    new_potential.parameters = torch.tensor(
        new_params, dtype=dtype, requires_grad=False, device=device_type
    )
    new_potential.parameter_units = (
        _KCAL_PER_MOL_RADSQ,
        _KCAL_PER_MOL_RADSQ,
        _RADIANS,
        _RADIANS,
    )
    return new_potential


def linearise_harmonics_force_field(
    ff: smee.TensorForceField, device_type: str
) -> smee.TensorForceField:
    """Linearize the harmonic potential parameters in the forcefield.

    This converts Bonds and Angles potentials to their linearized forms
    (LinearBonds and LinearAngles) for more robust optimization.
    """
    ff_copy = copy.deepcopy(ff)
    ff_copy.potentials = []

    for potential in ff.potentials:
        if potential.type == "Bonds":
            ff_copy.potentials.append(
                _linearize_bond_parameters(potential, device_type)
            )
        elif potential.type == "Angles":
            ff_copy.potentials.append(
                _linearize_angle_parameters(potential, device_type)
            )
        else:
            ff_copy.potentials.append(potential)

    return ff_copy


def linearise_harmonics_topology(
    topology: smee.TensorTopology, device_type: TorchDevice
) -> smee.TensorTopology:
    """Linearize harmonic potential parameters in the topology.

    This updates the topology to use LinearBonds and LinearAngles
    parameter maps instead of Bonds and Angles.
    """
    topology_copy = topology.to(device_type)
    if "Bonds" in topology_copy.parameters:
        topology_copy.parameters["LinearBonds"] = copy.deepcopy(
            topology_copy.parameters["Bonds"]
        )
        del topology_copy.parameters["Bonds"]
    if "Angles" in topology_copy.parameters:
        topology_copy.parameters["LinearAngles"] = copy.deepcopy(
            topology_copy.parameters["Angles"]
        )
        del topology_copy.parameters["Angles"]
    return topology_copy
