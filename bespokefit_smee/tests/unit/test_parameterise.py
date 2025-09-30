"""Unit tests for the parameterizer module."""

import pytest
from openff.toolkit import ForceField
from pint import Quantity

from ...parameterise import (
    convert_to_smirnoff,
    parameterise,
)
from ...settings import ParameterisationSettings


@pytest.mark.parametrize(
    "linear_harmonics",
    [
        True,
        False,
    ],
)
def test_round_trip(linear_harmonics: bool, jnk1_lig_smiles):
    """
    Check that we can convert a general force field to and from a
    molecule-specific TensorForceField while still assigning the same\
    parameters.
    """
    # base_ff = ForceField("openff_unconstrained-2.2.1.offxml")
    base_ff = ForceField("openff_unconstrained-2.3.0-rc1.offxml")
    settings = ParameterisationSettings(
        linear_harmonics=linear_harmonics,
        smiles=jnk1_lig_smiles,
        msm_settings=None,
        initial_force_field="openff_unconstrained-2.3.0-rc1.offxml",
        expand_torsions=False,
    )
    mol, _, _, tff, _ = parameterise(settings=settings, device="cpu")

    # Convert the TensorForceField back to a SMIRNOFF force field
    recreated_ff = convert_to_smirnoff(tff, base=base_ff)

    # Label the molecule with both ffs
    base_labels = base_ff.label_molecules(mol.to_topology())[0]
    recreated_labels = recreated_ff.label_molecules(mol.to_topology())[0]

    # Check that everything matches, other than the SMIRKS
    assert base_labels.keys() == recreated_labels.keys()

    for param_type in base_labels.keys():
        base_params = base_labels[param_type]
        recreated_params = recreated_labels[param_type]

        # Continue if both are empty (e.g. constraints)
        if not base_params and not recreated_params:
            continue

        for param_key in base_params.keys():
            base_param_dict = base_params[param_key].to_dict()
            recreated_param_dict = recreated_params[param_key].to_dict()

            # Make sure we haven't lost any keys
            assert set(base_param_dict.keys()).issubset(recreated_param_dict.keys()), (
                f"Parameter {param_key} dicts do not have the same keys: "
                f"Base: {base_param_dict.keys()} "
                f"Recreated: {recreated_param_dict.keys()}"
            )

            # Filter out attributes that are not relevant for comparison
            unwanted_keys = {
                "smirks",  # SMIRKS get modified during conversion
                "id",  # IDs will differ
            }

            for attr in base_param_dict.keys():
                if attr in unwanted_keys:
                    continue

                # If this is a pint Quantity, convert to base units for comparison
                if isinstance(base_param_dict[attr], Quantity):
                    assert base_param_dict[
                        attr
                    ].to_base_units().magnitude == pytest.approx(
                        recreated_param_dict[attr].to_base_units().magnitude, rel=1e-5
                    ), (
                        f"Parameter {param_key} attribute {attr} does not match: "
                        f"Base: {base_param_dict[attr]} "
                        f"Recreated: {recreated_param_dict[attr]}"
                    )
                else:
                    assert base_param_dict[attr] == recreated_param_dict[attr], (
                        f"Parameter {param_key} attribute {attr} does not match: "
                        f"Base: {base_param_dict[attr]} "
                        f"Recreated: {recreated_param_dict[attr]}"
                    )
