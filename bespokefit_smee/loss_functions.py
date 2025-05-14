"""
LOSS_FUNCTIONS:

Loss functions for tuning te forcefield
"""

###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

import datasets
import smee
import torch

###############################################################################
############################### FUNCTIONS #####################################
###############################################################################


def prediction_loss(
    dataset: datasets.Dataset,
    force_field: smee.TensorForceField,
    topology: smee.TensorTopology,
    loss_force_weight: float,
    device_type: str,
):
    """Predict the loss function for a guess forcefield against a dataset.

    Args:
        dataset: The dataset to predict the energies and forces of.
        force_field: The force field to use to predict the energies and forces.
        topologies: The topologies of the molecules in the dataset.
    Returns:
        Loss value.
    """
    energy_loss, forces_loss = [], []
    for entry in dataset:
        energy_ref = entry["energy"].to(device_type)
        forces_ref = entry["forces"].reshape(len(energy_ref), -1, 3).to(device_type)
        coords_ref = (
            entry["coords"]
            .reshape(len(energy_ref), -1, 3)
            .to(device_type)
            .detach()
            .requires_grad_(True)
        )
        weight_ref = entry["weight"].to(device_type)
        energy_prd = smee.compute_energy(topology, force_field, coords_ref)
        forces_prd = -torch.autograd.grad(
            energy_prd.sum(),
            coords_ref,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        energy_prd_0 = energy_prd.detach()[0]
        energy_loss.append((energy_prd - energy_ref - energy_prd_0) * weight_ref)
        forces_loss.append(
            (
                (forces_prd - forces_ref) * weight_ref.reshape(len(energy_ref), 1, 1)
            ).reshape(-1, 3)
        )
    lossE = (torch.cat(energy_loss) ** 2).mean()
    lossF = (torch.cat(forces_loss) ** 2).mean()
    return lossE + lossF * loss_force_weight
