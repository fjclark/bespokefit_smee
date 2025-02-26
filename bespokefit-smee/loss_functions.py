"""
LOSS_FUNCTIONS:

Loss functions for tuning te forcefield
"""
###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################
import smee
import torch
import datasets
import openff.toolkit
import copy
###############################################################################
############################### FUNCTIONS #####################################
###############################################################################

def prediction_loss(
    dataset: datasets.Dataset,
    force_field: smee.TensorForceField,
    topology: smee.TensorTopology,
    loss_force_weight: float
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
        energy_ref   = entry["energy"]
        forces_ref   = entry["forces"].reshape(len(energy_ref), -1, 3)
        coords_ref   = (entry["coords"].reshape(len(energy_ref), -1, 3).detach().requires_grad_(True))
        weight_ref   = entry["weight"]
        energy_prd   = smee.compute_energy(topology, force_field, coords_ref)
        forces_prd   = -torch.autograd.grad(energy_prd.sum(),coords_ref,create_graph=True,retain_graph=True,allow_unused=True)[0]    
        energy_prd_0 = energy_prd.detach()[0]
        energy_loss.append((energy_prd - energy_ref - energy_prd_0) * weight_ref)
        forces_loss.append(((forces_prd - forces_ref) * weight_ref.reshape(len(energy_ref),1,1)).reshape(-1, 3))
    lossE  = (torch.cat(energy_loss) ** 2).mean()
    lossF  = (torch.cat(forces_loss) ** 2).mean()
    return lossE + lossF * loss_force_weight

