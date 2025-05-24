"""
Loss functions for tuning the forcefield
"""

import typing

import datasets
import descent
import descent.optim
import descent.train
import descent.utils.loss
import smee
import torch


def prediction_loss(
    dataset: datasets.Dataset,
    force_field: smee.TensorForceField,
    topology: smee.TensorTopology,
    loss_force_weight: float,
    device_type: str,
) -> torch.Tensor:
    """Predict the loss function for a guess forcefield against a dataset.

    Args:
        dataset: The dataset to predict the energies and forces of.
        force_field: The force field to use to predict the energies and forces.
        topologies: The topologies of the molecules in the dataset.
        loss_force_weight: Weight for the force loss term.
        device_type: The device type (e.g., 'cpu' or 'cuda').

    Returns:
        Loss value.
    """
    energy_ref_all, energy_pred_all, forces_ref_all, forces_pred_all = predict(
        dataset,
        force_field,
        {dataset[0]["smiles"]: topology},
        device_type=device_type,
        normalize=False,
    )
    loss_energy: torch.Tensor = ((energy_ref_all - energy_pred_all) ** 2).mean()
    loss_forces: torch.Tensor = ((forces_ref_all - forces_pred_all) ** 2).mean()
    return loss_energy + loss_forces * loss_force_weight

    # energy_loss, forces_loss = [], []
    # for entry in dataset:
    #     energy_ref = entry["energy"].to(device_type)
    #     forces_ref = entry["forces"].reshape(len(energy_ref), -1, 3).to(device_type)
    #     coords_ref = (
    #         entry["coords"]
    #         .reshape(len(energy_ref), -1, 3)
    #         .to(device_type)
    #         .detach()
    #         .requires_grad_(True)
    #     )
    #     energy_prd = smee.compute_energy(topology, force_field, coords_ref)
    #     forces_prd = -torch.autograd.grad(
    #         energy_prd.sum(),
    #         coords_ref,
    #         create_graph=True,
    #         retain_graph=True,
    #         allow_unused=True,
    #     )[0]
    #     energy_prd_0 = energy_prd.detach()[0]
    #     energy_loss.append((energy_prd - energy_ref - energy_prd_0))
    #     forces_loss.append((forces_prd - forces_ref).reshape(-1, 3))
    # lossE: torch.Tensor = (torch.cat(energy_loss) ** 2).mean()
    # lossF: torch.Tensor = (torch.cat(forces_loss) ** 2).mean()
    # return lossE + lossF * loss_force_weight**0.5


def get_loss_closure_fn(
    trainable: descent.train.Trainable,
    topology: smee.TensorTopology,
    dataset: datasets.Dataset,
) -> descent.optim.ClosureFn:
    """
    Return a default closure function

    Args:
        trainable: The trainable object.
        topology: The topology of the system.
        dataset: The dataset to use for the loss function.

    Returns:
        A closure function that takes a tensor and returns the loss, gradient (if requested), and hessian (if requested).
    """

    def closure_fn(
        x: torch.Tensor,
        compute_gradient: bool,
        compute_hessian: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        loss, gradient, hessian = (
            torch.zeros(size=(1,), device=x.device.type),
            None,
            None,
        )

        def loss_fn(_x: torch.Tensor) -> torch.Tensor:
            """Compute the loss function for the given trainable parameters."""
            ff = trainable.to_force_field(_x)
            y_ref, y_pred = predict(
                dataset,
                ff,
                {dataset[0]["smiles"]: topology},
                device_type=x.device.type,
                normalize=False,
            )[:2]
            loss: torch.Tensor = ((y_pred - y_ref) ** 2).mean()
            return loss

        loss += loss_fn(x)

        if compute_hessian:
            hessian = torch.autograd.functional.hessian(  # type: ignore[no-untyped-call]
                loss_fn, x, vectorize=True, create_graph=False
            ).detach()
        if compute_gradient:
            (gradient,) = torch.autograd.grad(loss, x, create_graph=False)
            gradient = gradient.detach()

        return loss, gradient, hessian

    return closure_fn


def predict(
    dataset: datasets.Dataset,
    force_field: smee.TensorForceField,
    topologies: dict[str, smee.TensorTopology],
    reference: typing.Literal["mean", "min"] = "mean",
    normalize: bool = True,
    device_type: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict the relative energies [kcal/mol] and forces [kcal/mol/Å] of a dataset.

    Args:
        dataset: The dataset to predict the energies and forces of.
        force_field: The force field to use to predict the energies and forces.
        topologies: The topologies of the molecules in the dataset. Each key should be
            a fully indexed SMILES string.
        reference: The reference energy to compute the relative energies with respect
            to. This should be either the "mean" energy of all conformers, or the
            energy of the conformer with the lowest reference energy ("min").
        normalize: Whether to scale the relative energies by ``1/sqrt(n_confs_i)``
            and the forces by ``1/sqrt(n_confs_i * n_atoms_per_conf_i * 3)`` This
            is useful when wanting to compute the MSE per entry.

    Returns:
        The predicted and reference relative energies [kcal/mol] with
        ``shape=(n_confs,)``, and predicted and reference forces [kcal/mol/Å] with
        ``shape=(n_confs * n_atoms_per_conf, 3)``.
    """
    energy_ref_all, energy_pred_all = [], []
    forces_ref_all, forces_pred_all = [], []

    for entry in dataset:
        smiles = entry["smiles"]

        energy_ref = entry["energy"].to(device_type)
        forces_ref = entry["forces"].reshape(len(energy_ref), -1, 3).to(device_type)

        coords_flat = smee.utils.tensor_like(
            entry["coords"], force_field.potentials[0].parameters
        )

        coords = (
            (coords_flat.reshape(len(energy_ref), -1, 3))
            .to(device_type)
            .detach()
            .requires_grad_(True)
        )
        topology = topologies[smiles]

        energy_pred = smee.compute_energy(topology, force_field, coords)
        forces_pred = -torch.autograd.grad(
            energy_pred.sum(),
            coords,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if reference.lower() == "mean":
            energy_ref_0 = energy_ref.mean()
            energy_pred_0 = energy_pred.mean()
        elif reference.lower() == "min":
            min_idx = energy_ref.argmin()

            energy_ref_0 = energy_ref[min_idx]
            energy_pred_0 = energy_pred[min_idx]
        else:
            raise NotImplementedError(f"invalid reference energy {reference}")

        scale_energy, scale_forces = 1.0, 1.0

        if normalize:
            scale_energy = 1.0 / torch.sqrt(torch.tensor(energy_pred.numel()))
            scale_forces = 1.0 / torch.sqrt(torch.tensor(forces_pred.numel()))

        energy_ref_all.append(scale_energy * (energy_ref - energy_ref_0))
        forces_ref_all.append(scale_forces * forces_ref.reshape(-1, 3))

        energy_pred_all.append(scale_energy * (energy_pred - energy_pred_0))
        forces_pred_all.append(scale_forces * forces_pred.reshape(-1, 3))

    energy_pred_all_tensor = torch.cat(energy_pred_all)
    forces_pred_all_tensor = torch.cat(forces_pred_all)

    energy_ref_all_tensor = torch.cat(energy_ref_all)
    energy_ref_all_tensor = smee.utils.tensor_like(
        energy_ref_all_tensor, energy_pred_all_tensor
    )

    forces_ref_all_tensor = torch.cat(forces_ref_all)
    forces_ref_all_tensor = smee.utils.tensor_like(
        forces_ref_all_tensor, forces_pred_all_tensor
    )

    return (
        energy_ref_all_tensor,
        energy_pred_all_tensor,
        forces_ref_all_tensor,
        forces_pred_all_tensor,
    )
