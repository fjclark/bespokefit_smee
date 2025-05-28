"""
WRITERS:

Output functions for run-fit
"""

import contextlib
import pathlib
import typing
from typing import TextIO

import datasets
import descent.train
import loguru
import numpy
import pandas
import smee
import tensorboardX
import torch

from .loss_functions import predict
from .utils.typing import PathLike

logger = loguru.logger


@contextlib.contextmanager
def open_writer(path: pathlib.Path) -> tensorboardX.SummaryWriter:
    path.mkdir(parents=True, exist_ok=True)
    with tensorboardX.SummaryWriter(str(path)) as writer:
        yield writer


# def write_scatter(
#     dataset: datasets.Dataset,
#     force_field: smee.TensorForceField,
#     topology_in: smee.TensorTopology,
#     device_type: str,
#     filename: str,
# ) -> tuple[float, float, float, float]:
#     energy_ref_all, energy_prd_all = [], []
#     for entry in dataset:
#         energy_ref = entry["energy"].to(device_type)
#         forces_ref = entry["forces"].reshape(len(energy_ref), -1, 3).to(device_type)
#         coords_ref = (
#             entry["coords"]
#             .reshape(len(energy_ref), -1, 3)
#             .to(device_type)
#             .detach()
#             .requires_grad_(True)
#         )
#         topology = copy.deepcopy(topology_in)
#         energy_prd = smee.compute_energy(topology, force_field, coords_ref)
#         forces_prd = -torch.autograd.grad(
#             energy_prd.sum(),
#             coords_ref,
#             create_graph=True,
#             retain_graph=True,
#             # allow_unused=True,
#         )[0]
#         energy_prd_0 = energy_prd.detach()[0]
#         energy_ref_all.append(energy_ref)
#         energy_prd_all.append(energy_prd - energy_prd_0)
#     energy_out_ref = torch.cat(energy_ref_all).detach()
#     energy_out_prd = torch.cat(energy_prd_all).detach()
#     forces_out_ref = (forces_ref.detach()).sum(dim=(1, 2))
#     forces_out_prd = (forces_prd.detach()).sum(dim=(1, 2))
#     print_array = numpy.c_[
#         energy_out_ref.cpu()[:],
#         energy_out_prd.cpu()[:],
#         forces_out_ref.cpu()[:],
#         forces_out_prd.cpu()[:],
#     ]
#     with open(filename, "w") as out_file:
#         numpy.savetxt(out_file, print_array, delimiter=" ", newline="\n")
#     energy_summary = torch.std_mean(energy_out_prd - energy_out_ref)
#     forces_summary = torch.std_mean(forces_out_prd - forces_out_ref)
#     return (
#         energy_summary[1].item(),
#         energy_summary[0].item(),
#         forces_summary[1].item(),
#         forces_summary[0].item(),
#     )


def write_scatter(
    dataset: datasets.Dataset,
    force_field: smee.TensorForceField,
    topology_in: smee.TensorTopology,
    device_type: str,
    filename: PathLike,
) -> tuple[float, float, float, float]:
    energy_ref_all, energy_pred_all, forces_ref_all, forces_pred_all = predict(
        dataset,
        force_field,
        {dataset[0]["smiles"]: topology_in},
        device_type=device_type,
        normalize=False,
    )
    with torch.no_grad():
        # Write out the pre-conformer differences
        energy_diffs = energy_pred_all - energy_ref_all
        # Flatten forces to 1D for easier handling
        force_diffs = (forces_pred_all - forces_ref_all).reshape(-1)
        # Pad the energy differences to match the forces, with nan
        energy_diffs_padded = torch.full(
            force_diffs.shape, float("nan"), device=energy_diffs.device
        )
        energy_diffs_padded[: energy_diffs.numel()] = energy_diffs.reshape(-1)

        print_array = numpy.c_[energy_diffs_padded.cpu()[:], force_diffs.cpu()[:]]

        with open(filename, "w") as out_file:
            numpy.savetxt(out_file, print_array, delimiter=" ", newline="\n")
        energy_summary = torch.std_mean(energy_diffs)
        forces_summary = torch.std_mean(force_diffs)
        return (
            energy_summary[1].item(),
            energy_summary[0].item(),
            forces_summary[1].item(),
            forces_summary[0].item(),
        )


def report(
    step: int,
    x: torch.Tensor,
    loss: torch.Tensor,
    gradient: torch.Tensor,
    hessian: torch.Tensor,
    step_quality: float,
    accept_step: bool,
    trainable: descent.train.Trainable,
    topology: smee.TensorTopology,
    dataset_test: datasets.Dataset,
    metrics_file: PathLike,
    experiment_dir: pathlib.Path,
) -> None:
    if step % 1 == 0:
        with torch.enable_grad():  # type: ignore[no-untyped-call]
            y_ref, y_pred = predict(
                dataset_test,
                trainable.to_force_field(x),
                {dataset_test[0]["smiles"]: topology},
                device_type=x.device.type,
                normalize=False,
            )[:2]
            loss_tst = ((y_pred - y_ref) ** 2).mean()

            with open_writer(experiment_dir) as writer:
                with open(metrics_file, "a") as f:
                    write_metrics(step, loss, loss_tst, writer, f)


def write_metrics(
    i: int,
    loss_trn: torch.Tensor,
    loss_tst: torch.Tensor,
    writer: tensorboardX.SummaryWriter,
    outfile: TextIO,
) -> None:
    outfile.write(
        f"{i} {loss_trn.detach().item():.10f} {loss_tst.detach().item():.10f}\n"
    )
    writer.add_scalar("loss_trn", loss_trn.detach().item(), i)
    writer.add_scalar("loss_tst", loss_tst.detach().item(), i)
    writer.flush()


def _format_parameter_id(id_: typing.Any) -> str:
    """Format a parameter ID for display in a table."""
    id_str = id_ if "EP" not in id_ else id_[: id_.index("EP") + 2]
    return str(id_str[:60] + (id_str[60:] and "..."))


def get_potential_summary(potential: smee.TensorPotential) -> str:
    output = [""]
    parameter_rows = []
    for key_id, value in enumerate(potential.parameters.detach()):
        row: dict[str, int | str] = {"ID": key_id}
        row.update(
            {
                f"{col}": (f"{value[idx].item():.4f}")
                for idx, col in enumerate(potential.parameter_cols)
            }
        )
        parameter_rows.append(row)

    output.append(
        f" {potential.type} ".center(88, "="),
    )
    output.append(f"fn={potential.fn}")

    if potential.attributes is not None:
        assert potential.attribute_units is not None, (
            "Attribute units are None even though attributes are not None"
        )
        assert potential.attribute_cols is not None, (
            "Attribute columns are None even though attributes are not None"
        )
        attribute_rows = [
            {
                f"{col}{potential.attribute_units[idx]}": (
                    f"{potential.attributes[idx].item():.4f} "
                )
                for idx, col in enumerate(potential.attribute_cols)
            }
        ]
        output.append("")
        output.append("attributes=")
        output.append("")
        output.append(pandas.DataFrame(attribute_rows).to_string(index=False))

    output.append("")
    output.append("parameters=")
    output.append("")
    output.append(pandas.DataFrame(parameter_rows).to_string(index=False))

    return "\n".join(output)


def get_potential_comparison(
    pot1: smee.TensorPotential, pot2: smee.TensorPotential
) -> str:
    output = [""]
    parameter_rows = []
    for key_id, value in enumerate(
        zip(pot1.parameters.detach(), pot2.parameters.detach(), strict=False)
    ):
        row: dict[str, int | str] = {"ID": key_id}
        row.update(
            {
                f"{col}": (f"{value[0][idx].item():.4f} --> {value[1][idx].item():.4f}")
                for idx, col in enumerate(pot1.parameter_cols)
            }
        )
        parameter_rows.append(row)

    output.append(
        f" {pot1.type} vs {pot2.type} ".center(88, "="),
    )
    output.append(f"fn={pot1.fn}")

    if pot1.attributes is not None:
        assert pot1.attribute_units is not None, (
            "Attribute units are None even though attributes are not None"
        )
        assert pot1.attribute_cols is not None, (
            "Attribute columns are None even though attributes are not None"
        )
        attribute_rows = [
            {
                f"{col}{pot1.attribute_units[idx]}": (
                    f"{pot1.attributes[idx].item():.4f} "
                )
                for idx, col in enumerate(pot1.attribute_cols)
            }
        ]
        output.append("")
        output.append("attributes=")
        output.append("")
        output.append(pandas.DataFrame(attribute_rows).to_string(index=False))

    output.append("")
    output.append("parameters=")
    output.append("")
    output.append(pandas.DataFrame(parameter_rows).to_string(index=False))

    return "\n".join(output)
