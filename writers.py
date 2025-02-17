"""
WRITERS:

Output functions for run-fit
"""
###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################
import smee
import torch
import datasets
import tensorboardX
import contextlib
import pathlib
import numpy
import typing
import pandas
import copy
from openff.units import unit
    
###############################################################################
############################### FUNCTIONS #####################################
###############################################################################

@contextlib.contextmanager
def open_writer(path: pathlib.Path) -> tensorboardX.SummaryWriter:
    path.mkdir(parents=True, exist_ok=True)
    with tensorboardX.SummaryWriter(str(path)) as writer:
        yield writer

def write_scatter(
    dataset: datasets.Dataset,
    force_field: smee.TensorForceField,
    topology_in: smee.TensorTopology,
    filename: str
):
    energy_ref_all, energy_prd_all = [], []
    index_tracker = []
    for i, entry in enumerate(dataset):
        energy_ref   = entry["energy"]
        coords_ref   = (entry["coords"].reshape(len(energy_ref), -1, 3).detach().requires_grad_(True))
        topology     = copy.deepcopy(topology_in)
        energy_prd   = smee.compute_energy(topology, force_field, coords_ref)
        energy_prd_0 = energy_prd.detach()[0]
        energy_ref_all.append(energy_ref)
        energy_prd_all.append(energy_prd - energy_prd_0)
        index_tracker.append([i]*len(energy_ref))
    energy_out_ref   = torch.cat(energy_ref_all).detach()
    energy_out_prd   = torch.cat(energy_prd_all).detach()
    index_out        = [x for xi in index_tracker for x in xi]
    print_array      = numpy.c_[energy_out_ref[:],energy_out_prd[:],index_out[:]]
    with open(filename, "w") as out_file:
           numpy.savetxt(out_file,print_array,delimiter=' ',newline='\n') 

def write_metrics(
    i: int,
    loss_trn: torch.Tensor,
    loss_tst: torch.Tensor,
    writer: tensorboardX.SummaryWriter,
    filename: str
):
    filename.write(f"{i} {loss_trn:.10f} {loss_tst:.10f}\n")
    writer.add_scalar("loss_trn", loss_trn.detach().item(), i)
    writer.add_scalar("loss_tst", loss_tst.detach().item(), i)
    writer.flush()

def _format_parameter_id(id_: typing.Any) -> str:
    """Format a parameter ID for display in a table."""
    id_str = id_ if "EP" not in id_ else id_[: id_.index("EP") + 2]
    return id_str[:60] + (id_str[60:] and "...")
    
def write_potential_summary(potential: smee.TensorPotential):
    parameter_rows = []
    for key_id, value in enumerate(potential.parameters.detach()):
#    for key, value in zip(potential.parameter_keys, potential.parameters.detach(), strict=True):
        row = {"ID": key_id}
        row.update(
            {
                f"{col}": (f"{value[idx].item():.4f}")
                for idx, col in enumerate(potential.parameter_cols)
            }
        )
        parameter_rows.append(row)

    print(f" {potential.type} ".center(88, "="), flush=True)
    print(f"fn={potential.fn}", flush=True)

    if potential.attributes is not None:
        attribute_rows = [
            {
                f"{col}{_format_unit(potential.attribute_units[idx])}": (f"{potential.attributes[idx].item():.4f} ")
                for idx, col in enumerate(potential.attribute_cols)
            }
        ]
        print("")
        print("attributes=", flush=True)
        print("")
        print(pandas.DataFrame(attribute_rows).to_string(index=False), flush=True)

    print("")
    print("parameters=", flush=True)
    print("")
    print(pandas.DataFrame(parameter_rows).to_string(index=False), flush=True)
