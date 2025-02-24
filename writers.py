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
        forces_ref  = entry["forces"].reshape(len(energy_ref), -1, 3)
        coords_ref   = (entry["coords"].reshape(len(energy_ref), -1, 3).detach().requires_grad_(True))
        topology     = copy.deepcopy(topology_in)
        energy_prd   = smee.compute_energy(topology, force_field, coords_ref)
        forces_prd   = -torch.autograd.grad(energy_prd.sum(),coords_ref,create_graph=True,retain_graph=True,allow_unused=True)[0]
        energy_prd_0 = energy_prd.detach()[0]
        energy_ref_all.append(energy_ref)
        energy_prd_all.append(energy_prd - energy_prd_0)
    energy_out_ref = torch.cat(energy_ref_all).detach()
    energy_out_prd = torch.cat(energy_prd_all).detach()
    forces_out_ref = (forces_ref.detach()).sum(dim=(1,2))
    forces_out_prd = (forces_prd.detach()).sum(dim=(1,2))
    print_array    = numpy.c_[energy_out_ref[:],energy_out_prd[:],forces_out_ref[:],forces_out_prd[:]]
    with open(filename, "w") as out_file:
           numpy.savetxt(out_file,print_array,delimiter=' ',newline='\n')     
    energy_summary = torch.std_mean(energy_out_prd - energy_out_ref)
    return energy_summary[1].item(),energy_summary[0].item()

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
    
def write_potential_comparison(pot1: smee.TensorPotential, pot2: smee.TensorPotential):
    parameter_rows = []
    for key_id, value in enumerate(zip(pot1.parameters.detach(),pot2.parameters.detach())):
        row = {"ID": key_id}
        row.update(
            {
                f"{col}": (f"{value[0][idx].item():.4f} --> {value[1][idx].item():.4f}")
                for idx, col in enumerate(pot1.parameter_cols)
            }
        )
        parameter_rows.append(row)

    print(f" {pot1.type} ".center(88, "="), flush=True)
    print(f"fn={pot1.fn}", flush=True)

    if pot1.attributes is not None:
        attribute_rows = [
            {
                f"{col}{_format_unit(pot1.attribute_units[idx])}": (f"{pot1.attributes[idx].item():.4f} ")
                for idx, col in enumerate(pot1.attribute_cols)
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
