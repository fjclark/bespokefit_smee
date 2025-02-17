"""Apply OpenFF parameters to molecule, cluster conformers by RMSD and train"""
from argparse import ArgumentParser
import datetime
import os
import sys
import pathlib
from contextlib import redirect_stdout, redirect_stderr

from data_maker     import *
from writers        import *
from parameterizer  import *
from loss_functions import *

import openff.interchange
import openff.toolkit
import openff.units
from openmm.app  import *
from openmm      import *
from openmm.unit import *
from openmmml    import MLPotential
import tensorboardX
import torch
import torch.distributed
import datasets
import datasets.distributed
import datasets.table
import datasets.combine
import typing

def main(world_size: int, args: list):
#   read in the command line inputs
    smiles       = args.smiles  # SMILES string
    method       = args.method  # Method for generating data
    params       = args.params  # Parameters used in the ML fit
    memory       = args.memory  # Include previous data on iteration
    n_epochs     = args.Nepochs # Number of epochs in the ML fit
    lr           = args.lr      # Learning Rate in the ML fit
    lrd          = args.lrd     # Learning Rate Decay
    ff_path      = [args.ff]    # Starting guess force field
    ML_path      = [args.pot]   # Name of the MD potential used
    temperature  = args.temp    # Temperature in Kelvin
    Ntrn         = args.Ntrn    # Number of MD steps in training sets
    Ntst         = args.Ntst    # Number of MD steps in testing sets
    Ncnf         = args.Ncnf    # Number of Starting Conformers
    Ngap         = args.Ngap    # Number of Steps Between MD Snapshot
    Nign         = args.Nign    # Number of Steps Ignored
    Nits         = args.Nits    # Number of ML Iterations performed
    dt           = args.dt      # MD Stepsize in picoseconds
    source_train = args.data    # Location of pre-calculated data set: only used if method = "DATA"
#   parameterize the molecule and output the forcefield to a file
    print("Generating Molecule from Smiles",flush=True)
    mol = openff.toolkit.Molecule.from_smiles(smiles,allow_undefined_stereo=True,hydrogens_are_explicit=False)
    off = expand_torsions(openff.toolkit.ForceField(*ff_path))
    print("Generating Initial Force-Field",flush=True)
    force_field, topology   = apply_parameters(mol,off,*ML_path) 
    trainable               = parameter_builder(params,force_field)
    if params in ["LBA", "LBAT","LBATI","LLBAT"]:
        topology            = linearize_topologies_1(topology)
    if params in ["LLBATI"]:
        topology            = linearize_topologies_2(topology)
    VdW_forcefield          = openff.toolkit.typing.engines.smirnoff.forcefield.ForceField(*ff_path)
    off_force_field         = convert_to_smirnoff(trainable.force_field,base = VdW_forcefield)
    off_force_field.to_file("default.offxml")
# get the inital training data
    print("Preparing Initial Training Dataset",flush=True)
    if method == "DATA":    
        dataset = datasets.Dataset.load_from_disk(f"{source_train}")
    elif method == "MMMD":
        dataset = get_data_MMMD(mol,off_force_field,ML_path,temperature,dt,Ntrn,Ncnf,Ngap,Nign)
    elif method == "cMMMD":
        dataset = get_data_cMMMD(mol,off_force_field,ML_path,temperature,dt,Ntrn,Ncnf,Ngap,Nign)
    elif method == "MLMD":
        dataset = get_data_MLMD(mol,off_force_field,ML_path,temperature,dt,Ntrn,Ncnf,Ngap,Nign)
    else:
        print("ERROR:: Chosen method is not supported (method = ",method,")")
        exit(0)
    with open("/dev/null", 'w') as f:
        with redirect_stderr(f):
            dataset.save_to_disk("data_it_0")
# Generate the test set
    print("Preparing Initial Test Dataset")
    if method == "cMMMD":
#        dataset_test = get_data_cMMMD(mol,off_force_field,ML_path,temperature,dt,Ntst,Ncnf,Ngap,Nign)
        dataset_test = get_data_MMMD(mol,off_force_field,ML_path,temperature,dt,Ntst,Ncnf,Ngap,Nign)
    else:
        dataset_test = get_data_MMMD(mol,off_force_field,ML_path,temperature,dt,Ntst,Ncnf,Ngap,Nign)
# Generate the Energy Scatter Plot
    write_scatter(dataset_test, trainable.force_field, topology, "default.scat")
    for iteration in tqdm(range(Nits),leave=False,colour='magenta',desc="Iterating the Fit"):
# some book-keeping and outputing
        with open("trained-" + str(iteration) + ".param-in", 'w') as f:
            with redirect_stdout(f):
                for potential_type in trainable.potential_types:
                    write_potential_summary(trainable.force_field.potentials_by_type[potential_type])
        timestamp      = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = pathlib.Path(f"{timestamp}")
# run the ML training
        with open("training-" + str(iteration) + ".data", 'w') as metrics_file:
            with open_writer(experiment_dir) as writer:
                optimizer = torch.optim.Adam(trainable.parameters, lr=lr, amsgrad=True)
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
#                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
                for v in tensorboardX.writer.hparams({"optimizer": "Adam", "lr": lr}, {}):
                    writer.file_writer.add_summary(v)
                for i in tqdm(range(n_epochs),leave=False,colour='blue',desc="Running ML Optimization"):
                    loss_trn = prediction_loss(dataset, trainable.force_field, topology)
                    if i % 10 == 0:
                        loss_tst = prediction_loss(dataset_test, trainable.force_field, topology)
                        write_metrics(i, loss_trn, loss_tst, writer, metrics_file)
                    loss_trn.backward(retain_graph=True)
                    trainable.freeze_grad()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    trainable.clamp()
#                    print(i,scheduler.get_lr())
                    if i % 10 == 0:
                        scheduler.step()
#                    scheduler.step()
# some book-keeping and outputing
            loss_tst = prediction_loss(dataset_test, trainable.force_field, topology)
            write_metrics(n_epochs, loss_trn, loss_tst, writer, metrics_file)
        with open("trained-" + str(iteration) + ".param-out", 'w') as f:
            with redirect_stdout(f):
                for potential_type in trainable.potential_types:
                    write_potential_summary(trainable.force_field.potentials_by_type[potential_type])
# Output the forcefield to a file
        off_force_field = convert_to_smirnoff(trainable.force_field,base = VdW_forcefield)
        off_force_field.to_file("trained-" + str(iteration) + ".offxml")
# get the next iteration of training data
        if memory:
            if method == "cMMMD":
                dataset = datasets.combine.concatenate_datasets([dataset,get_data_cMMMD(mol,off_force_field,ML_path,temperature,dt,Ntrn,Ncnf,Ngap,Nign)])
            else:
                dataset = datasets.combine.concatenate_datasets([dataset,get_data_MMMD(mol,off_force_field,ML_path,temperature,dt,Ntrn,Ncnf,Ngap,Nign)])
        else:
            if method == "cMMMD":
                dataset = get_data_cMMMD(mol,off_force_field,ML_path,temperature,dt,Ntrn,Ncnf,Ngap,Nign)
            else:
                dataset = get_data_MMMD(mol,off_force_field,ML_path,temperature,dt,Ntrn,Ncnf,Ngap,Nign)
        with open("/dev/null", 'w') as f:
            with redirect_stderr(f):
                dataset.save_to_disk("data_it_" + str(iteration))
# Generate the Energy Scatter Plot
        write_scatter(dataset_test, trainable.force_field, topology, "trained-" + str(iteration) + ".scat")
if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    parser = ArgumentParser()
    parser.add_argument("--smiles",  type=str,   help="SMILES string",                        required=True,  default=None)
    parser.add_argument("--method",  type=str,   help="Method for generating data",           required=False, default="MMMD")
    parser.add_argument("--params",  type=str,   help="Parameters used in the ML fit",        required=False, default="LLBATI")
    parser.add_argument("--memory",  type=bool,  help="Include previous data on iteration",   required=False, default=True)    
    parser.add_argument("--Nepochs", type=int,   help="Number of epochs in the ML fit",       required=False, default=1000)
    parser.add_argument("--lr",      type=float, help="Learning Rate in the ML fit",          required=False, default=0.1)
    parser.add_argument("--lrd",     type=float, help="Learning Rate Decay",                  required=False, default=0.99)
    parser.add_argument("--ff",      type=str,   help="Starting guess force field",           required=False, default="openff-2.2.0.offxml")
    parser.add_argument("--pot",     type=str,   help="Name of the MD potential used",        required=False, default="mace-off23-small")
    parser.add_argument("--temp",    type=int,   help="Temperature in Kelvin",                required=False, default=500)
    parser.add_argument("--Ntrn",    type=int,   help="Number of MD steps in training sets",  required=False, default=100)
    parser.add_argument("--Ntst",    type=int,   help="Number of MD steps in testing sets",   required=False, default=100)
    parser.add_argument("--Ncnf",    type=int,   help="Number of Starting Conformers",        required=False, default=10)
    parser.add_argument("--Ngap",    type=int,   help="Number of Steps Between MD Snapshots", required=False, default=10)
    parser.add_argument("--Nign",    type=int,   help="Number of Steps Ignored",              required=False, default=100)
    parser.add_argument("--Nits",    type=int,   help="Number of ML Iterations Performed",    required=False, default=5)
    parser.add_argument("--dt",      type=float, help="MD Stepsize in picoseconds",           required=False, default=0.001)
    parser.add_argument("--data",    type=str,   help="Location of pre-calculated data set",  required=False, default="train_data")
    args = parser.parse_args()
    world_size = 1
    main(world_size, args)

