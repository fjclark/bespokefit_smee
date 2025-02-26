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
import copy

def main(world_size: int, args: list):
#   read in the command line inputs
    smiles                   = args.smiles                   # SMILES string
    method                   = args.method                   # Method for generating data
    memory                   = args.memory                   # Include previous data on iteration
    n_epochs                 = args.N_epochs                 # Number of epochs in the ML fit
    linear_harmonics         = args.linear_harmonics         # Flag to turn of linearized harmonic potentials
    linear_torsions          = args.linear_torsions          # Flag to turn of linearized torsions potentials
    learning_rate            = args.learning_rate            # Learning Rate in the ML fit
    learning_rate_decay      = args.learning_rate_decay      # Learning Rate Decay Rate
    learning_rate_decay_step = args.learning_rate_decay_step # Learning Rate Decay Step
    loss_force_weight        = args.loss_force_weight        # Scaling Factor for the Force loss term
    ff_path                  = args.force_field_init         # Starting guess force field
    ML_path                  = args.MLMD_potential           # Name of the MLMD potential used
    MD_temperature           = args.MD_temperature           # MD Temperature in Kelvin
    MD_dt                    = args.MD_dt                    # MD Stepsize in picoseconds
    MD_stepsize              = args.MD_stepsize              # Number of Steps Between MD Snapshot
    MD_startup               = args.MD_startup               # Number of Steps Ignored
    MD_energy_lower_cutoff   = args.MD_energy_lower_cutoff   # Lower bound for the energy cutoff smoothing function
    MD_energy_upper_cutoff   = args.MD_energy_upper_cutoff   # Upper bound for the energy cutoff smoothing function
    Cluster_tolerance        = args.Cluster_tolerance        # Tolerance used in the RMSD clustering
    Cluster_Parallel         = args.Cluster_Parallel         # MPI nodes used in the RMSD clustering
    Ntrn                     = args.N_train                  # Number of MD steps in training sets
    Ntst                     = args.N_test                   # Number of MD steps in testing sets
    Ncnf                     = args.N_conformers             # Number of Starting Conformers
    Nits                     = args.N_iterations             # Number of ML Iterations performed
    source_train             = args.data                     # Location of pre-calculated data set: only used if method = "DATA"
    modSem                   = args.modSem                   # Activate the modified Seminario Starting guess
    modSem_finite_step       = args.modSem_finite_step       # Finite Step for the Hessian Calculation
    modSem_vib_scaling       = args.modSem_vib_scaling       # Vibrational Scaling Parameter
    modSem_tolerance         = args.modSem_tolerance         # Tolerance for the geometry optimize
#   Summarize input parameters
    print("Input Summary".center(88, "="), flush=True)
    print("")
    print(f"    Smiles                   :{smiles:>20}")
    print(f"    Method                   :{method:>20}")
    print(f"    Memory                   :{memory!s:>20}")
    print(f"    N_epochs                 :{n_epochs:>20}")
    print(f"    linear_harmonics         :{linear_harmonics!s:>20}")
    print(f"    linear_torsions          :{linear_torsions!s:>20}")
    print(f"    learning_rate            :{learning_rate:>20}")
    print(f"    learning_rate_decay      :{learning_rate_decay:>20}")
    print(f"    learning_rate_decay_step :{learning_rate_decay_step:>20}")
    print(f"    loss_force_weight        :{loss_force_weight:>20}")
    print(f"    force_field_init         :{ff_path:>20}")
    print(f"    MLMD_potential           :{ML_path:>20}")
    print(f"    MD_temperature           :{MD_temperature:>20} K")
    print(f"    MD_dt                    :{MD_dt:>20} fs")
    print(f"    MD_stepsize              :{MD_stepsize:>20} steps ({int(MD_stepsize*MD_dt):5d} fs)")
    print(f"    MD_startup               :{MD_startup:>20} steps ({int(MD_startup*MD_dt):5d} fs)")
    print(f"    MD_energy_lower_cutoff   :{MD_energy_lower_cutoff:>20} kcal/mole")
    print(f"    MD_energy_upper_cutoff   :{MD_energy_upper_cutoff:>20} kcal/mole")
    if(method == "cMMMD"):
        print(f"    Cluster_tolerance        :{Cluster_tolerance:>20}")
        print(f"    Cluster_Parallel         :{Cluster_Parallel:>20}")
    print(f"    N_train                  :{Ntrn:>20}")
    print(f"    N_test                   :{Ntst:>20}")
    print(f"    N_conformers             :{Ncnf:>20}")
    print(f"    N_iterations             :{Nits:>20}")
    if(method == "DATA"):
        print(f"    source_train             :{source_train:>20}")
    print(f"    modSem                   :{modSem!s:>20}")
    if(modSem):
        print(f"    modSem_finite_step       :{modSem_finite_step:>20} ang")
        print(f"    modSem_vib_scaling       :{modSem_vib_scaling:>20}")
        print(f"    modSem_tolerance         :{modSem_tolerance:>20} kcal/(mole*ang)")
    print("")
    Ntrn, Ntst = int(Ntrn / Ncnf), int(Ntst / Ncnf) #   Convert to "per-conformer"
    MD_dt = MD_dt / 1000                            #   Convert to ps
    modSem_finite_step = modSem_finite_step / 10    #   Convert to nm
#   parameterize the molecule and output the forcefield to a file
    mol = openff.toolkit.Molecule.from_smiles(smiles,allow_undefined_stereo=True,hydrogens_are_explicit=False)
    VdW_forcefield = openff.toolkit.ForceField(ff_path)
    old_force_field, trainable, topology = build_parameters(mol,VdW_forcefield,ML_path,linear_harmonics,linear_torsions,modSem,modSem_finite_step,modSem_vib_scaling,modSem_tolerance) 
    off_force_field     = convert_to_smirnoff(trainable.force_field,base = VdW_forcefield)
    off_force_field.to_file("default.offxml")
# get the inital training data
    if method == "DATA":    
        dataset = datasets.Dataset.load_from_disk(f"{source_train}")
    elif method == "MMMD":
        dataset = get_data_MMMD(mol,off_force_field,ML_path,MD_temperature,MD_dt,Ntrn,Ncnf,MD_stepsize,MD_startup,MD_energy_upper_cutoff,MD_energy_lower_cutoff)
    elif method == "cMMMD":
        dataset = get_data_cMMMD(mol,off_force_field,ML_path,MD_temperature,MD_dt,Ntrn,Ncnf,MD_stepsize,MD_startup,MD_energy_upper_cutoff,MD_energy_lower_cutoff,Cluster_tolerance,Cluster_Parallel)
    elif method == "MLMD":
        dataset = get_data_MLMD(mol,off_force_field,ML_path,MD_temperature,MD_dt,Ntrn,Ncnf,MD_stepsize,MD_startup,MD_energy_upper_cutoff,MD_energy_lower_cutoff)
    else:
        print("ERROR:: Chosen method is not supported (method = ",method,")")
        exit(0)
    with open("/dev/null", 'w') as f:
        with redirect_stderr(f):
            dataset.save_to_disk("data_it_0")
# Generate the test set
    dataset_test = get_data_MMMD(mol,off_force_field,ML_path,MD_temperature,MD_dt,Ntst,Ncnf,MD_stepsize,MD_startup,MD_energy_upper_cutoff,MD_energy_lower_cutoff)
# Generate the Energy Scatter Plot
    energy_mean, energy_SD, forces_mean, forces_SD = write_scatter(dataset_test, trainable.force_field, topology, "default.scat")
    for iteration in tqdm(range(Nits),leave=False,colour='magenta',desc="Iterating the Fit"):
# some book-keeping
        timestamp      = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = pathlib.Path(f"{timestamp}")
# run the ML training
        with open("training-" + str(iteration) + ".data", 'w') as metrics_file:
            with open_writer(experiment_dir) as writer:
                optimizer = torch.optim.Adam(trainable.parameters, lr=learning_rate, amsgrad=True)
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=learning_rate_decay)
                for v in tensorboardX.writer.hparams({"optimizer": "Adam", "lr": learning_rate}, {}):
                    writer.file_writer.add_summary(v)
                for i in tqdm(range(n_epochs),leave=False,colour='blue',desc="Running ML Optimization"):
                    loss_trn = prediction_loss(dataset, trainable.force_field, topology, loss_force_weight)
                    if i % 10 == 0:
                        loss_tst = prediction_loss(dataset_test, trainable.force_field, topology, loss_force_weight)
                        write_metrics(i, loss_trn, loss_tst, writer, metrics_file)
                    loss_trn.backward(retain_graph=True)
                    trainable.freeze_grad()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    trainable.clamp()
                    if i % learning_rate_decay_step == 0:
                        scheduler.step()
# some book-keeping and outputing
            loss_tst = prediction_loss(dataset_test, trainable.force_field, topology, loss_force_weight)
            write_metrics(n_epochs, loss_trn, loss_tst, writer, metrics_file)
        print(f"Summary for Iteration {iteration+1}".center(88, "="))
        print("")
        print(f"Parameterization".center(88, "="))
        print("")
        print("")
        for potential_type in trainable.potential_types:
            write_potential_comparison(old_force_field.potentials_by_type[potential_type],trainable.force_field.potentials_by_type[potential_type])
            old_force_field.potentials_by_type[potential_type].parameters  = copy.copy(trainable.force_field.potentials_by_type[potential_type].parameters)
# Output the forcefield to a file
        off_force_field = convert_to_smirnoff(trainable.force_field,base = VdW_forcefield)
        off_force_field.to_file("trained-" + str(iteration) + ".offxml")
# Generate the Energy Scatter Plot and summarize convergence
        energy_mean_new, energy_SD_new, forces_mean_new, forces_SD_new = write_scatter(dataset_test, trainable.force_field, topology, "trained-" + str(iteration) + ".scat")
        print("")
        print("")
        print(f"Convergence".center(88, "="))
        print("")
        print(f"    Energy Error (Mean): {energy_mean:10.3e}->{energy_mean_new:10.3e} : Change = {energy_mean_new - energy_mean:10.3e}")
        print(f"                 (SD):   {energy_SD:10.3e}->{energy_SD_new:10.3e} : Change = {energy_SD_new - energy_SD:10.3e}")
        print(f"    Forces Error (Mean): {forces_mean:10.3e}->{forces_mean_new:10.3e} : Change = {forces_mean_new - forces_mean:10.3e}")
        print(f"                 (SD):   {forces_SD:10.3e}->{forces_SD_new:10.3e} : Change = {forces_SD_new - forces_SD:10.3e}")
        print("")
        energy_mean, energy_SD = energy_mean_new, energy_SD_new
        forces_mean, forces_SD = forces_mean_new, forces_SD_new
# get the next iteration of training data, unless the loop is finished
        if iteration + 1 < Nits:
            if memory:
                if method == "cMMMD":
                    dataset = datasets.combine.concatenate_datasets([dataset,get_data_cMMMD(mol,off_force_field,ML_path,MD_temperature,MD_dt,Ntrn,Ncnf,MD_stepsize,MD_startup,MD_energy_upper_cutoff,MD_energy_lower_cutoff,Cluster_tolerance,Cluster_Parallel)])
                else:
                    dataset = datasets.combine.concatenate_datasets([dataset,get_data_MMMD(mol,off_force_field,ML_path,MD_temperature,MD_dt,Ntrn,Ncnf,MD_stepsize,MD_startup,MD_energy_upper_cutoff,MD_energy_lower_cutoff)])
            else:
                if method == "cMMMD":
                    dataset = get_data_cMMMD(mol,off_force_field,ML_path,MD_temperature,MD_dt,Ntrn,Ncnf,MD_stepsize,MD_startup,MD_energy_upper_cutoff,MD_energy_lower_cutoff,Cluster_tolerance,Cluster_Parallel)
                else:
                    dataset = get_data_MMMD(mol,off_force_field,ML_path,MD_temperature,MD_dt,Ntrn,Ncnf,MD_stepsize,MD_startup,MD_energy_upper_cutoff,MD_energy_lower_cutoff)
            with open("/dev/null", 'w') as f:
                with redirect_stderr(f):
                    dataset.save_to_disk("data_it_" + str(iteration))
if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    parser = ArgumentParser()
    parser.add_argument("--smiles",                   type=str,   help="SMILES string",                              required=True,  default=None)
    parser.add_argument("--method",                   type=str,   help="Method for generating data",                 required=False, default="MMMD")
    parser.add_argument("--N_epochs",                 type=int,   help="Number of epochs in the ML fit",             required=False, default=1000)
    parser.add_argument("--learning_rate",            type=float, help="Learning Rate in the ML fit",                required=False, default=0.1)
    parser.add_argument("--learning_rate_decay",      type=float, help="Learning Rate Decay",                        required=False, default=0.99)
    parser.add_argument("--learning_rate_decay_step", type=int,   help="Learning Rate Decay Step",                   required=False, default=10)
    parser.add_argument("--loss_force_weight",        type=float, help="Scaling Factor for the Force loss term",     required=False, default=1e5)
    parser.add_argument("--force_field_init",         type=str,   help="Starting guess force field",                 required=False, default="openff-2.2.0.offxml")
    parser.add_argument("--MLMD_potential",           type=str,   help="Name of the MD potential used",              required=False, default="mace-off23-small")
    parser.add_argument("--N_train",                  type=int,   help="Number of datapoints in training sets",      required=False, default=1000)
    parser.add_argument("--N_test",                   type=int,   help="Number of datapoints in testing sets",       required=False, default=1000)
    parser.add_argument("--N_conformers",             type=int,   help="Number of Starting Conformers",              required=False, default=10)
    parser.add_argument("--N_iterations",             type=int,   help="Number of ML Iterations Performed",          required=False, default=5)
    parser.add_argument("--MD_stepsize",              type=int,   help="Number of Time Steps Between MD Snapshots",  required=False, default=10)
    parser.add_argument("--MD_startup",               type=int,   help="Number of Time Steps Ignored",               required=False, default=100)
    parser.add_argument("--MD_temperature",           type=int,   help="Temperature in Kelvin",                      required=False, default=500)
    parser.add_argument("--MD_dt",                    type=float, help="MD Stepsize in femtoseconds",                required=False, default=1.0)
    parser.add_argument("--MD_energy_lower_cutoff",   type=float, help="Lower bound for the energy cutoff function", required=False, default=1.0)
    parser.add_argument("--MD_energy_upper_cutoff",   type=float, help="Upper bound for the energy cutoff function", required=False, default=10.0)
    parser.add_argument("--Cluster_tolerance",        type=float, help="Tolerance used in the RMSD clustering",      required=False, default=0.075)
    parser.add_argument("--Cluster_Parallel",         type=int,   help="MPI nodes used in the RMSD clustering",      required=False, default=1)
    parser.add_argument("--data",                     type=str,   help="Location of pre-calculated data set",        required=False, default="train_data")
    parser.add_argument("--modSem_finite_step",       type=float, help="Finite Step to Calculate Hessian in Ang",    required=False, default=0.005291772)
    parser.add_argument("--modSem_vib_scaling",       type=float, help="Vibrational Scaling Parameter",              required=False, default=0.957)
    parser.add_argument("--modSem_tolerance",         type=float, help="Tolerance for the geometry optimizer",       required=False, default=0.0001)
    parser.add_argument('--memory',    action='store_true', help="Retain data upon iteration (Default)")
    parser.add_argument('--no-memory', dest='memory', action='store_false', help="Don't retain data upon iteration")
    parser.set_defaults(memory=True)
    parser.add_argument('--linear_harmonics',    action='store_true', help="Linearize the Harmonic potentials in the Force Field (Default)")
    parser.add_argument('--no-linear_harmonics', dest='linear_harmonics', action='store_false', help="Don't Linearize the Harmonic potentials in the Force Field")
    parser.set_defaults(linear_harmonics=True)
    parser.add_argument('--linear_torsions',    action='store_true', help="Linearize the Torsion potentials in the Force Field (Default)")
    parser.add_argument('--no-linear_torsions', dest='linear_torsions', action='store_false', help="Don't Linearize the Torsion potentials in the Force Field")
    parser.set_defaults(linear_torsions=True)
    parser.add_argument('--modSem',    action='store_true', help="Use mod-Seminario method to initialize the Force Field (Default)")
    parser.add_argument('--no-modSem', dest='modSem', action='store_false', help="Don't use mod-Seminario method to initialize the Force Field")
    parser.set_defaults(modSem=True)
    args = parser.parse_args()
    world_size = 1
    main(world_size, args)

