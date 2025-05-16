"""Apply OpenFF parameters to molecule, cluster conformers by RMSD and train"""

import copy
import datetime
import pathlib
from contextlib import redirect_stderr

import datasets
import datasets.combine
import openff.toolkit
import tensorboardX
import torch
from tqdm import tqdm

from .data_maker import get_data_cMMMD, get_data_MLMD, get_data_MMMD
from .loss_functions import prediction_loss
from .parameterizer import build_parameters, convert_to_smirnoff
from .settings import TrainingConfig
from .writers import (
    open_writer,
    write_metrics,
    write_potential_comparison,
    write_scatter,
)


def train(world_size: int, args: TrainingConfig) -> None:
    smiles = args.smiles  # SMILES string
    device_type = args.device_type
    device = torch.device(device_type)
    method = args.method  # Method for generating data
    memory = args.memory  # Include previous data on iteration
    n_epochs = args.N_epochs  # Number of epochs in the ML fit
    linear_harmonics = (
        args.linear_harmonics
    )  # Flag to turn of linearized harmonic potentials
    linear_torsions = (
        args.linear_torsions
    )  # Flag to turn of linearized torsions potentials
    learning_rate = args.learning_rate  # Learning Rate in the ML fit
    learning_rate_decay = args.learning_rate_decay  # Learning Rate Decay Rate
    learning_rate_decay_step = args.learning_rate_decay_step  # Learning Rate Decay Step
    loss_force_weight = args.loss_force_weight  # Scaling Factor for the Force loss term
    ff_path = args.force_field_init  # Starting guess force field
    ML_path = args.MLMD_potential  # Name of the MLMD potential used
    MD_temperature = args.MD_temperature  # MD Temperature in Kelvin
    MD_dt = args.MD_dt  # MD Stepsize in picoseconds
    MD_stepsize = args.MD_stepsize  # Number of Steps Between MD Snapshot
    MD_startup = args.MD_startup  # Number of Steps Ignored
    MD_energy_lower_cutoff = (
        args.MD_energy_lower_cutoff
    )  # Lower bound for the energy cutoff smoothing function
    MD_energy_upper_cutoff = (
        args.MD_energy_upper_cutoff
    )  # Upper bound for the energy cutoff smoothing function
    Cluster_tolerance = args.Cluster_tolerance  # Tolerance used in the RMSD clustering
    Cluster_Parallel = args.Cluster_Parallel  # MPI nodes used in the RMSD clustering
    Ntrn = args.N_train  # Number of MD steps in training sets
    Ntst = args.N_test  # Number of MD steps in testing sets
    Ncnf = args.N_conformers  # Number of Starting Conformers
    Nits = args.N_iterations  # Number of ML Iterations performed
    source_train = (
        args.data
    )  # Location of pre-calculated data set: only used if method = "DATA"
    modSem = args.modSem  # Activate the modified Seminario Starting guess
    modSem_finite_step = (
        args.modSem_finite_step
    )  # Finite Step for the Hessian Calculation
    modSem_vib_scaling = args.modSem_vib_scaling  # Vibrational Scaling Parameter
    modSem_tolerance = args.modSem_tolerance  # Tolerance for the geometry optimize
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
    print(
        f"    MD_stepsize              :{MD_stepsize:>20} steps ({int(MD_stepsize * MD_dt):5d} fs)"
    )
    print(
        f"    MD_startup               :{MD_startup:>20} steps ({int(MD_startup * MD_dt):5d} fs)"
    )
    print(f"    MD_energy_lower_cutoff   :{MD_energy_lower_cutoff:>20} kcal/mole")
    print(f"    MD_energy_upper_cutoff   :{MD_energy_upper_cutoff:>20} kcal/mole")
    if method == "cMMMD":
        print(f"    Cluster_tolerance        :{Cluster_tolerance:>20}")
        print(f"    Cluster_Parallel         :{Cluster_Parallel:>20}")
    print(f"    N_train                  :{Ntrn:>20}")
    print(f"    N_test                   :{Ntst:>20}")
    print(f"    N_conformers             :{Ncnf:>20}")
    print(f"    N_iterations             :{Nits:>20}")
    if method == "DATA":
        print(f"    source_train             :{source_train:>20}")
    print(f"    modSem                   :{modSem!s:>20}")
    if modSem:
        print(f"    modSem_finite_step       :{modSem_finite_step:>20} ang")
        print(f"    modSem_vib_scaling       :{modSem_vib_scaling:>20}")
        print(f"    modSem_tolerance         :{modSem_tolerance:>20} kcal/(mole*ang)")
    print("")
    Ntrn, Ntst = int(Ntrn / Ncnf), int(Ntst / Ncnf)  #   Convert to "per-conformer"
    MD_dt = MD_dt / 1000  #   Convert to ps
    modSem_finite_step = modSem_finite_step / 10  #   Convert to nm

    # Save config to YAML file
    args.to_yaml()

    #   parameterize the molecule and output the forcefield to a file
    mol = openff.toolkit.Molecule.from_smiles(
        smiles, allow_undefined_stereo=True, hydrogens_are_explicit=False
    )
    VdW_forcefield = openff.toolkit.ForceField(ff_path)
    old_force_field, trainable, topology = build_parameters(
        mol,
        VdW_forcefield,
        ML_path,
        linear_harmonics,
        linear_torsions,
        modSem,
        modSem_finite_step,
        modSem_vib_scaling,
        modSem_tolerance,
        device_type=device_type,
    )

    # Move to the requested device (not strictly necessary if on CPU)
    trainable_parameters = trainable.to_values().to((device_type))
    topology = topology.to(device_type)

    off_force_field = convert_to_smirnoff(
        trainable.to_force_field(trainable_parameters), base=VdW_forcefield
    )
    off_force_field.to_file("default.offxml")

    # get the inital training data
    if method == "DATA":
        dataset = datasets.Dataset.load_from_disk(f"{source_train}")
    elif method == "MMMD":
        dataset = get_data_MMMD(
            mol,
            off_force_field,
            ML_path,
            MD_temperature,
            MD_dt,
            Ntrn,
            Ncnf,
            MD_stepsize,
            MD_startup,
            MD_energy_upper_cutoff,
            MD_energy_lower_cutoff,
        )
    elif method == "cMMMD":
        dataset = get_data_cMMMD(
            mol,
            off_force_field,
            ML_path,
            MD_temperature,
            MD_dt,
            Ntrn,
            Ncnf,
            MD_stepsize,
            MD_startup,
            MD_energy_upper_cutoff,
            MD_energy_lower_cutoff,
            Cluster_tolerance,
            Cluster_Parallel,
        )
    elif method == "MLMD":
        dataset = get_data_MLMD(
            mol,
            off_force_field,
            ML_path,
            MD_temperature,
            MD_dt,
            Ntrn,
            Ncnf,
            MD_stepsize,
            MD_startup,
            MD_energy_upper_cutoff,
            MD_energy_lower_cutoff,
        )
    else:
        print("ERROR:: Chosen method is not supported (method = ", method, ")")
        exit(0)
    with open("/dev/null", "w") as f:
        with redirect_stderr(f):
            dataset.save_to_disk("data_it_0")
    # Generate the test set
    dataset_test = get_data_MLMD(
        mol,
        off_force_field,
        ML_path,
        MD_temperature,
        MD_dt,
        Ntst,
        Ncnf,
        MD_stepsize,
        MD_startup,
        MD_energy_upper_cutoff,
        MD_energy_lower_cutoff,
    )
    # print("Generating Test Set with MLMD")
    # dataset_test = get_data_MLMD(mol,off_force_field,ML_path,MD_temperature,MD_dt,Ntrn,Ncnf,MD_stepsize,MD_startup,MD_energy_upper_cutoff,MD_energy_lower_cutoff)

    # Generate the Energy Scatter Plot
    energy_mean, energy_SD, forces_mean, forces_SD = write_scatter(
        dataset_test,
        trainable.to_force_field(trainable_parameters),
        topology,
        device.type,
        "default.scat",
    )
    for iteration in tqdm(
        range(Nits), leave=False, colour="magenta", desc="Iterating the Fit"
    ):
        # some book-keeping
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = pathlib.Path(f"{timestamp}")

        # run the ML training
        with open("training-" + str(iteration) + ".data", "w") as metrics_file:
            with open_writer(experiment_dir) as writer:
                optimizer = torch.optim.Adam(
                    [trainable_parameters], lr=learning_rate, amsgrad=True
                )
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=learning_rate_decay
                )
                for v in tensorboardX.writer.hparams(
                    {"optimizer": "Adam", "lr": learning_rate}, {}
                ):
                    writer.file_writer.add_summary(v)
                for i in tqdm(
                    range(n_epochs),
                    leave=False,
                    colour="blue",
                    desc="Running ML Optimization",
                ):
                    loss_trn = prediction_loss(
                        dataset,
                        trainable.to_force_field(trainable_parameters),
                        topology,
                        loss_force_weight,
                        device.type,
                    )
                    if i % 10 == 0:
                        loss_tst = prediction_loss(
                            dataset_test,
                            trainable.to_force_field(trainable_parameters),
                            topology,
                            loss_force_weight,
                            device.type,
                        )
                        write_metrics(i, loss_trn, loss_tst, writer, metrics_file)
                    loss_trn.backward(retain_graph=True)  # type: ignore[no-untyped-call]
                    # trainable.freeze_grad()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    trainable.clamp(trainable_parameters)
                    if i % learning_rate_decay_step == 0:
                        scheduler.step()
            # some book-keeping and outputing
            loss_tst = prediction_loss(
                dataset_test,
                trainable.to_force_field(trainable_parameters),
                topology,
                loss_force_weight,
                device.type,
            )
            write_metrics(n_epochs, loss_trn, loss_tst, writer, metrics_file)
        print(f"Summary for Iteration {iteration + 1}".center(88, "="))
        print("")
        print("Parameterization".center(88, "="))
        print("")
        print("")
        for potential_type in trainable._param_types:
            write_potential_comparison(
                old_force_field.potentials_by_type[potential_type],
                trainable.to_force_field(trainable_parameters).potentials_by_type[
                    potential_type
                ],
            )
            old_force_field.potentials_by_type[potential_type].parameters = copy.copy(
                trainable.to_force_field(trainable_parameters)
                .potentials_by_type[potential_type]
                .parameters
            )
        # Output the forcefield to a file
        off_force_field = convert_to_smirnoff(
            trainable.to_force_field(trainable_parameters), base=VdW_forcefield
        )
        off_force_field.to_file("trained-" + str(iteration) + ".offxml")
        # Generate the Energy Scatter Plot and summarize convergence
        energy_mean_new, energy_SD_new, forces_mean_new, forces_SD_new = write_scatter(
            dataset_test,
            trainable.to_force_field(trainable_parameters),
            topology,
            device.type,
            "trained-" + str(iteration) + ".scat",
        )
        print("")
        print("")
        print("Convergence".center(88, "="))
        print("")
        print(
            f"    Energy Error (Mean): {energy_mean:10.3e}->{energy_mean_new:10.3e} : Change = {energy_mean_new - energy_mean:10.3e}"
        )
        print(
            f"                 (SD):   {energy_SD:10.3e}->{energy_SD_new:10.3e} : Change = {energy_SD_new - energy_SD:10.3e}"
        )
        print(
            f"    Forces Error (Mean): {forces_mean:10.3e}->{forces_mean_new:10.3e} : Change = {forces_mean_new - forces_mean:10.3e}"
        )
        print(
            f"                 (SD):   {forces_SD:10.3e}->{forces_SD_new:10.3e} : Change = {forces_SD_new - forces_SD:10.3e}"
        )
        print("")
        energy_mean, energy_SD = energy_mean_new, energy_SD_new
        forces_mean, forces_SD = forces_mean_new, forces_SD_new
        # get the next iteration of training data, unless the loop is finished
        if iteration + 1 < Nits:
            if memory:
                if method == "cMMMD":
                    dataset = datasets.combine.concatenate_datasets(
                        [
                            dataset,
                            get_data_cMMMD(
                                mol,
                                off_force_field,
                                ML_path,
                                MD_temperature,
                                MD_dt,
                                Ntrn,
                                Ncnf,
                                MD_stepsize,
                                MD_startup,
                                MD_energy_upper_cutoff,
                                MD_energy_lower_cutoff,
                                Cluster_tolerance,
                                Cluster_Parallel,
                            ),
                        ]
                    )
                else:
                    dataset = datasets.combine.concatenate_datasets(
                        [
                            dataset,
                            get_data_MMMD(
                                mol,
                                off_force_field,
                                ML_path,
                                MD_temperature,
                                MD_dt,
                                Ntrn,
                                Ncnf,
                                MD_stepsize,
                                MD_startup,
                                MD_energy_upper_cutoff,
                                MD_energy_lower_cutoff,
                            ),
                        ]
                    )
            else:
                if method == "cMMMD":
                    dataset = get_data_cMMMD(
                        mol,
                        off_force_field,
                        ML_path,
                        MD_temperature,
                        MD_dt,
                        Ntrn,
                        Ncnf,
                        MD_stepsize,
                        MD_startup,
                        MD_energy_upper_cutoff,
                        MD_energy_lower_cutoff,
                        Cluster_tolerance,
                        Cluster_Parallel,
                    )
                else:
                    dataset = get_data_MMMD(
                        mol,
                        off_force_field,
                        ML_path,
                        MD_temperature,
                        MD_dt,
                        Ntrn,
                        Ncnf,
                        MD_stepsize,
                        MD_startup,
                        MD_energy_upper_cutoff,
                        MD_energy_lower_cutoff,
                    )
            with open("/dev/null", "w") as f:
                with redirect_stderr(f):
                    dataset.save_to_disk("data_it_" + str(iteration))
