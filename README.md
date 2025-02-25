# Bespokefit2
Generate a Force-Field Parameterization from ML-Potential MD
#### **Newcastle University UK - Cole Group**
---
## Table of Contents

* [What is BespokeFit2?]
* [Installation]
* [Running]

## What is BespokeFit2? 
BespokeFit2 is a Force-Field parameterization tool. For a given molecule, it will generate a data set of conformers using machine learning models in [OpenMM-ML](https://github.com/openmm/openmm-ml) simulations. This dataset is used to minimize the Force-Field parameterization. 

---
From a SMILES string, we generate a initial parameterization using a default open-ff force field - and optionally, adding in modified-Seminario derived bond and angle force constants. This is used to generate a dataset of conformers by running either ML-Potential MD of Force-Field MD and grabbing a number of snapshots. For every snapshot, the energies and forces are taken using the ML-Potental. 

This dataset is used to minimize the given force field parameters using the ADAM stochastic optimization method, where the loss function is the squared difference between energies and forces for the conformer dataset predicted by the force-field parameterization and the stored values calculated with the ML-potential. 

After a given number of epochs, the new parameterization is stored. The new force-field is used to generate another set of MD snapshots, which are used in the same way to further optimize the force field. This continues for a given number of iterations, where the relative reduction is error is tracked. The number of iterations should be increased upto convergence.

---
Four methods for generating the initial dataset are implemented:

 1 - "DATA" : Read the dataset from a file
 
 2 - "MLMD" : Run the ML-Potential MD to get the snapshots. This is the most expensive option.
 
 3 - "MMMD" : Run Force-Field MD using the initial guess to generate the snapshots. Then use the ML-Potential to generate energies and forces
 
 4 - "cMMMD" : Run Force-Field MD using the initial guess to generate the snapshots. Cluster the snapshots with respect to their pairwise RMSD and then use the ML-Potential to generate energies and forces

## Installation

The easiest way to install BespokeFit2 is with conda:
```
   git clone https://github.com/thomasjamespope/Bespokefit2.git
   cd bespokefit2
```
## Running
| Input parameter | Varaible | Default | Description |
| --- | --- | --- | --- |
| `--smiles` | *str* | None | SMILES string of the molecule |
| `--method` | *str* | MMMD | Method for generating data: (DATA,MLMD,MMMD,cMMMD) |
| `--N_epochs` | *int* | 1000 | Number of epochs in the ML fit |
| `--learning_rate` | *float* | 0.1 | Learning Rate in the ML fit |
| `--learning_rate_decay` | *float* | 0.99 | Learning Rate Decay |
| `--learning_rate_decay_step` | *int* | 10 | Learning Rate Decay Step |
| `--loss_force_weight` | *float* | 1e5 | Scaling Factor for the Force loss term |

| `--force_field_init` | *str* | "openff-2.2.0.offxml" | Starting guess force field |
| `--MLMD_potential` | *str* | "mace-off23-small" | Name of the MD potential used |
| `--N_train` | *int* | 1000 | Number of datapoints in training set |
| `--N_test` | *int* | 1000 | Number of datapoints in test set |
| `--N_conformers` | *int* | 10 | Number of Starting Conformers |
| `--N_iterations` | *int* | 5 | Number of ML Iterations Performed |
| `--MD_stepsize` | *int* | 10 | Number of Time Steps Between MD Snapshots |
| `--MD_startup` | *int* | 100 | Number of Time Steps Ignored |
| `--MD_temperature` | *int* | 500 | Temperature in Kelvin |
| `--MD_dt` | *float* | 1.0 | MD Stepsize in femtoseconds |
| `--MD_energy_lower_cutoff` | *float* | 1.0 | Lower bound for the energy cutoff function in kcal/mol |
| `--MD_energy_upper_cutoff` | *float* | 10.0 | Upper bound for the energy cutoff function in kcal/mol |
| `--Cluster_tolerance` | *float* | 0.075 | Tolerance used in the RMSD clustering |
| `--Cluster_Parallel` | *int* | 1 | MPI nodes used in the RMSD clustering |
| `--data` | *str* | "train_data" | Location of pre-calculated data set |
| `--modSem_finite_step` | *float* | 0.005291772 | Finite Step to Calculate Hessian in Ang |
| `--modSem_vib_scaling` | *float* | 0.957 | Vibrational Scaling Parameter |
| `--modSem_tolerance` | *float* | 0.0001 | Tolerance for the geometry optimizer |
