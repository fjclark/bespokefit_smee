# Bespokefit2
Generate a Force-Field Parameterization from ML-Potential MD
#### **Newcastle University UK - Cole Group**
---
## Table of Contents

* [What is BespokeFit2?]([https://github.com/thomasjamespope/Bespokefit2#what-is-BespokeFit2])
* [Installation]([https://github.com/thomasjamespope/Bespokefit2#Installation)
    * [Requirements](https://github.com/qubekit/QUBEKit#requirements)
    * [Installing as Dev](https://github.com/qubekit/QUBEKit#installing-as-dev)
* [Help](https://github.com/qubekit/QUBEKit#help)

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

 
