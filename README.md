
# Bespokefit_smee

Generate a Bespoke Force-Field Parametrization Quickly and Reliably 

#### **Newcastle University UK - Cole Group**


## Table of Contents

* [What is Bespokefit_smee?](https://github.com/thomasjamespope/bespokefit_smee?tab=readme-ov-file#what-is-bespokefit_smee)
    * [In-Detail](https://github.com/thomasjamespope/bespokefit_smee?tab=readme-ov-file#In-Detail)
* [Installation](https://github.com/thomasjamespope/bespokefit_smee?tab=readme-ov-file#installation)
* [Running](https://github.com/thomasjamespope/bespokefit_smee?tab=readme-ov-file#running)



## What is Bespokefit_smee? 

Bespokefit_smee is a Force-Field parametrization tool. For a given molecule, it will generate a data set of conformers using machine learning models in [OpenMM-ML](https://github.com/openmm/openmm-ml) simulations. This dataset is used to minimize the Force-Field parametrization. 

### In Detail

From a SMILES string, we generate a initial parametrization using a default open-ff force field - and optionally, adding in modified-Seminario derived bond and angle force constants. This is used to generate a dataset of conformers by running either ML-Potential MD of Force-Field MD and grabbing a number of snapshots. For every snapshot, the energies and forces are taken using the ML-Potental. 

This dataset is used to minimize the given force field parameters using the ADAM stochastic optimization method, where the loss function is the squared difference between energies and forces for the conformer dataset predicted by the force-field parametrization and the stored values calculated with the ML-potential. 

After a given number of epochs, the new parametrization is stored. The new force-field is used to generate another set of MD snapshots, which are used in the same way to further optimize the force field. This continues for a given number of iterations, where the relative reduction is error is tracked. The number of iterations should be increased up to convergence.

---
Four methods for generating the initial dataset are implemented:

 1 - "DATA" : Read the dataset from a file
 
 2 - "MLMD" : Run the ML-Potential MD to get the snapshots. This is the most expensive option.
 
 3 - "MMMD" : Run Force-Field MD using the initial guess to generate the snapshots. Then use the ML-Potential to generate energies and forces
 
 4 - "cMMMD" : Run Force-Field MD using the initial guess to generate the snapshots. Cluster the snapshots with respect to their pairwise RMSD and then use the ML-Potential to generate energies and forces

---
The functional form of the force-field is as follows:

- Bonds and angles are defined by a harmonic function, 
$u(x;k,x_0)=\frac{k}{2}\left(x-x_0\right)^2$, 
where the position of the minimum, $x_0$, and the magnitude, $k$, are the fitting parameters. 
- Proper and improper torsions are defined by a set of cosine functions, 
$u_p(\phi;k,\phi_0)=k\left(1+\cos{\left(p\phi-\phi_0\right)}\right)$, 
where the phase, $\phi_0$, and the magnitude, $k$, are the fitted parameters. Here, proper torsions are expanded to include four periodicities, whereas improper torsions include only one. It is also noted that for symmetry, the phase $\phi_0$ is expected to be either 0 or $\pi$

To stabilize and speed up convergence of the parameter fitting, these potentials are *linearized*. 

The linearization of the harmonic terms followed the approach by [espaloma](https://doi.org/10.1039/D2SC02739A), where the minimum is assumed to be within a window given by $x_1$ and $x_2$, such that the fitting parameters may by remapped onto linear terms,

$$k_1=k\frac{x_2-x_0}{x_2-x_1} \quad\text{and}\quad k_2=k\frac{x_0-x_1}{x_2-x_1}$$

These terms give the original parameters via,

$$k=k_1+k_2 \quad\text{and}\quad x_0=\frac{k_1x_1+k_2x_2}{k_1+k_2}$$

Crucially, the gradient along $k_1$ and $k_2$ behaves more reliably and so the parameters minimize faster.  

In a similar way, the cosine functions are linearized by defining a *phase window* of 0 to $\pi$, such that the parameters may be mapped onto,

$$k_0=\frac{k}{2}\left(1+\cos{\phi_0}\right) \quad\text{and}\quad k_{\pi}=\frac{k}{2}\left(1-\cos{\phi_0}\right)$$

which yield the original parameters via,

$$k=k_0+k_{\pi} \quad\text{and}\quad \cos{\phi_0}=\frac{k_0-k_{\pi}}{k_0+k_{\pi}}$$

Again, the gradient along $k_0$ and $k_{\pi}$ is more reliable and the parametrization proceed faster. 

## Installation

The easiest way to install Bespokefit_smee is with conda:

```
   git clone https://github.com/thomasjamespope/bespokefit_smee.git
   cd bespokefit_smee
   mamba env create -n bespokefit_smee --file enviroment.yaml
   mamba activate bespokefit_smee
   pip install mace-torch
```
In addition, an updated version on the package [smee](https://github.com/SimonBoothroyd/smee) is required:
```
   mamba uninstall smee
   mamba install pydantic-units msgpack-python nnpops
   git clone https://github.com/thomasjamespope/smee
   cd smee
   pip install .
```

## Running

Running Bespokefit_smee is easy. Simply determine the SMILES string of your molecule are run:
```
    python bespokefit_smee.py --smiles SMILES [options]
```
where a full list of options follows. Note, only the SMILES string is required. The other options have reasonable defaults.

| Input parameter | Variable | Default | Description |
| --- | --- | --- | --- |
| `--smiles` | *str* | None | SMILES string of the molecule |
| `--method` | *str* | MMMD | Method for generating data: (DATA,MLMD,MMMD,cMMMD) |
| `--N_epochs` | *int* | 1000 | Number of epochs in the ML fit |
| `--learning_rate` | *float* | 0.1 | Learning Rate in the ML fit |
| `--learning_rate_decay` | *float* | 0.99 | Learning Rate Decay |
| `--learning_rate_decay_step` | *int* | 10 | Learning Rate Decay Step |
| `--loss_force_weight` | *float* | 1e5 | Scaling Factor for the Force loss term |
| `--force_field_init` | *str* | openff-2.2.0.offxml | Starting guess force field |
| `--modSem` <br/> `--no-modSem` | | `--modSem` | Use mod-Seminario method to initialize the Force Field |
| `--linear_harmonics` <br/> `--no-linear_harmonics` | | `--linear_harmonics` | Linearize the Harmonic potentials in the Force Field |
| `--linear_torsions` <br/> `--no-linear_torsions` | | `--linear_torsions` | Linearize the Torsion potentials in the Force Field |
| `--MLMD_potential` | *str* | mace-off23-small | Name of the MD potential used |
| `--N_train` | *int* | 1000 | Number of data-points in training set |
| `--N_test` | *int* | 1000 | Number of data-points in test set |
| `--N_conformers` | *int* | 10 | Number of Starting Conformers |
| `--N_iterations` | *int* | 5 | Number of ML Iterations Performed |
| `--memory` <br/> `--no-memory` | | `--memory` | Retain data upon iteration |
| `--MD_stepsize` | *int* | 10 | Number of Time Steps Between MD Snapshots |
| `--MD_startup` | *int* | 100 | Number of Time Steps Ignored |
| `--MD_temperature` | *int* | 500 | Temperature in Kelvin |
| `--MD_dt` | *float* | 1.0 | MD Stepsize in femtoseconds |
| `--MD_energy_lower_cutoff` | *float* | 1.0 | Lower bound for the energy cut-off function in kcal/mol |
| `--MD_energy_upper_cutoff` | *float* | 10.0 | Upper bound for the energy cut-off function in kcal/mol |
| `--Cluster_tolerance` | *float* | 0.075 | Tolerance used in the RMSD clustering |
| `--Cluster_Parallel` | *int* | 1 | MPI nodes used in the RMSD clustering |
| `--data` | *str* | train_data | Location of pre-calculated data set |
| `--modSem_finite_step` | *float* | 0.005291772 | Finite Step to Calculate Hessian in Ang |
| `--modSem_vib_scaling` | *float* | 0.957 | Vibrational Scaling Parameter |
| `--modSem_tolerance` | *float* | 0.0001 | Tolerance for the geometry optimizer |
