<h2 align="center">presto</h2>

<p align="center"> Parameter Refinement Engine for Smirnoff Training / Optimisation</p>

<p align="center">
  <a href="https://github.com/cole-group/presto/actions/workflows/ci.yaml">
    <img src="https://github.com/cole-group/presto/actions/workflows/ci.yaml/badge.svg" alt="CI" />
  </a>
  <a href="https://codecov.io/gh/cole-group/presto" >
    <img src="https://codecov.io/gh/cole-group/presto/graph/badge.svg?token=IBZ2H0NL58"/>
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="license" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" />
  </a>
  <a href="https://mypy-lang.org/">
    <img alt="Checked with mypy" src="https://www.mypy-lang.org/static/mypy_badge.svg" />
  </a>
</p>

---

Train bespoke SMIRNOFF force fields quickly using a machine learning potential (MLP). All valence parameters (bonds, angles, proper torsions, and improper torsions) are trained to MLP energies sampled using molecular dynamics. Please see the [**documentation**](https://cole-group.github.io/presto/latest/).

***Warning**: ⚠️ This code is under active development and the API may change without notice.*

Please note that the MACE-OFF models are released under the [Academic Software License](https://github.com/gabor1/ASL/blob/main/ASL.md) which **does not permit commercial use**. However, the default AIMNet-2 model (as well as Egret-1, Orb-v3 OMOL, and others) does.

## Installation

Ensuring that you have pixi installed, install and start a shell with the current environment with:
```bash
git clone https://github.com/cole-group/presto.git
cd presto
pixi shell
```
This will create an environment with CUDA 12.9. **You'll need to update to CUDA >= 12.9 (check with `nvidia-smi`) to use `presto`** (older versions are not usable as we require OpenMM 8.5 for the PythonForce class, and this requires CUDA 12.9).

For more information on activating pixi environments, see [the documentation](https://pixi.sh/latest/advanced/pixi_shell/#traditional-conda-activate-like-activation).

`presto` is also available on `conda-forge` as `presto-fit`, but note that this comes without the MLP dependencies (install these separately, e.g. `pip install aimnet`). See the [installation docs](https://cole-group.github.io/presto/get-started/installation/) for details.

### Docker

A prebuilt GPU image is published to the GitHub Container Registry, so you can run `presto` without pixi:

```bash
docker run --rm --gpus all --shm-size=1g --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" -v presto-cache:/cache \
  ghcr.io/cole-group/presto:latest \
  presto train --param-settings.molecules "CCO"
```

The host still needs an NVIDIA driver supporting CUDA >= 12.9. See the [container docs](https://cole-group.github.io/presto/latest/get-started/containers/) for Apptainer on HPC and for troubleshooting.

## Usage

Run with command line arguments:
```bash
presto train --param-settings.molecules "CCC(CC)C(=O)Nc2cc(NC(=O)c1c(Cl)cccc1Cl)ccn2"
```
then see the bespoke force field at `training_iteration_2/bespoke_ff.offxml`.

Sensible defaults have been set, but all available options can be viewed with:
```bash
presto train --help
```

Run from a yaml file:
```bash
presto write-default-yaml default.yaml
# Modify the yaml to set the desired input_type and molecule input(s)
presto train-from-yaml default.yaml
```

For SDF inputs, set `molecule_input_type: sdf` and list one or more `.sdf` files under `molecules`. Each SDF may contain one or more molecules. For example, with the CLI:
```bash
presto train --param-settings.molecule-input-type sdf --param-settings.molecules input_molecule.sdf
```

For more details on the theory and implementation, please see the [documentation](https://cole-group.github.io/presto/latest/).


## Citation

If you use `presto` in your work, please cite:

> Clark, F.; Pope, T.; Maier, S.; Boothroyd, S.; Horton, J. T.; Ryczko, K.; Bortolato, A.; Cole, D. J. *Fast Training of Bespoke SMIRNOFF-format Molecular Mechanics Force Fields Using Machine Learning Potentials.* ChemRxiv **2026**. [doi:10.26434/chemrxiv.15004169/v2](https://doi.org/10.26434/chemrxiv.15004169/v2)

Because `presto` builds on the Open Force Field ecosystem, please also cite the relevant OpenFF publications listed at [openforcefield.org/science/how-to-cite](https://openforcefield.org/science/how-to-cite/).

A `CITATION.cff` file is provided in the repository, and GitHub will generate formatted citations (including BibTeX) from the *Cite this repository* link on the project page.

## Copyright

Copyright (c) 2025-2026, Finlay Clark, Newcastle University, UK

Copyright (c) 2025-2026, Thomas James Pope, Newcastle University, UK


This package includes models from other projects under the MIT license. See `presto/models/LICENSES.md` for details.

## Acknowledgements

Early development was completed by Thomas James Pope. Many ideas taken from Simon Boothroyd's super helpful [python-template](https://github.com/SimonBoothroyd/python-template).
