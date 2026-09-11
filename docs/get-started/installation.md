# Installation

The easiest way to install `presto` is from the GitHub repo using [pixi](https://pixi.sh/), which is the recommended approach. This creates environments containing dependencies for several MLPs. Alternatively, `presto` is available on `conda-forge` as `presto-fit` — see [Install from conda-forge](#install-from-conda-forge) below, but comes without MLP dependencies.

If you would rather not install anything, a prebuilt GPU container image is also published. See **[Run with containers](containers.md)** for Docker on a workstation and Apptainer on HPC.

## Prerequisites

- **pixi** — see [pixi installation docs](https://pixi.sh/latest/).
- **NVIDIA driver compatible with CUDA 12.9 or newer** for the GPU environment. Check with `nvidia-smi`. Older driver versions are not usable because `presto` requires OpenMM 8.5 (for `PythonForce`), and OpenMM 8.5 requires CUDA 12.9.
- A CPU-only environment is available for development and docs builds but is very slow for real fits.

## Install from GitHub with pixi

```bash
git clone https://github.com/cole-group/presto.git
cd presto
pixi shell
```

This creates the default GPU environment (`gpu-py313-cuda129`) and activates a shell in it. For more on activating pixi environments, see [the pixi shell docs](https://pixi.sh/latest/advanced/pixi_shell/).

## Install from conda-forge

`presto` is published on `conda-forge` as `presto-fit`. This is an alternative to the recommended `pixi` install which may be useful for running `presto` in e.g. CI/ CD pipelines and including it as a dependency of other packages.

```bash
mamba env create -n presto -c conda-forge presto-fit "cuda-version==12.9.*"
mamba activate presto
pip install aimnet
```

The `conda-forge` `presto-fit` package comes **without the MLP dependencies**, so you must install the ones you need separately. The `pip install aimnet` step above adds the default AIMNet2 MLP; install other MLPs as required (see [How-to → Choose an MLP](../how-to/choose-an-mlp.md)). Adjust the `cuda-version` virtual package pin as required for your driver.

## Verify the install

```bash
presto version
```

If `presto version` prints a version string, you're ready to move on to **[Quickstart](quickstart.md)**.

## Optional environments

The default GPU environment includes every supported MLP feature. Defined in `pyproject.toml [tool.pixi.environments]`:

| Environment | Use when |
|---|---|
| `gpu-py313-cuda129` (default) | Normal GPU workflow on Python 3.13. |
| `gpu-py312-cuda129` | GPU workflow on Python 3.12.  |
| `cpu-py312`, `cpu-py313` | Tests, docs, or environments without a CUDA-capable GPU. |
| `base*` (No MLP features) | No MLP deps desired |

Activate a non-default environment with:

```bash
pixi shell -e cpu-py312
```

## MLP licensing notes

The `pixi` `presto` install from GitHub brings in several reference MLPs by default. Most have permissive licences but a few do not:

- **AIMNet2** (MIT) — current default; permits commercial use.
- **Orb-v3 (OMol25 variants)** (Apache-2.0) — permits commercial use.
- **Egret-1** (MIT) — permits commercial use.
- **AceFF-2.0** — permits commercial use, but [is currently broken upstream](https://github.com/openmm/openmm-ml/issues/136).
- **MACE-OFF** — released under the [Academic Software License](https://github.com/gabor1/ASL/blob/main/ASL.md). **Not for commercial use.**

For more, see **[Concepts → MLPs in presto](../concepts/mlps.md)** and **[How-to → Choose an MLP](../how-to/choose-an-mlp.md)**.

## Troubleshooting

If anything went wrong during install or first run, see **[Reference → Troubleshooting](../reference/troubleshooting.md)**.

For container-specific problems, see the troubleshooting table in **[Run with containers](containers.md#troubleshooting)**.
