# Run with containers

A prebuilt GPU image is published to the GitHub Container Registry; this page covers Docker on a workstation and Apptainer on HPC.

## What is and is not in the image

The image contains the whole `default` pixi GPU environment: `presto`, OpenMM, PyTorch, and every supported MLP, but the host still needs a driver supporting **CUDA >= 12.9**, exactly as the [pixi install](installation.md) does.

## Prerequisites

- An NVIDIA driver supporting CUDA >= 12.9 (check with `nvidia-smi`)
- The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Docker Engine](https://docs.docker.com/engine/install/)

Verify the whole chain before going further:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi
```

and check that prints your GPU.

## Get the image

```bash
docker pull ghcr.io/cole-group/presto:latest
```

## Run a fit

```bash
docker run --rm --gpus all --shm-size=1g \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" \
  -v presto-cache:/cache \
  ghcr.io/cole-group/presto:latest \
  presto train --param-settings.molecules "CCO"
```

| Part | Why |
|---|---|
| `--rm` | Delete the container when it exits. Anything not on a mount is discarded. |
| `--gpus all` | Hand the host GPUs and driver to the container. |
| `--shm-size=1g` | Docker caps `/dev/shm` at 64 MB, which may be too small for `n_sampling_processes > 1`. |
| `--user "$(id -u):$(id -g)"` | Run as you, so output files are yours and not root-owned. |
| `-v "$PWD:/work"` | Expose the current directory as the container's working directory. |
| `-v presto-cache:/cache` | A persistent volume for downloaded MLP weights. |
| `ghcr.io/cole-group/presto:latest` | The image. Pin an exact version for real work. |
| `presto train ...` | Everything after the image name is the command run inside. |

Because `/work` is your current directory, output goes exactly where a native run would put it:
`training_iteration_2/bespoke_ff.offxml` and the rest of the
[output layout](../concepts/output-layout.md).

Running from a YAML settings file works the same way, since the file is visible through the mount:

```bash
docker run --rm --gpus all --shm-size=1g --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" -v presto-cache:/cache \
  ghcr.io/cole-group/presto:latest \
  presto train-from-yaml workflow_settings.yaml
```

For an interactive poke around, override the command with a shell:

```bash
docker run --rm -it --gpus all --shm-size=1g --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" -v presto-cache:/cache \
  ghcr.io/cole-group/presto:latest bash
```

## The cache volume

`presto-cache` is a Docker-managed volume mounted at `/cache`. MACE-OFF, Orb-v3 and AceFF weights land there
on first use, along with the HuggingFace, torch and matplotlib caches. Reusing it between runs avoids
re-downloading several GB. Inspect or clear it with:

```bash
docker volume inspect presto-cache
docker volume rm presto-cache
```

Some supported MLP backends load checkpoints with PyTorch's `weights_only=False`, so on shared systems
the cache must be private to you and not writable by other users. AIMNet2 is the exception to the volume:
its weights live inside the image already, so removing `presto-cache` does not remove them.

## Build the image yourself

From a clone of the repository:

```bash
pixi run container-build
```

Expect ~ 15 minutes for a cold build.

Two more tasks wrap the run commands. `container-run` appends whatever you pass after it:

```bash
pixi run container-run presto train --param-settings.molecules "CCO"
pixi run container-shell   # interactive bash inside the image
```

They build on `presto:local`, so run `container-build` first. Note that these won't work if you're not in the docker group (just copy the commands from `pyproject.toml` and add `sudo` in this case).

## Apptainer on HPC

Most clusters forbid Docker but provide [Apptainer](https://apptainer.org/), which converts the same image:

```bash
apptainer pull presto.sif docker://ghcr.io/cole-group/presto:latest

: "${SCRATCH:?Set SCRATCH to a writable scratch directory}"
mkdir -p "$SCRATCH/presto-cache"
chmod 700 "$SCRATCH/presto-cache"

# Preserve the GPUs assigned by Slurm; --cleanenv would otherwise discard this.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi

apptainer run --nv --cleanenv \
  --bind "$PWD:/work" \
  --bind "$SCRATCH/presto-cache:/cache" \
  presto.sif presto train --param-settings.molecules "CCO"
```

Three differences from Docker:

- `--nv` is Apptainer's `--gpus all`.
- Apptainer already runs as you, so there is no `--user` flag to pass.
- `--cleanenv` matters. Apptainer inherits the host environment by default, and a stray `PYTHONPATH`,
  `CONDA_PREFIX` or `LD_LIBRARY_PATH` from a `module load` will break the container.
- On Slurm, forward `CUDA_VISIBLE_DEVICES` as `APPTAINERENV_CUDA_VISIBLE_DEVICES` so that `--cleanenv`
  does not discard the scheduler's GPU assignment. The conditional in the example avoids hiding all GPUs
  when `CUDA_VISIBLE_DEVICES` is unset.

The `.sif` is a single file, which is easier on a shared filesystem than a 9 GB conda environment. Keep the
cache bind on scratch rather than in the image.

Podman users can substitute `podman` for `docker` throughout, replacing `--gpus all` with
`--device nvidia.com/gpu=all`.

## Licensing

The image ships no MLP weights except the bundled Egret-1 model (MIT) and AIMNet2 (MIT). In particular it does
**not** ship MACE-OFF weights, which are under the
[Academic Software License](https://github.com/gabor1/ASL/blob/main/ASL.md) and **do not permit commercial
use**. Selecting a MACE-OFF model downloads those weights at run time, under that licence. See
[Choose an MLP](../how-to/choose-an-mlp.md).
