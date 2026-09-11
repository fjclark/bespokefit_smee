# Run with containers

A prebuilt GPU image is published to the GitHub Container Registry, so you can run `presto` without pixi and
without a dependency solve. This page covers Docker on a workstation and Apptainer on HPC.

If pixi already works for you, keep using it. Containers are worth it when you want a fixed, citable
environment, when you are handing `presto` to a collaborator, or when you are on a cluster that will not let
you install a 9 GB conda environment.

## What is and is not in the image

The image contains the whole `default` pixi environment: `presto`, OpenMM, PyTorch, and every supported MLP,
along with the CUDA **userspace libraries** from conda-forge. It does **not** contain the NVIDIA **kernel
driver**, which stays on the host and is handed to the container by `--gpus all`. So the image is portable
between machines, but the host still needs a driver supporting **CUDA >= 12.9**, exactly as the
[pixi install](installation.md) does.

The default AIMNet2 weights are baked in, so a default fit runs with no network access. All other MLP weights
download on first use into a cache volume (see [below](#the-cache-volume)).

## Prerequisites

- [Docker Engine](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- An NVIDIA driver supporting CUDA >= 12.9 (check with `nvidia-smi`)

Verify all three at once:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi
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
| `--gpus all` | Hand the host GPUs and driver to the container. Without it, `presto` fails with "CUDA is not available on this system." |
| `--shm-size=1g` | Docker caps `/dev/shm` at 64 MB, which is too small for `n_sampling_processes > 1`. Harmless otherwise. |
| `--user "$(id -u):$(id -g)"` | Run as you, so output files are yours and not root-owned. |
| `-v "$PWD:/work"` | Expose the current directory as the container's working directory. |
| `-v presto-cache:/cache` | A persistent volume for downloaded MLP weights. |
| `ghcr.io/cole-group/presto:latest` | The image. Pin an exact version for real work. |
| `presto train ...` | Everything after the image name is the command run inside. |

Because `/work` is your current directory, output lands exactly where a native run would put it:
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
docker run --rm -it --gpus all -v "$PWD:/work" -v presto-cache:/cache \
  --entrypoint bash ghcr.io/cole-group/presto:latest
```

## The cache volume

`presto-cache` is a Docker-managed volume mounted at `/cache`. MACE-OFF, Orb-v3 and AceFF weights land there
on first use, along with the HuggingFace, torch and matplotlib caches. Reusing it between runs avoids
re-downloading several GB. Inspect or clear it with:

```bash
docker volume inspect presto-cache
docker volume rm presto-cache
```

AIMNet2 is the exception: its weights live inside the image already.

## Build the image yourself

From a clone of the repository:

```bash
pixi run container-build
```

Expect 20-60 minutes for a cold build and roughly 12 GB on disk. The task passes `PRESTO_VERSION` from
`git describe`, which is needed because `.git` is excluded from the build context and the version therefore
cannot be derived from the repository inside the build.

Two more tasks wrap the run commands. `container-run` appends whatever you pass after it:

```bash
pixi run container-run presto train --param-settings.molecules "CCO"
pixi run container-shell   # interactive bash inside the image
```

They build on `presto:local`, so run `container-build` first.

## Apptainer on HPC

Most clusters forbid Docker but provide [Apptainer](https://apptainer.org/), which converts the same image:

```bash
apptainer pull presto.sif docker://ghcr.io/cole-group/presto:latest

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

The `.sif` is a single file, which is easier on a shared filesystem than a 9 GB conda environment. Keep the
cache bind on scratch rather than in the image.

Podman users can substitute `podman` for `docker` throughout, replacing `--gpus all` with
`--device nvidia.com/gpu=all`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `CUDA is not available on this system.` | `--gpus all` was omitted, or the NVIDIA Container Toolkit is not installed. |
| `could not select device driver "" with capabilities: [[gpu]]` | The NVIDIA Container Toolkit is missing or the Docker daemon was not restarted after installing it. |
| `nvidia-smi` works inside the container but `torch.cuda.is_available()` is `False` | The container toolkit did not inject `libcuda.so.1`. Check `NVIDIA_DRIVER_CAPABILITIES` includes `compute`. |
| CUDA errors mentioning the driver version | The host driver is older than CUDA 12.9. This is the same requirement as a native install; the container cannot work around it. |
| Output files owned by `root` | Add `--user "$(id -u):$(id -g)"`. |
| Bus errors or worker crashes with `n_sampling_processes > 1` | `/dev/shm` is too small. Raise `--shm-size`, or use `--ipc=host`. |
| Odd import errors under Apptainer | Add `--cleanenv`. |
| `no space left on device` while pulling | The image is several GB compressed and around 12 GB unpacked. Check the Docker data root. |

## Licensing

The image ships no MLP weights except the bundled Egret-1 model (MIT) and AIMNet2 (MIT). In particular it does
**not** ship MACE-OFF weights, which are under the
[Academic Software License](https://github.com/gabor1/ASL/blob/main/ASL.md) and **do not permit commercial
use**. Selecting a MACE-OFF model downloads those weights at run time, under that licence. See
[Choose an MLP](../how-to/choose-an-mlp.md).
