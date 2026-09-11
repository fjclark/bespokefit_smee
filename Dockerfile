# syntax=docker/dockerfile:1
#
# GPU image for presto.
#
# Stage 1 installs the pixi `default` environment and renders it down to a plain
# bash activation script. Stage 2 receives only the finished environment, so the
# runtime image carries no pixi and no package cache.
#
# The image ships the CUDA userspace libraries (from conda-forge, inside the pixi
# environment) but not the NVIDIA kernel driver, which is injected at run time by
# `docker run --gpus all`. The host driver must still support CUDA >= 12.9.
#
# See docs/get-started/containers.md for how to build and run this.

ARG PIXI_VERSION=0.67.2
ARG PRESTO_ENV=default

FROM ghcr.io/prefix-dev/pixi:${PIXI_VERSION}-noble AS build

ARG PRESTO_ENV
ARG PRESTO_VERSION=0.0.0.dev0

WORKDIR /app

# The pixi image ships no git, but the `orb` feature pulls cached-path from a git
# URL and uv shells out to the git binary to fetch it. Build stage only, so this
# does not reach the runtime image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# A build machine has no NVIDIA driver, so the `__cuda` virtual package that the
# CUDA pytorch builds depend on is absent and pixi would refuse the cuda129
# system-requirement. Assert it for the build only.
ENV CONDA_OVERRIDE_CUDA=12.9
# hatch-vcs reads the version from git, and .git is excluded from the build context.
# This must be the global variable: hatch-vcs does not pass a dist name through to
# setuptools-scm, so the scoped SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PRESTO is ignored
# and the build fails with "setuptools-scm was unable to detect version".
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${PRESTO_VERSION}

COPY pyproject.toml pixi.lock README.md LICENSE ./
COPY presto ./presto

# --locked fails loudly if pixi.lock is out of date with pyproject.toml.
RUN --mount=type=cache,target=/root/.cache/rattler \
    pixi install --locked --environment "${PRESTO_ENV}"

# Render the environment into a standalone bash prologue so the runtime image
# does not need pixi.
RUN printf '#!/bin/bash\n' > /entrypoint.sh \
 && pixi shell-hook --environment "${PRESTO_ENV}" -s bash >> /entrypoint.sh \
 && printf '\nexec "$@"\n' >> /entrypoint.sh \
 && chmod 0755 /entrypoint.sh

# Warm the default MLP. AIMNet2 writes its weights inside site-packages and has no
# cache-directory override, so baking them here is what lets a non-root container
# run `presto train` with the default settings, and offline.
RUN /entrypoint.sh python -c "\
from openff.toolkit import Molecule; \
from openmmml import MLPotential; \
m = Molecule.from_smiles('CCO'); m.generate_conformers(n_conformers=1); \
MLPotential('aimnet2').createSystem(m.to_topology().to_openmm(), charge=0, device='cpu')"

# -----------------------------------------------------------------------------
# The `base` CUDA tag carries no CUDA toolkit (that comes from conda-forge inside
# the pixi environment). We use it rather than plain ubuntu because it sets
# NVIDIA_DRIVER_CAPABILITIES=compute,utility. The container toolkit otherwise
# defaults to `utility` alone, which injects nvidia-smi but not libcuda.so.1, and
# torch.cuda.is_available() then returns False even though nvidia-smi works.
FROM nvidia/cuda:12.9.1-base-ubuntu24.04 AS runtime

ARG PRESTO_ENV

LABEL org.opencontainers.image.source="https://github.com/cole-group/presto" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.title="presto" \
      org.opencontainers.image.description="Fast parameterisation of bespoke SMIRNOFF force fields using MLPs (GPU)"

# The prefix must match the build stage exactly: conda environments embed absolute
# paths in their binaries and scripts, so they cannot be relocated.
COPY --from=build /app/.pixi/envs/${PRESTO_ENV} /app/.pixi/envs/${PRESTO_ENV}
# presto is installed as an editable path dependency, so site-packages holds a .pth
# file pointing at /app rather than a copy of the code. The source has to be here.
COPY --from=build /app/presto /app/presto
COPY --from=build /app/pyproject.toml /app/README.md /app/LICENSE /app/
COPY --from=build --chmod=0755 /entrypoint.sh /usr/local/bin/entrypoint.sh

# Every MLP except AIMNet2 downloads its weights on first use. Point all of the
# caches at /cache so a single mounted volume persists them between runs.
ENV XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    CACHED_PATH_CACHE_ROOT=/cache/cached_path \
    MPLCONFIGDIR=/cache/matplotlib \
    HOME=/tmp

# Mode 0777 so that `--user $(id -u):$(id -g)` works for any host UID.
RUN mkdir -p /cache /work && chmod 0777 /cache /work

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["presto", "--help"]
