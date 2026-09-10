"""Coordinate non-resumable, molecule-level sampling on one node."""

from __future__ import annotations

import multiprocessing
import os
import pickle
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import datasets
import loguru
import torch
from loguru import logger
from openff.toolkit import ForceField, Molecule
from rich.progress import track

from . import mlp
from . import sample as sample_module
from .outputs import OutputType
from .sample import _SAMPLING_FNS_REGISTRY
from .settings import PreComputedDatasetSettings, SamplingSettings
from .utils._suppress_output import suppress_unwanted_output
from .utils.gpu import free_gpu_memory

_WORKER_DEVICE = "cpu"
"""The torch device this worker samples on, set once it has claimed a GPU.
The parent never reads it."""

_WORKER_LOGS: list[tuple[str, str]] = []
"""Records logged while sampling one molecule, replayed by the parent in order.
Stays empty in the parent, where logging reaches the real sinks directly."""

_WARMED_POTENTIALS: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
"""Potentials this process has already pulled weights for; see `_warm_ml_potential`."""


def sampling_devices(device_type: str, n_workers: int) -> list[str]:
    """Return the CUDA_VISIBLE_DEVICES entry each worker should claim, or "cpu"."""
    if device_type != "cuda":
        return ["cpu"] * n_workers
    # Identities come from the env var, which may name UUIDs or MIG devices; the count
    # comes from torch, which reports only the leading entries CUDA actually accepted.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    n_gpus = max(1, torch.cuda.device_count())
    ids = visible.split(",")[:n_gpus] if visible else [str(i) for i in range(n_gpus)]
    return [ids[i % len(ids)] for i in range(n_workers)]


def _capture_log(message: loguru.Message) -> None:
    """Buffer one log record for the parent to replay."""
    _WORKER_LOGS.append((message.record["level"].name, message.record["message"]))


def _init_worker(devices: multiprocessing.Queue) -> None:  # type: ignore[type-arg]
    """Claim one device for this worker and buffer its logs for the parent."""
    global _WORKER_DEVICE
    claimed = devices.get()
    if claimed != "cpu":
        # Hide every other GPU so that OpenMM's DeviceIndex, torch's cuda:0, and any
        # backend which picks its own device all resolve to the claimed one. Must run
        # before torch initialises CUDA in this process: spawn imports alone do not,
        # but a future import calling torch.cuda.* would break this.
        os.environ["CUDA_VISIBLE_DEVICES"] = claimed
        claimed = "cuda:0"
    _WORKER_DEVICE = claimed
    suppress_unwanted_output()
    logger.remove()
    logger.add(_capture_log)
    # Workers share one terminal, so per-molecule bars would interleave. Replace the
    # progress-bar helper sample.py calls with a pass-through, so this worker renders
    # no bar of its own and the parent's ligand bar is the only one on screen.
    sample_module.track = lambda sequence, *args, **kwargs: sequence  # type: ignore[attr-defined]


def _sample_worker(
    molecule_json: str,
    molecule_index: int,
    offxml_path: str,
    device: str | None,
    sampling_settings: SamplingSettings,
    output_paths: dict[OutputType, Path],
) -> tuple[datasets.Dataset, list[tuple[str, str]]]:
    """Sample one molecule, naming its side outputs with its workflow index.

    Returns the dataset and any log records buffered by ``_init_worker``. The
    buffer is empty in the parent, where logging reaches the real sinks directly.
    """
    sample_module._MOL_INDEX_OFFSET = molecule_index  # type: ignore[attr-defined]
    _WORKER_LOGS.clear()
    try:
        dataset = _SAMPLING_FNS_REGISTRY[type(sampling_settings)](
            mols=[Molecule.from_json(molecule_json)],
            off_ff=ForceField(offxml_path),
            device=torch.device(device or _WORKER_DEVICE),
            settings=sampling_settings,
            output_paths=output_paths,
        )[0]
        return dataset, list(_WORKER_LOGS)
    finally:
        sample_module._MOL_INDEX_OFFSET = 0  # type: ignore[attr-defined]


def sample_ligands(
    *,
    mols: list[Molecule],
    offxml_path: Path,
    device_type: str,
    sampling_settings: SamplingSettings,
    output_paths: dict[OutputType, Path],
    dataset_output_paths: list[Path],
    n_processes: int,
    process_dataset: Callable[[int, datasets.Dataset], datasets.Dataset] | None = None,
) -> list[datasets.Dataset]:
    """Sample every ligand, process it in the parent, and save it.

    Generated sampling is an internal, non-resumable workflow operation: the
    workflow rejects non-clean output before calling it, so destinations are absent.
    """
    precomputed = isinstance(sampling_settings, PreComputedDatasetSettings)
    if precomputed:
        raw = _SAMPLING_FNS_REGISTRY[type(sampling_settings)](
            mols=mols,
            off_ff=ForceField(str(offxml_path)),
            device=torch.device(device_type),
            settings=sampling_settings,
            output_paths=output_paths,
        )
    else:
        raw = _sample_every_ligand(
            mols=mols,
            offxml_path=offxml_path,
            device_type=device_type,
            sampling_settings=sampling_settings,
            output_paths=output_paths,
            n_processes=n_processes,
        )

    results = [
        dataset if process_dataset is None else process_dataset(mol_idx, dataset)
        for mol_idx, dataset in enumerate(raw)
    ]
    if not precomputed:
        for dataset, path in zip(results, dataset_output_paths, strict=True):
            dataset.save_to_disk(str(path))
    return results


def _warm_ml_potential(
    mol: Molecule, sampling_settings: SamplingSettings, device_type: str
) -> None:
    """Cache the reference potential's weights before any worker exists.

    Backends download straight to their final path (aimnet creates the file
    before fetching it, mace uses ``urlretrieve``), so workers racing a
    first-time download read each other's partial files. Building one system
    here means every worker finds a complete cache.
    """
    mlp_settings = sampling_settings.mlp_settings  # type: ignore[union-attr]
    key = (
        mlp_settings.ml_potential,
        mlp._freeze_kwargs(mlp_settings.ml_potential_kwargs),
    )
    if key in _WARMED_POTENTIALS:
        return
    mlp.get_ml_omm_system(mol, mlp_settings, torch.device(device_type))
    _WARMED_POTENTIALS.add(key)
    free_gpu_memory()


def _sample_every_ligand(
    *,
    mols: list[Molecule],
    offxml_path: Path,
    device_type: str,
    sampling_settings: SamplingSettings,
    output_paths: dict[OutputType, Path],
    n_processes: int,
) -> list[datasets.Dataset]:
    """Sample every molecule, reporting all per-ligand failures together."""
    workers = max(1, min(n_processes, len(mols)))

    # Each worker claims one GPU at start-up and keeps it, so oversubscribing a GPU
    # stays deliberate rather than depending on which worker picks up a molecule.
    assigned_devices = sampling_devices(device_type, workers)

    worker_args = [
        (
            mol.to_json(),
            mol_idx,
            str(offxml_path),
            device_type if workers == 1 else None,
            sampling_settings,
            output_paths,
        )
        for mol_idx, mol in enumerate(mols)
    ]

    sampled: dict[int, datasets.Dataset] = {}
    worker_logs: dict[int, list[tuple[str, str]]] = {}
    failures: dict[int, Exception] = {}

    if workers == 1:
        for mol_idx, args in enumerate(worker_args):
            try:
                sampled[mol_idx], worker_logs[mol_idx] = _sample_worker(*args)
            except Exception as exc:
                failures[mol_idx] = exc
    else:
        try:
            pickle.dumps(sampling_settings)
        except Exception as exc:
            raise RuntimeError(
                "Parallel sampling requires picklable sampling settings. Runtime "
                "objects such as custom ASE calculators require "
                "n_sampling_processes: 1."
            ) from exc

        _warm_ml_potential(mols[0], sampling_settings, device_type)

        # Spawn, never fork: a forked child inherits the parent's initialised CUDA
        # context, which CUDA does not support, and CUDA_VISIBLE_DEVICES is only read
        # while CUDA initialises, so _init_worker could no longer pin the child to the
        # one GPU it claimed.
        context = multiprocessing.get_context("spawn")

        device_queue = context.Queue()
        for device in assigned_devices:
            device_queue.put(device)

        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(device_queue,),
        ) as executor:
            futures = {
                executor.submit(_sample_worker, *args): mol_idx
                for mol_idx, args in enumerate(worker_args)
            }
            for future in track(
                as_completed(futures),
                total=len(futures),
                description="Ligands completed",
            ):
                try:
                    mol_idx = futures[future]
                    sampled[mol_idx], worker_logs[mol_idx] = future.result()
                except Exception as exc:
                    failures[futures[future]] = exc

    # Replay every worker's records in molecule order, not completion order.
    for mol_idx in sorted(worker_logs):
        for level, message in worker_logs[mol_idx]:
            logger.log(level, f"[molecule {mol_idx}] {message}")

    if failures:
        details = "; ".join(
            f"molecule {i}: {exc}" for i, exc in sorted(failures.items())
        )
        raise RuntimeError(
            f"Sampling failed for {len(failures)} ligand(s): {details}"
        ) from failures[min(failures)]
    return [sampled[mol_idx] for mol_idx in range(len(mols))]
