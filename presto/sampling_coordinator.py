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

from . import mlp, sample as sample_module
from .outputs import OutputType
from .sample import _SAMPLING_FNS_REGISTRY
from .settings import PreComputedDatasetSettings, SamplingSettings
from .utils._suppress_output import suppress_unwanted_output
from .utils.gpu import free_gpu_memory

# The device this worker process claimed at start-up; unused in the parent.
_WORKER_DEVICE = "cpu"

# Records logged while sampling one molecule, replayed by the parent in order.
# Stays empty in the parent, where logging reaches the real sinks directly.
_WORKER_LOGS: list[tuple[str, str]] = []

# The GPUs visible to the parent, read before _init_worker narrows them to one.
_PARENT_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")

# Potentials this process has already pulled weights for; see _warm_ml_potential.
_WARMED_POTENTIALS: set[tuple[str, tuple[tuple[str, str], ...]]] = set()


def sampling_devices(device_type: str, n_workers: int) -> list[str]:
    """Return round-robin logical devices for the requested workers."""
    if device_type != "cuda":
        return ["cpu"] * n_workers
    # torch.cuda.device_count() already reports only CUDA_VISIBLE_DEVICES.
    n_devices = max(1, torch.cuda.device_count())
    return [f"cuda:{i % n_devices}" for i in range(n_workers)]


def _capture_log(message: loguru.Message) -> None:
    """Buffer one log record for the parent to replay."""
    _WORKER_LOGS.append((message.record["level"].name, message.record["message"]))


def _init_worker(devices: multiprocessing.Queue) -> None:  # type: ignore[type-arg]
    """Claim one device for this worker and buffer its logs for the parent."""
    global _WORKER_DEVICE
    _WORKER_DEVICE = devices.get()
    # Hide every other GPU so that OpenMM's DeviceIndex, torch's cuda:0, and any
    # backend which picks its own device all resolve to the one device claimed here.
    # Must run before torch initialises CUDA in this process: spawn imports alone
    # do not, but a future import calling torch.cuda.* would break this.
    if _WORKER_DEVICE.startswith("cuda:"):
        # sampling_devices() indexes into the parent's visible list, not the machine.
        ids = _PARENT_VISIBLE_DEVICES.split(",") if _PARENT_VISIBLE_DEVICES else None
        claimed = int(_WORKER_DEVICE.split(":")[1])
        os.environ["CUDA_VISIBLE_DEVICES"] = ids[claimed] if ids else str(claimed)
        _WORKER_DEVICE = "cuda:0"
    suppress_unwanted_output()
    logger.remove()
    logger.add(_capture_log)
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
    canonical_paths: list[Path],
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
        for dataset, path in zip(results, canonical_paths, strict=True):
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
    devices = sampling_devices(device_type, workers)
    # Each worker claims one device at start-up and keeps it, so oversubscribing a
    # GPU stays deliberate rather than depending on which worker picks up a molecule.
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
        context = multiprocessing.get_context("spawn")
        device_queue = context.Queue()
        for device in devices:
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
