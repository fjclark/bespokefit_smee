"""Unit tests for the sampling coordinator."""

import multiprocessing
import os
import threading
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import datasets
import pytest
from openff.toolkit import Molecule

from presto import sample, sampling_coordinator
from presto.sampling_coordinator import sample_ligands, sampling_devices
from presto.settings import MMMDSamplingSettings, PreComputedDatasetSettings


def _precomputed_settings(n_datasets: int) -> PreComputedDatasetSettings:
    return PreComputedDatasetSettings(
        dataset_paths=[Path(f"dataset-{i}") for i in range(n_datasets)]
    )


def _generated_inputs(tmp_path: Path, n_mols: int = 2) -> dict:
    settings = MMMDSamplingSettings()
    return {
        "mols": [Molecule.from_smiles("C") for _ in range(n_mols)],
        "offxml_path": tmp_path / "force-field.offxml",
        "device_type": "cpu",
        "sampling_settings": settings,
        "output_paths": {
            output_type: tmp_path / output_type.value
            for output_type in settings.output_types
        },
        "canonical_paths": [
            tmp_path / f"energy_and_force_data_mol{i}" for i in range(n_mols)
        ],
        "n_processes": 1,
    }


def _successful_worker(molecule_json, molecule_index, *args) -> tuple:
    return datasets.Dataset.from_dict({"molecule": [molecule_index]}), []


class _InlineExecutor:
    """Run submitted work inline while exposing the executor context/Future API."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except BaseException as exc:
            future.set_exception(exc)
        return future


def test_sampling_devices_round_robin(monkeypatch):
    """Workers share the visible CUDA devices round-robin, or all run on the CPU."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert sampling_devices("cpu", 3) == ["cpu"] * 3
    with patch("torch.cuda.device_count", return_value=2):
        assert sampling_devices("cuda", 3) == ["0", "1", "0"]
    with patch("torch.cuda.device_count", return_value=0):
        assert sampling_devices("cuda", 2) == ["0", "0"]


def test_sampling_devices_names_the_parents_gpus(monkeypatch):
    """Workers are handed the parent's device ids, not indices into its visible list."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    with patch("torch.cuda.device_count", return_value=2):
        assert sampling_devices("cuda", 3) == ["4", "5", "4"]
    # CUDA truncates the visible list at the first entry it rejects, so a worker is
    # never handed an id past the count torch reports.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,99")
    with patch("torch.cuda.device_count", return_value=1):
        assert sampling_devices("cuda", 2) == ["4", "4"]


def test_processes_each_precomputed_dataset(monkeypatch):
    """Precomputed training data is passed through the processing callback."""
    loaded = [
        datasets.Dataset.from_dict({"value": [10]}),
        datasets.Dataset.from_dict({"value": [20]}),
    ]
    processed = [
        datasets.Dataset.from_dict({"value": [11]}),
        datasets.Dataset.from_dict({"value": [21]}),
    ]
    monkeypatch.setitem(
        sampling_coordinator._SAMPLING_FNS_REGISTRY,
        PreComputedDatasetSettings,
        MagicMock(return_value=loaded),
    )
    process_dataset = MagicMock(side_effect=processed)

    with patch("presto.sampling_coordinator.ForceField"):
        result = sample_ligands(
            mols=[MagicMock(), MagicMock()],
            offxml_path=Path("force-field.offxml"),
            device_type="cpu",
            sampling_settings=_precomputed_settings(len(loaded)),
            output_paths={},
            canonical_paths=[Path("canonical-0"), Path("canonical-1")],
            n_processes=1,
            process_dataset=process_dataset,
        )

    assert result == processed
    assert [call.args for call in process_dataset.call_args_list] == [
        (0, loaded[0]),
        (1, loaded[1]),
    ]


def test_returns_precomputed_datasets_without_callback(monkeypatch):
    """Precomputed test data remains unchanged when no callback is supplied."""
    loaded = [datasets.Dataset.from_dict({"value": [10]})]
    monkeypatch.setitem(
        sampling_coordinator._SAMPLING_FNS_REGISTRY,
        PreComputedDatasetSettings,
        MagicMock(return_value=loaded),
    )

    with patch("presto.sampling_coordinator.ForceField"):
        result = sample_ligands(
            mols=[MagicMock()],
            offxml_path=Path("force-field.offxml"),
            device_type="cpu",
            sampling_settings=_precomputed_settings(len(loaded)),
            output_paths={},
            canonical_paths=[Path("canonical-0")],
            n_processes=1,
        )

    assert result == loaded


@pytest.mark.parametrize("n_processes", [1, 2])
def test_samples_processes_and_saves_every_molecule(tmp_path, monkeypatch, n_processes):
    """Generated sampling runs, processes, and saves every molecule in order."""
    inputs = _generated_inputs(tmp_path)
    inputs["n_processes"] = n_processes
    inputs["process_dataset"] = lambda mol_idx, dataset: dataset.add_column(
        "processed", [mol_idx]
    )
    worker = MagicMock(side_effect=_successful_worker)
    monkeypatch.setattr(sampling_coordinator, "_sample_worker", worker)
    warm = MagicMock()
    monkeypatch.setattr(sampling_coordinator, "_warm_ml_potential", warm)
    if n_processes > 1:
        monkeypatch.setattr(
            sampling_coordinator, "ProcessPoolExecutor", _InlineExecutor
        )

    result = sample_ligands(**inputs)

    # The weight cache is warmed once in the parent, never on the serial path.
    assert warm.call_count == (1 if n_processes > 1 else 0)

    assert [dataset["molecule"][0] for dataset in result] == [0, 1]
    assert [dataset["processed"][0] for dataset in result] == [0, 1]
    assert [call.args[1] for call in worker.call_args_list] == [0, 1]
    committed = [datasets.load_from_disk(path) for path in inputs["canonical_paths"]]
    assert [dataset["processed"][0] for dataset in committed] == [0, 1]


def test_worker_failures_are_aggregated_after_all_ligands(tmp_path, monkeypatch):
    """Ordinary failures remain per-ligand errors and do not stop sampling early."""
    inputs = _generated_inputs(tmp_path)
    worker = MagicMock(
        side_effect=[RuntimeError("first failed"), ValueError("second failed")]
    )
    monkeypatch.setattr(sampling_coordinator, "_sample_worker", worker)

    with pytest.raises(RuntimeError) as exc_info:
        sample_ligands(**inputs)

    assert "molecule 0: first failed" in str(exc_info.value)
    assert "molecule 1: second failed" in str(exc_info.value)
    assert worker.call_count == 2
    assert not any(path.exists() for path in inputs["canonical_paths"])


@pytest.mark.parametrize("process_control_exception", [KeyboardInterrupt, SystemExit])
def test_process_control_exception_escapes_immediately(
    tmp_path, monkeypatch, process_control_exception
):
    """Process-control exceptions are not aggregated as ligand failures."""
    inputs = _generated_inputs(tmp_path)
    worker = MagicMock(side_effect=process_control_exception("stop sampling"))
    monkeypatch.setattr(sampling_coordinator, "_sample_worker", worker)

    with pytest.raises(process_control_exception, match="stop sampling"):
        sample_ligands(**inputs)

    assert worker.call_count == 1


def test_parallel_sampling_rejects_unpicklable_settings(tmp_path):
    """Runtime objects which cannot cross to a spawned worker fail up front."""
    inputs = _generated_inputs(tmp_path)
    inputs["n_processes"] = 2
    inputs["sampling_settings"].mlp_settings.ml_system_kwargs = {
        "calculator": threading.Lock()
    }

    with pytest.raises(RuntimeError, match="n_sampling_processes: 1"):
        sample_ligands(**inputs)


@pytest.mark.parametrize(
    ("devices", "expected"),
    [
        # A CUDA worker hides every other GPU, so its own is always index 0.
        (["0", "1"], ["cuda:0", "cuda:0"]),
        (["0", "0"], ["cuda:0", "cuda:0"]),
        (["cpu", "cpu"], ["cpu", "cpu"]),
    ],
)
def test_each_worker_claims_one_device(monkeypatch, devices, expected):
    """Workers claim a device each at start-up, oversubscribed GPUs included."""
    # _init_worker silences the process it runs in, so undo that for the test session.
    monkeypatch.setattr(sampling_coordinator, "logger", MagicMock())
    # _init_worker narrows this process's own CUDA_VISIBLE_DEVICES; setenv restores it.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(sample, "track", sample.track)
    monkeypatch.setattr(sampling_coordinator, "_WORKER_DEVICE", "cpu")
    queue = multiprocessing.get_context("spawn").Queue()
    for device in devices:
        queue.put(device)

    claimed = []
    for _ in devices:
        sampling_coordinator._init_worker(queue)
        claimed.append(sampling_coordinator._WORKER_DEVICE)

    assert claimed == expected


def test_warms_the_weight_cache_before_spawning_workers(tmp_path, monkeypatch):
    """One system is built in the parent so workers never race a first download."""
    inputs = _generated_inputs(tmp_path)
    inputs["n_processes"] = 2
    monkeypatch.setattr(sampling_coordinator, "_WARMED_POTENTIALS", set())
    monkeypatch.setattr(
        sampling_coordinator,
        "_sample_worker",
        MagicMock(side_effect=_successful_worker),
    )
    monkeypatch.setattr(sampling_coordinator, "ProcessPoolExecutor", _InlineExecutor)
    get_system = MagicMock()
    monkeypatch.setattr(sampling_coordinator.mlp, "get_ml_omm_system", get_system)

    sample_ligands(**inputs)
    sample_ligands(**inputs)

    # Warmed once per process, not once per sampling call.
    assert get_system.call_count == 1
    assert get_system.call_args.args[1] is inputs["sampling_settings"].mlp_settings


def test_worker_logs_are_replayed_by_the_parent(tmp_path, monkeypatch):
    """Warnings raised inside a worker reach the parent, tagged by molecule."""
    inputs = _generated_inputs(tmp_path)
    inputs["n_processes"] = 2

    def worker(molecule_json, molecule_index, *args):
        return (
            datasets.Dataset.from_dict({"molecule": [molecule_index]}),
            [("WARNING", "No rotatable bonds found")],
        )

    monkeypatch.setattr(sampling_coordinator, "_sample_worker", worker)
    monkeypatch.setattr(sampling_coordinator, "_warm_ml_potential", MagicMock())
    monkeypatch.setattr(sampling_coordinator, "ProcessPoolExecutor", _InlineExecutor)
    replayed = []
    monkeypatch.setattr(
        sampling_coordinator,
        "logger",
        MagicMock(log=lambda level, message: replayed.append((level, message))),
    )

    sample_ligands(**inputs)

    assert replayed == [
        ("WARNING", "[molecule 0] No rotatable bonds found"),
        ("WARNING", "[molecule 1] No rotatable bonds found"),
    ]


def test_worker_claims_one_visible_gpu(monkeypatch):
    """A CUDA worker sees only the one device it claimed."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    monkeypatch.setattr(sampling_coordinator, "logger", MagicMock())
    monkeypatch.setattr(sampling_coordinator, "_WORKER_DEVICE", "cpu")
    queue = multiprocessing.get_context("spawn").Queue()
    queue.put("5")

    sampling_coordinator._init_worker(queue)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"
    # The claimed device is now the only one visible, so it is index 0 everywhere.
    assert sampling_coordinator._WORKER_DEVICE == "cuda:0"
