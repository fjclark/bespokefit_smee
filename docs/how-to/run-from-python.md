# Run from Python

The CLI is an easy way to run `presto` for one-off fits, but the Python API gives you better control when you're sweeping settings, doing batch fits, or composing `presto` into a larger pipeline. The end-to-end notebook is at **[Walk-through (Python API)](../examples/basic-walk-through-python-api.ipynb)**; this page is the short prose reference.

## Build a `WorkflowSettings` object

```python
from presto.settings import ParamSettings, WorkflowSettings
from presto.workflow import get_bespoke_force_field

if __name__ == "__main__":
    settings = WorkflowSettings(
        param_settings=ParamSettings(
            molecule_input_type="smiles",
            molecules="CCO",
        ),
        device_type="cuda",
    )

    bespoke_ff = get_bespoke_force_field(settings)
```

`get_bespoke_force_field` returns the final fitted `openff.toolkit.ForceField` and writes the same output tree the CLI would. Pass `write_settings=False` to skip writing `workflow_settings.yaml` (useful when you've loaded settings from a YAML file already).

## Override individual fields after load

`WorkflowSettings.from_yaml` accepts an `overwrite` dict that is deep-merged into the YAML before validation:

```python
settings = WorkflowSettings.from_yaml(
    "workflow_settings.yaml",
    overwrite={"n_iterations": 1, "device_type": "cuda"},
)
```

This is the recommended way to inject runtime objects (e.g. an ASE calculator) that can't round-trip through YAML — see **[Use an ASE calculator](use-ase-calculator.md)**.

## Parallel ligand sampling

Set `n_sampling_processes` on `WorkflowSettings` (or `--n-sampling-processes` on
`presto train`) to sample independent ligands concurrently on one node. The default of
one process keeps the serial behaviour.

On CUDA, workers are assigned round-robin to the devices visible through
`CUDA_VISIBLE_DEVICES`. Each worker loads its own force field and ML model, so model
and CUDA memory scale with the worker count, and more workers can be slower once one
already saturates a GPU. Concurrent execution on a single GPU needs NVIDIA MPS,
configured outside Presto. Only sampling is parallelised; parameterisation, fitting,
and analysis stay in the parent process.

On CPU, cap the threads per worker before launching. OpenMM and PyTorch each size
their thread pools to the whole machine, so every worker claims all cores no matter
how many workers there are, and the processes then fight for them. Divide the cores
you have by the number of workers:

```bash
export OPENMM_CPU_THREADS=6 OMP_NUM_THREADS=6  # 24 cores, 4 workers
```

Workers inherit these, so setting them in the launching shell is enough. Without
them, raising `n_sampling_processes` on CPU could make the fit slower rather than
faster.

!!! warning "Guard the Python entry point"
    Workers are fresh Python processes that re-import your script. With
    `n_sampling_processes > 1`, call `get_bespoke_force_field` behind an
    `if __name__ == "__main__":` guard, as in the example above; without it every
    worker re-runs the script and may spawn workers recursively or fail with a
    multiprocessing bootstrap error. Interactive sessions and notebooks have no
    importable guard, so use `n_sampling_processes=1` there, a guarded `.py` script,
    or the `presto` CLI.

## Reference

- API reference: [`WorkflowSettings`](../reference/api/settings.md#presto.settings.WorkflowSettings), [`get_bespoke_force_field`](../reference/api/workflow.md#presto.workflow.get_bespoke_force_field).
