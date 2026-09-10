# Speed up fitting with parallelism

Sampling dominates the wall-clock time of a fit. When you're fitting more than one ligand, `presto` can sample them concurrently in worker processes on a single node, which is usually the cheapest speed-up available.

## Turn it on

`n_sampling_processes` sets the number of worker processes. The default of `1` keeps the previous serial behaviour.

```bash
presto train --param-settings.molecules "CCO" --n-sampling-processes 4
```

```yaml
# workflow_settings.yaml
n_sampling_processes: 4
```

```python
settings = WorkflowSettings(param_settings=..., n_sampling_processes=4)
```

Only sampling is parallelised: test-data generation and the sampling half of each training iteration. Parameterisation, training, and analysis stay in the parent process, so the speed-up is bounded by how much of your fit is spent sampling.

## Choose a number of processes

Ligands are the unit of work, so the worker count is capped at the number of input molecules. Asking for more processes than you have ligands does nothing, and a single-molecule fit gains nothing at all. The feature pays off for a **[congeneric series](fit-congeneric-series.md)**.

Beyond that, the limit is hardware:

- **On GPU**, the best case is one worker per GPU. Workers are assigned round-robin across the devices in `CUDA_VISIBLE_DEVICES` and each keeps the device it claimed for the whole run, so `n_sampling_processes` equal to the number of visible GPUs gives one ligand per GPU. Every worker loads its own force field and ML model, so GPU memory scales with the worker count.
- **Sharing one GPU** between workers requires [NVIDIA MPS](https://docs.nvidia.com/deploy/mps/), which is configured outside `presto`. Without it, the workers' kernels are time-sliced rather than run concurrently. Even with it, extra workers can be slower once one already saturates the card.
- **On CPU**, cap the threads each worker uses (see below). Note that fitting on CPU is very slow regardless of the worker count.

## Cap the threads per worker on CPU

OpenMM and PyTorch each size their thread pools to the whole machine, so without a cap every worker claims all cores and the processes fight over them. Divide the cores you have by the number of workers and export the limits in the launching shell, which the workers inherit:

```bash
export OPENMM_CPU_THREADS=6 OMP_NUM_THREADS=6  # 24 cores, 4 workers
presto train-from-yaml workflow_settings.yaml
```

Without this, raising `n_sampling_processes` on CPU can make the fit slower rather than faster.

## Guard your Python entry point

Workers are spawned as fresh Python processes which re-import your script, so if you drive `presto` from Python rather than the CLI, `get_bespoke_force_field` must sit behind an `if __name__ == "__main__":` guard, as in **[Run from Python](run-from-python.md)**. Without it every worker re-runs the script, and may spawn workers recursively or fail with a multiprocessing bootstrap error.

!!! warning "Notebooks and interactive sessions"
    A REPL or notebook has no importable entry point to guard, so `n_sampling_processes > 1` cannot work there. Use `n_sampling_processes=1`, a guarded `.py` script, or the CLI.

## What the run looks like

Per-ligand progress bars are replaced by a single `Ligands sampled` bar counting completed ligands, because the workers share one terminal. Log records from the workers are buffered and replayed by the parent once the batch is sampled, in molecule order and prefixed with `[molecule N]`, so they do not appear live as they would with a single process.

Every ligand is attempted even if one fails; a single error at the end lists each ligand that could not be sampled.

## When it won't help

- **One ligand.** The worker count is capped at the number of molecules.
- **A pre-computed dataset.** `PreComputedDatasetSettings` loads from disk in the parent process, so it ignores `n_sampling_processes`.
- **Settings holding runtime objects.** Settings are sent to the workers by pickle, so a custom ASE calculator passed through `ml_system_kwargs` cannot be. `presto` fails with an error asking for `n_sampling_processes: 1` when it reaches the sampling stage. See **[Use an ASE calculator](use-ase-calculator.md)**.
- **Across nodes.** This is `multiprocessing` on one node, not a split across SLURM jobs. To use several nodes, fit disjoint subsets of your ligands as separate jobs.
