# Wipe, Re-run, and Re-analyse

How to wipe output and rerun.

## Clean a directory

`presto clean` removes the stage directories the settings predict, but **keeps the settings YAML**:

```bash
presto clean workflow_settings.yaml
```

The settings file is the single source of truth: `initial_statistics/`, `test_data/`,
`plots/`, and `training_iteration_1` through `training_iteration_<n_iterations>` are
reserved for Presto and are removed recursively. Nothing else is touched, so files at
the output root and iteration directories beyond `n_iterations` are left alone.

If one of those directories holds a path the settings do not predict, such as your own
notes or a plot you made, `presto clean` refuses and deletes nothing. Move the file
out, or use a different `output_dir`. The only way your own input is deleted is by
placing it at a path Presto generates, for example a pre-computed dataset saved to
`test_data/energy_and_force_data_mol0/`.

The output manager reports a fit as:

- `clean` when none of the predicted stage directories exists;
- `partial` when one exists without the final fitted force field; or
- `complete` when `training_iteration_<n_iterations>/bespoke_ff.offxml` exists.

Both partial and complete output must be cleaned before another fit can start. This
prevents an interrupted run from silently reusing an old trajectory or metadynamics
bias.

## Re-run from a YAML

After cleaning, any `workflow_settings.yaml` written by a previous run can be replayed with:

```bash
presto train-from-yaml workflow_settings.yaml
```

To make small changes before replaying, either edit the YAML directly or use the `overwrite` argument on `WorkflowSettings.from_yaml`:

```python
from presto.settings import WorkflowSettings
from presto.workflow import get_bespoke_force_field

settings = WorkflowSettings.from_yaml(
    "workflow_settings.yaml",
    overwrite={"n_iterations": 4, "training_settings": {"n_epochs": 2000}},
)
get_bespoke_force_field(settings, write_settings=False)
```
