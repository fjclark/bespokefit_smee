# How-to guides

Short, task-oriented recipes. Each page assumes you've finished **[Get started](../get-started/index.md)**.

## By task

- **[Fit a single molecule](fit-single-molecule.md)** — defaults you may want to change.
- **[Fit a congeneric series](fit-congeneric-series.md)** — share parameters across related molecules using `max_extend_distance`.
- **[Use SDF inputs](use-sdf-inputs.md)** — switch from SMILES to one or more `.sdf` files.
- **[Use your own starting conformers](use-starting-conformers.md)** — seed a sampling stage (or MSM) from an SDF instead of ETKDG.
- **[Use custom charges](use-custom-charges.md)** — bake your own partial charges into the force field as library charges.
- **[Wipe output and rerun](clean-rerun.md)**
- **[Speed up fitting with parallelism](speed-up-with-parallelism.md)** — sample independent ligands concurrently on one node.

## By component

- **[Choose an MLP](choose-an-mlp.md)** — what each supported MLP gives you, and how to pin it.
- **[Use an ASE calculator](use-ase-calculator.md)** — bring your own ML potential via ASE.
- **[Inspect outputs and plots](inspect-outputs.md)** — what each file in the output directory means.
- **[Run from Python](run-from-python.md)** — build `WorkflowSettings` programmatically instead of via the CLI.
