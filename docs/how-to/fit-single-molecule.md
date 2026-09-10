# Fit a single molecule

The defaults are tuned to be generally robust for single molecule fits typical drug-like molecules. However, you might like to tweak your setting to prioritise accuracy or speed. This page details a few settings you might want to change.

## Defaults you may want to change

- **Reference MLP**: `aimnet2`. Fast and generally robust. See **[how to choose an MLP](choose-an-mlp.md)** for more details.
- **`n_iterations`** (default `2`). Each iteration retrains using MD sampled with the previous iteration's force field. Iteration 2 typically improves test loss over iteration 1. Set to `1` for fast iteration.
- **`training_sampling_settings.n_conformers`** (default `10`). More conformers means more diverse starting points for sampling, and since sampling time is per-conformer, this increases sampling time — useful for flexible molecules with many rotatable bonds.
- **`param_settings.msm_settings`** (default: enabled). Set to `null` in the YAML (`None` in Python) to disable MSM initialisation. The [modified Seminario method](../concepts/method-overview.md#initial-force-field) initialises bond and angle equilibrium values directly from MLP-minimised geometries, ignoring the effect of non-bonded interactions on equilibrium geometry. This can introduce instabilities due to, for example, problematic S–N bonds in sulfonamides. If you see large initial bond/angle deviations, try disabling MSM so that bond and angle parameters start from the parent force field values instead. Initial losses will typically be higher but generall converge to similar values as with MSM.
- **`training_sampling_settings.sampling_protocol`** (default `mm_md_metadynamics_torsion_minimisation`). The default protocol includes short MLP minimisations which improve torsion scan performance. However, these are expensive and can cause connectivity changes (e.g. proton hopping in phosphates). Switching to `mm_md_metadynamics`, which uses only MM-driven sampling, substantially reduces cost and avoids connectivity changes.

## Run it (CLI)

```bash
presto train --param-settings.molecules "CCC(CC)C(=O)Nc2cc(NC(=O)c1c(Cl)cccc1Cl)ccn2"
```

Override individual fields with dotted flags:

```bash
presto train \
    --param-settings.molecules "CCO" \
    --n-iterations 1 \
    --training-settings.n-epochs 200
```

## Run it (YAML)

```bash
presto write-default-yaml workflow_settings.yaml
# edit param_settings.molecules, optionally tune the fields above
presto train-from-yaml workflow_settings.yaml
```

See the **[API reference](../reference/api/settings.md)** for the full settings documentation.

## After the fit

Check the diagnostic plots in `plots/`. The first things to look at are `loss.png`, `correlation_mol0.png`, and `parameter_differences_mol0.png` — see **[Inspect outputs and plots](inspect-outputs.md)**.
