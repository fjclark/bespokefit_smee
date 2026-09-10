# Troubleshooting

First-stop for known failure modes. If your symptom isn't listed here, check the [GitHub issue tracker](https://github.com/cole-group/presto/issues).

## Unstable Fit

**Symptom**: `plots/loss.png` does not flatten; losses blow up; MD is unstable and fit does not finish; parameters change massively.

**Likely causes and fixes**:

- Change in connectivity caused by e.g. protons hopping during the MLP minimisation stages. Inspect the output pdbs to see what samples went into training. Fix by avoiding MLP minimisations by switching to the `mm_md_metadynamics` sampling protocol (and deleting the parameters associated with the minimisation stages). Often occurs for phosphates.
- Poor initial equilibrium values for bonds and angles from MSM initialisation. The [modified Seminario method](../concepts/method-overview.md#initial-force-field) initialises bond and angle equilibrium values directly from MLP-minimised geometries, ignoring the effect of non-bonded interactions on equilibrium geometry. This can introduce instabilities due to, for example, problematic S–N bonds in sulfonamides. If you see large initial bond/angle deviations, try disabling MSM so that bond and angle parameters start from the parent force field values instead. Initial losses will typically be higher but generall converge to similar values as with MSM. Disable MSM by setting **`param_settings.msm_settings`** to `None` (Python) or `null` (YAML).


## ASE charge handling silently wrong

**Symptom**: Fits to charged species drift or never reach reasonable accuracy when `ml_potential="ase"`.

**Cause**: The ASE bridge in OpenMM-ML does not propagate molecular charge automatically. Other MLPs do — this is specific to ASE.

**Fix**: Pass charge explicitly via `mlp_settings.ml_system_kwargs`:

```python
MLPSettings(
    ml_potential="ase",
    ml_system_kwargs={"calculator": calc, "info": {"charge": -1}},
)
```

See **[How-to → Use an ASE calculator → Charge handling caveat](../how-to/use-ase-calculator.md#charge-handling-caveat)**.

## Reload of an ASE settings YAML fails validation

**Symptom**:

```
InvalidSettingsError: ml_system_kwargs contains runtime-only placeholder values
at keys: ['[calculator]']. Supply the actual objects via from_yaml(..., overwrite=...)
before validation.
```

**Cause**: ASE calculators don't serialise to YAML, so `presto` writes a placeholder. On reload the placeholder fails validation by design.

**Fix**: Inject the calculator on load:

```python
settings = WorkflowSettings.from_yaml(
    "workflow_settings.yaml",
    overwrite={
        "training_sampling_settings": {
            "mlp_settings": {"ml_system_kwargs": {"calculator": calculator}}
        }
    },
)
```

## Too few conformations after outlier filtering

**Symptom**:

```
Filtering would remove too many conformations: ... below min_conformations=...
```

**Cause**: The outlier filter rejected most conformations as having unreasonable MM-vs-MLP differences. See "Unstable Fit" above.


## Parameterisation fails for some molecules

**Symptom**: the run stops at the end of parameterisation, before any sampling:

```
MoleculeParameterisationError: Parameterisation failed for 2 of 20 molecules:
  - molecule 3 (ligand-4, CC(C)Cc1ccc(cc1)C(C)C(=O)O): ValueError: No registered toolkits can provide the capability "generate_conformers" ...
  - molecule 11 (c1ccccc1): ValueError: ...
```

**Cause**: OpenFF could not handle those molecules, most often because RDKit's ETKDG
cannot embed them (needed by the modified Seminario method) or because charge assignment
failed.

**Fix**: every molecule is attempted at both the modified Seminario and the charge
assignment stage before the error is raised, so the list is complete and a molecule with
problems at both stages reports both (separated by `;`). Remove or replace all of the
listed molecules in `param_settings.molecules` in one pass rather than resubmitting once
per failure. Supplying `msm_settings.starting_conformers` sidesteps the requirement
for conformer generation (see
**[How-to → Use your own starting conformers](../how-to/use-starting-conformers.md)**).


## Version mismatch warning on YAML load

**Symptom**:

```
WARNING: Version mismatch: settings version 0.6.0 may not be compatible with current version 0.8.0.
```

**Cause**: `version` in the YAML disagrees with the installed `presto` version at the major or minor level.

**Fix**: Usually benign for patch-level differences. For major/minor differences, regenerate the YAML with `presto write-default-yaml` and re-apply your customisations, especially if the changelog flags a breaking change.
