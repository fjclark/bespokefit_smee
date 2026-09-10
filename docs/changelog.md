# Changelog

## 0.9.0

### Features

- Add an optional `starting_conformers` setting to each sampling stage (`training_sampling_settings`, `testing_sampling_settings`) and to `msm_settings`. When set to an SDF path, that stage starts from the supplied conformers (matched to each molecule by graph isomorphism and realigned automatically) instead of generating them with ETKDG; `n_conformers` is ignored for that stage. The default remains ETKDG.In [#78](https://github.com/cole-group/presto/pull/78).
- Add `presto.create_types.add_library_charges_to_forcefield` to write custom partial charges from OpenFF `Molecule` objects into a force field as library charges, addressing [#64](https://github.com/cole-group/presto/issues/64).

### Fixes

- Correct linear and near-linear MSM angle force constants that were half their
  intended SMIRNOFF / OpenMM values due to an uncancelled intermediate QUBEKit
  convention factor.
- Fix a latent `IndexError` in the MSM step when fewer conformers were available than `n_conformers`; the conformer loop now iterates the conformers actually present.

### Documentation

- Add [Use custom charges](how-to/use-custom-charges.md) how-to guide.
- Add [Use your own starting conformers](how-to/use-starting-conformers.md) how-to guide.
- Add a `CITATION.cff` file and cite the presto preprint in the README and docs, and point users to the OpenFF publications to cite, in [#76](https://github.com/cole-group/presto/pull/76).
- Document installing `presto` from `conda-forge` as `presto-fit` (note this comes without the MLP dependencies, which must be installed separately) in [#70](https://github.com/cole-group/presto/pull/70).

## 0.8.1

### Maintenance

- Bundle full Egret license in [#66](https://github.com/cole-group/presto/pull/66).

## 0.8.0

### Breaking changes

- Rename `ParameterisationSettings` to `ParamSettings` and the `parameterisation_settings` field to `param_settings` throughout. This avoids confusion between British/American spellings. YAML configs must update the key from `parameterisation_settings:` to `param_settings:`. In [#62](https://github.com/cole-group/presto/pull/62).

- Replace flat ML potential settings with nested `mlp_settings` in sampling and MSM settings in [#58](https://github.com/cole-group/presto/issues/58). Legacy flat fields such as `ml_potential` and `ml_potential_kwargs` are no longer accepted.
- Remove `AvailableModels` and `validate_model_charge_compatibility`. `presto` now accepts any OpenMM-ML model identifier supported by the local OpenMM-ML install and trusts the user to pick a model compatible with their system. Also in [#58](https://github.com/cole-group/presto/issues/58).
- Add runtime-object placeholder handling for `mlp_settings.ml_system_kwargs`: non-serializable values (e.g ASE Calculator objects) are written as placeholders and loading fails with `InvalidSettingsError` until placeholders are replaced (for example with `from_yaml(..., overwrite=...)`) before validation. Again, in [#58](https://github.com/cole-group/presto/issues/58).

### Documentation

- Rewrite documentation with Diátaxis structure: Get started, How-to, Concepts, and Reference sections.
- Add new pages: installation guide, quickstart, bespoke SMIRNOFF primer, sampling protocols, type generation, MLP guide, output layout, troubleshooting, and glossary. Add Python API walk-through notebook.  - Polish Pydantic Field descriptions in settings to improve auto-generated API reference. Add example invocations to CLI subcommand docstrings. Add Python API examples for running fitting via `get_bespoke_force_field(...)`, including optional OpenMM-ML ASE calculator usage through `ml_system_kwargs`. In [#61](https://github.com/cole-group/presto/pull/61).
- Clarify charge behavior: automatic molecular charge propagation applies to non-ASE MLPs; for `ml_potential="ase"` charge must be passed explicitly in `ml_system_kwargs`.

### Maintenance

- Switch to installing `smee-base` and `descent` from `conda-forge`, which unblocks the conda package. See [#65](https://github.com/cole-group/presto/pull/65).
- Update the default env to use Python 3.12 to avoid dm-tree (an Orb dep) build issues. In [#60](https://github.com/cole-group/presto/pull/60).

## 0.7.0

### Fixes

- Guard against lack of any constaints in the input force field in [#56](https://github.com/cole-group/presto/pull/56). This allows use of the unconstrained version of Parsely.
- Set default model to AIMNet2 in light of current AceFF 2.0 OpenMM-ML issues in [#49](https://github.com/cole-group/presto/pull/49). Also add README warning about AceFF 2.0.

### Improvements

- Make molecule loading more modular and accept SDFs in [#55](https://github.com/cole-group/presto/pull/55). Note this is a breaking change -- the old param_settings.smiles field must now be replaced by param_settings.molecules.
- Add CLI sub-command which shows version in [#54](https://github.com/cole-group/presto/pull/54).
- Add Orb-v3 OMOL model in [#49](https://github.com/cole-group/presto/pull/49).

## 0.6.0

### Fixes

- Raise an error if too few conformers survive filtering in [#38](https://github.com/cole-group/presto/pull/38). Fixes [#30](https://github.com/cole-group/presto/issues/30).
- Remove stereochemical information from generated types in [#36](https://github.com/cole-group/presto/pull/37). This avoids a niche issue where mixing the RDKit and OpenEye toolkits would result in failed type generation for e.g. chiral sulfoxides.

### Maintenance

- Update dependabot options in [#40](https://github.com/cole-group/presto/pull/40)
- Update Egret-1 model to the lastest version in [#35](https://github.com/cole-group/presto/pull/35)

### Improvements

- Make linting more strict and improve docstrings in [#39](https://github.com/cole-group/presto/pull/39)
- Update aromaticity model used for selecting rotatable torsions for metadynamics to give more intuitive results (ditch MDL and go for RDKit default) in [#33](https://github.com/cole-group/presto/pull/33).
- Improve consistency of how the device arguments are passed around (literal or torch device). [#36](https://github.com/cole-group/presto/pull/36)
- Refactor simulation creation logic to reduce duplication and always set the platform to CPU for ML systems (required so that MACE models work with PythonForce). See [this commit](https://github.com/cole-group/presto/commit/c0f9dfc7fd3055c20ca87975ec9703c129e2a070).
- Update environments to OpenMM 8.5 (with PythonForce) and OpenMM-ML. This simplifies the environments required (as we can drop NNPOPs) and means we have have all MLPs in one env. The only issue is that we can now only support CUDA 12.9. [#34](https://github.com/cole-group/presto/pull/34).

## 0.5.1

### Fixes

- Fixes typo in default settings which meant that linear torsions matching ``[*:1]~[*:2]=[#6,#7,#16,#15;X2:3]=[*:4]`` were not excluded from training ([#32](https://github.com/cole-group/presto/pull/32))

## 0.5.0

### Improvements

- Use MDTraj to calculate torsions rather than doing this manually
- Make metadynamics more aggressive, allow user to specify torsions targeted by metadynamics, and include in-ring aliphatic torsions by default in [#26](https://github.com/cole-group/presto/pull/26)

### Fixes

- Remove unused settings fields in [#25](https://github.com/cole-group/presto/pull/25#issuecomment-3861009692)

## 0.4.0

### Improvements

- Improve cleanliness/prettiness of CLI output
- Reduce memory use during congeneric series fitting in [#17](https://github.com/cole-group/presto/pull/17)
- Improve test quality, speed, and coverage in [#19](https://github.com/cole-group/presto/pull/19)
- Update environments with explicit python and cuda versions and document installation with different cuda versions
- Plot sampling of rotatable torsions in [#23](https://github.com/cole-group/presto/pull/23)

### Fixes

- Update ParameterConfig defaults so that linear torsions are not trained in [#18](https://github.com/cole-group/presto/pull/18)

## 0.3.0

- Renamed `bespokefit_smee` -> `presto` (Parameter Refinement Engine for Smirnoff Training / Optimisation)
- Added more documentation on the method, recommended settings, and outputs.

## 0.2.0

### New Features

- Implement new default protocol with MLP-minimised configurations
- Add support for loading multiple pre-computed datasets for training
- Implement flexible bespoke SMARTS type generation with `MergeQueryHs`
- Add support for multi-molecule simultaneous fits
- Add dataset filtering function for preprocessing
- Add function for calculating Hessian matrices

### Improvements

- **GPU Memory Management**: Significantly improved GPU memory handling
  - Add GPU memory cleanup utility function
  - Clear GPU memory after sampling operations
  - Reduce GPU memory usage throughout training pipeline
  - Fix GPU memory leaks with LM optimizer
  - Make CUDA operations conditional on availability for CPU-only environments
- **Modified Seminario Method (MSM)**
  - Complete reimplementation of MSM for better performance
- **ML Potential Updates**
  - Add aceff-2.0 and make it the default MLP
  - Update default sampling settings (consistent with higher speed of aceff-2.0)
- **Force Field Updates**
  - Update default MM-FF to 2.3.0
  - Ensure we train FF with bespoke types added
  - Ensure parameter names are not overwritten
- **Regularization Improvements**
  - Overhaul regularization and calculation of loss
  - Normalize regularization per-parameter
  - Decouple type generation from regularization
- **Optimizer Fixes**
  - Fix LM optimizer implementation
  - Clear cache between iterations with Adam
  - Avoid GPU memory leaks with LM optimizer
- **Path Management**
  - Fix path management for multiple molecules
  - Improve handling of output paths for per-molecule outputs

### Maintenance

- Remove bespoke toolkit wrapper

## 0.1.1

### Documentation

- Added example notebook

## 0.1.0

- Initial implementation.
