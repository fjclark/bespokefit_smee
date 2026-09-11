# Releasing

## Conda-Forge

`presto` is not yet available on conda-forge.

## Versioning

Release tags must be valid, monotonically increasing semantic versions. To add
a new tag:

```shell
git tag <new version>
git push origin <new version>
```

This triggers new documentation and container image builds. Do not rerun an
older release workflow: mutable aliases such as `stable`, `latest`, and the
container's major-minor tag always follow the most recently published release.

Version numbers are derived from git tags using `hatch-vcs` (configured in `pyproject.toml [tool.hatch.version]`). The `presto.__version__` string is set at install time.
