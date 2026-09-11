# Contributing

## Development environment

You'll need [`pixi` installed](https://pixi.sh/latest/). A development environment (the default environment) can be created and activated with:

```shell
pixi shell
```

## Common tasks

```shell
pixi run lint        # Lint the codebase with Ruff
pixi run format      # Format the codebase with Ruff
pixi run type-check  # Type-check the codebase with Mypy
pixi run test        # Run the fast unit tests with Pytest
pixi run test-slow   # Run the slow tests as well
```

## Tests

Unit tests live under `presto/tests/unit/`; integration tests live under `presto/tests/integration/`. Slow tests are marked with `@pytest.mark.slow` and skipped from the default `test` task.

### Test Docker image builds

In GitHub, open **Actions → Publish Docker Image → Run workflow**, select the branch to build, and enter a tag suffix. The workflow builds and pushes the image to GHCR with the tag `test-<suffix>`.

## Style

Ruff is configured in `pyproject.toml` (`[tool.ruff.lint]` and `[tool.ruff.lint.pydocstyle]`). Mypy is run in strict mode (`pyproject.toml [tool.mypy]`). Both run in CI via `.github/workflows/ci.yaml`.
