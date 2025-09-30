"""Integration test for the bespokefit_smee package."""

import subprocess

from ...settings import WorkflowSettings


def test_integration_cli(tmp_cwd) -> None:
    """Test running bespokefit smee via CLI with ethanol, and analysing."""

    # Run the command
    args = [
        "bespokefit_smee",
        "train",
        "--device-type",
        "cpu",  # So we can run on GH actions
        "--n-iterations",
        "1",  # Super short run for testing
        "--parameterisation-settings.smiles",
        "CCO",  # Ethanol
        "--training-settings.n-epochs",
        "10",
        "--training-sampling-settings.sampling-protocol",
        "mm_md",
        "--training-sampling-settings.n-conformers",
        "1",
        "--training-sampling-settings.production-sampling-time-per-conformer",
        "5 ps",
        "--testing-sampling-settings.sampling-protocol",
        "mm_md",
        "--testing-sampling-settings.n-conformers",
        "1",
        "--testing-sampling-settings.production-sampling-time-per-conformer",
        "1 ps",
    ]
    print(" ".join(args))

    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Check the command executed successfully
    assert result.returncode == 0, f"Command failed: {result.stderr}"

    # Create the Path manager and make sure all the expected files have been created
    # and are in the right places
    workflow_settings_yaml = tmp_cwd / "workflow_settings.yaml"
    assert workflow_settings_yaml.exists(), "Workflow settings YAML not found."

    settings = WorkflowSettings.from_yaml(workflow_settings_yaml)
    path_manager = settings.get_path_manager()

    expected_output_files = path_manager.get_all_output_paths(only_if_exists=False)
    for stage, outputs in expected_output_files.items():
        for output_type, path in outputs.items():
            assert path.exists(), (
                f"Expected output {output_type} in stage {stage} not found at {path}."
            )
