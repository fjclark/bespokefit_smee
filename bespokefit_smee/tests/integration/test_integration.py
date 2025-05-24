"""Integration test for the bespokefit_smee package."""

import subprocess


def test_integration_cli(tmp_cwd) -> None:
    """Test running bespokefit smee via CLI with ethanol, and analysing."""

    # Run the command
    args = [
        "bespokefit_smee",
        "train",
        "--smiles",
        "CCO",  # Ethanol
        "--device-type",
        "cpu",  # So we can run on GH actions
        "--n-iterations",
        "1",  # Super short run for testing
        "--n-epochs",
        "10",
        "--n-train-snapshots",
        "10",
        "--n-test-snapshots",
        "5",
        "--n-conformers",
        "5",
    ]

    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Check the command executed successfully
    assert result.returncode == 0, f"Command failed: {result.stderr}"

    # Check that the expected output is generated
    expected_files = [
        "training_config.yaml",
        "default.offxml",
        "default.scat",
        "trained-0.offxml",
        "trained-0.scat",
        "training-0.data",
    ]
    for file_name in expected_files:
        assert (tmp_cwd / file_name).exists(), (
            f"Expected file '{file_name}' not found in the current working directory."
        )

    directories_found = [d for d in tmp_cwd.iterdir() if d.is_dir()]
    assert len(directories_found) == 3, (
        "Expected exactly three directories in the current working directory."
    )

    # Now, analyse
    args = [
        "bespokefit_smee",
        "analyse",
    ]

    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Check the command executed successfully
    assert result.returncode == 0, f"Analysis command failed: {result.stderr}"

    # Check that the expected output is generated
    expected_plots = ["loss.png", "error_distributions.png"]
    for plot_name in expected_plots:
        assert (tmp_cwd / plot_name).exists(), (
            f"Expected plot '{plot_name}' not found in the current working directory."
        )
