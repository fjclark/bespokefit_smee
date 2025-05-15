"""Integration test for the bespokefit_smee package."""

import subprocess


def test_integration_cli(tmp_cwd) -> None:
    """Test running bespokefit smee via CLI with ethanol."""

    # Run the command
    args = [
        "bespokefit_smee",
        "train",
        "--smiles",
        "CCO",  # Ethanol
        "--device-type",
        "cpu",  # So we can run on GH actions
        "--N-iterations",
        "1",  # Super short run for testing
        "--N-epochs",
        "10",
        "--N-train",
        "10",
        "--N-test",
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
    assert len(directories_found) == 2, (
        "Expected exactly two directories in the current working directory."
    )
