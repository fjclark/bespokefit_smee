"""Unit tests for workflow lifecycle behavior."""

from unittest.mock import MagicMock, patch

import pytest

from presto.outputs import WorkflowPathManager, WorkflowStatus
from presto.workflow import get_bespoke_force_field


@pytest.mark.parametrize(
    ("stage_name", "expected_status"),
    [
        ("test_data", WorkflowStatus.PARTIAL),
        ("training_iteration_2", WorkflowStatus.COMPLETE),
    ],
)
def test_non_clean_fit_fails_before_parameterisation(
    tmp_path, stage_name, expected_status
):
    """The workflow rejects reruns before any molecule setup or output is written."""
    stage_path = tmp_path / stage_name
    stage_path.mkdir()
    if expected_status == WorkflowStatus.COMPLETE:
        (stage_path / "bespoke_ff.offxml").write_text("force field")

    settings = MagicMock()
    settings.get_path_manager.return_value = WorkflowPathManager(
        output_dir=tmp_path, n_iterations=2
    )

    with (
        patch("presto.workflow.parameterise") as parameterise,
        pytest.raises(RuntimeError, match=rf"{expected_status.value}.*presto clean"),
    ):
        get_bespoke_force_field(settings)

    parameterise.assert_not_called()
