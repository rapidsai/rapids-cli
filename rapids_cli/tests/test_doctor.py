# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import warnings
from unittest.mock import MagicMock, patch

from rapids_cli.doctor.doctor import CheckResult, doctor_check


def mock_passing_check(verbose=False, **kwargs):
    """Mock check that passes."""
    return "Check passed"


def mock_failing_check(verbose=False, **kwargs):
    """Mock check that fails."""
    raise ValueError("Check failed")


def mock_warning_check(verbose=False, **kwargs):
    """Mock check that issues a warning."""
    warnings.warn("This is a warning", stacklevel=2)
    return True


def test_doctor_check_all_pass(capsys):
    """Test doctor_check with all checks passing."""
    mock_ep1 = MagicMock()
    mock_ep1.name = "test_check_1"
    mock_ep1.value = "test.module:check1"
    mock_ep1.load.return_value = mock_passing_check

    mock_ep2 = MagicMock()
    mock_ep2.name = "test_check_2"
    mock_ep2.value = "test.module:check2"
    mock_ep2.load.return_value = mock_passing_check

    with patch(
        "rapids_cli.doctor.doctor.entry_points", return_value=[mock_ep1, mock_ep2]
    ):
        result = doctor_check(verbose=False, dry_run=False)
        assert result is True

    captured = capsys.readouterr()
    assert "All checks passed!" in captured.out


def test_doctor_check_with_failure(capsys):
    """Test doctor_check with one check failing."""
    mock_ep1 = MagicMock()
    mock_ep1.name = "passing_check"
    mock_ep1.value = "test.module:check1"
    mock_ep1.load.return_value = mock_passing_check

    mock_ep2 = MagicMock()
    mock_ep2.name = "failing_check"
    mock_ep2.value = "test.module:check2"
    mock_ep2.load.return_value = mock_failing_check

    with patch(
        "rapids_cli.doctor.doctor.entry_points", return_value=[mock_ep1, mock_ep2]
    ):
        result = doctor_check(verbose=False, dry_run=False)
        assert result is False

    captured = capsys.readouterr()
    assert "failing_check failed" in captured.out


def test_doctor_check_verbose(capsys):
    """Test doctor_check with verbose flag."""
    mock_ep = MagicMock()
    mock_ep.name = "test_check"
    mock_ep.value = "test.module:check"
    mock_ep.load.return_value = mock_passing_check

    with patch("rapids_cli.doctor.doctor.entry_points", return_value=[mock_ep]):
        result = doctor_check(verbose=True, dry_run=False)
        assert result is True

    captured = capsys.readouterr()
    assert "Discovering checks" in captured.out
    assert "Found check 'test_check'" in captured.out
    assert "Discovered 1 checks" in captured.out


def test_doctor_check_dry_run(capsys):
    """Test doctor_check with dry_run flag."""
    mock_ep = MagicMock()
    mock_ep.name = "test_check"
    mock_ep.value = "test.module:check"
    mock_ep.load.return_value = mock_passing_check

    with patch("rapids_cli.doctor.doctor.entry_points", return_value=[mock_ep]):
        result = doctor_check(verbose=False, dry_run=True)
        assert result is True

    captured = capsys.readouterr()
    assert "Dry run, skipping checks" in captured.out


def test_doctor_check_with_filters(capsys):
    """Test doctor_check with filters."""
    mock_ep1 = MagicMock()
    mock_ep1.name = "cudf_check"
    mock_ep1.value = "cudf.module:check"
    mock_ep1.load.return_value = mock_passing_check

    mock_ep2 = MagicMock()
    mock_ep2.name = "cuml_check"
    mock_ep2.value = "cuml.module:check"
    mock_ep2.load.return_value = mock_passing_check

    with patch(
        "rapids_cli.doctor.doctor.entry_points", return_value=[mock_ep1, mock_ep2]
    ):
        result = doctor_check(verbose=False, dry_run=False, filters=["cudf"])
        assert result is True


def test_doctor_check_with_warnings(capsys):
    """Test doctor_check with checks that issue warnings."""
    mock_ep = MagicMock()
    mock_ep.name = "warning_check"
    mock_ep.value = "test.module:check"
    mock_ep.load.return_value = mock_warning_check

    with patch("rapids_cli.doctor.doctor.entry_points", return_value=[mock_ep]):
        result = doctor_check(verbose=False, dry_run=False)
        assert result is True

    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "This is a warning" in captured.out


def test_check_result_creation():
    """Test CheckResult dataclass creation."""
    result = CheckResult(
        name="test_check",
        description="Test check description",
        status=True,
        value="Success",
        error=None,
        warnings=None,
    )
    assert result.name == "test_check"
    assert result.description == "Test check description"
    assert result.status is True
    assert result.value == "Success"
    assert result.error is None
    assert result.warnings is None


def test_doctor_check_import_error():
    """Test that import errors are suppressed during check discovery."""
    mock_ep = MagicMock()
    mock_ep.name = "broken_check"
    mock_ep.value = "broken.module:check"
    mock_ep.load.side_effect = ImportError("Module not found")

    with patch("rapids_cli.doctor.doctor.entry_points", return_value=[mock_ep]):
        result = doctor_check(verbose=False, dry_run=False)
        # Should still pass with no checks discovered
        assert result is True
