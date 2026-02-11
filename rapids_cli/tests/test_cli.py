# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

from click.testing import CliRunner

from rapids_cli.cli import debug, doctor, rapids


def test_rapids_cli_help():
    """Test rapids CLI help output."""
    runner = CliRunner()
    result = runner.invoke(rapids, ["--help"])
    assert result.exit_code == 0
    assert "The Rapids CLI is a command-line interface for RAPIDS" in result.output


def test_doctor_command_help():
    """Test doctor command help output."""
    runner = CliRunner()
    result = runner.invoke(rapids, ["doctor", "--help"])
    assert result.exit_code == 0
    assert "Run health checks" in result.output


def test_debug_command_help():
    """Test debug command help output."""
    runner = CliRunner()
    result = runner.invoke(rapids, ["debug", "--help"])
    assert result.exit_code == 0
    assert "Gather debugging information" in result.output


def test_doctor_command_success():
    """Test doctor command with successful checks."""
    runner = CliRunner()
    with patch("rapids_cli.cli.doctor_check", return_value=True):
        result = runner.invoke(rapids, ["doctor"])
        assert result.exit_code == 0


def test_doctor_command_failure():
    """Test doctor command with failed checks."""
    runner = CliRunner()
    with patch("rapids_cli.cli.doctor_check", return_value=False):
        result = runner.invoke(rapids, ["doctor"])
        assert result.exit_code == 1
        assert "Health checks failed" in result.output


def test_doctor_command_verbose():
    """Test doctor command with verbose flag."""
    runner = CliRunner()
    with patch("rapids_cli.cli.doctor_check", return_value=True) as mock_check:
        result = runner.invoke(rapids, ["doctor", "--verbose"])
        assert result.exit_code == 0
        mock_check.assert_called_once_with(True, False, ())


def test_doctor_command_dry_run():
    """Test doctor command with dry-run flag."""
    runner = CliRunner()
    with patch("rapids_cli.cli.doctor_check", return_value=True) as mock_check:
        result = runner.invoke(rapids, ["doctor", "--dry-run"])
        assert result.exit_code == 0
        mock_check.assert_called_once_with(False, True, ())


def test_doctor_command_with_filters():
    """Test doctor command with filters."""
    runner = CliRunner()
    with patch("rapids_cli.cli.doctor_check", return_value=True) as mock_check:
        result = runner.invoke(rapids, ["doctor", "cudf", "cuml"])
        assert result.exit_code == 0
        mock_check.assert_called_once_with(False, False, ("cudf", "cuml"))


def test_debug_command_console():
    """Test debug command with console output."""
    runner = CliRunner()
    with patch("rapids_cli.cli.run_debug") as mock_debug:
        result = runner.invoke(rapids, ["debug"])
        assert result.exit_code == 0
        mock_debug.assert_called_once_with(output_format="console")


def test_debug_command_json():
    """Test debug command with JSON output."""
    runner = CliRunner()
    with patch("rapids_cli.cli.run_debug") as mock_debug:
        result = runner.invoke(rapids, ["debug", "--json"])
        assert result.exit_code == 0
        mock_debug.assert_called_once_with(output_format="json")


def test_doctor_standalone():
    """Test doctor command as standalone function."""
    runner = CliRunner()
    with patch("rapids_cli.cli.doctor_check", return_value=True):
        result = runner.invoke(doctor)
        assert result.exit_code == 0


def test_debug_standalone():
    """Test debug command as standalone function."""
    runner = CliRunner()
    with patch("rapids_cli.cli.run_debug"):
        result = runner.invoke(debug)
        assert result.exit_code == 0
