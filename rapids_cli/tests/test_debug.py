# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import MagicMock, patch

from rapids_cli.debug.debug import (
    gather_command_output,
    gather_cuda_version,
    gather_package_versions,
    gather_tools,
    run_debug,
)


def test_gather_cuda_version():
    """Test CUDA version gathering."""
    with patch("pynvml.nvmlSystemGetCudaDriverVersion", return_value=12040):
        result = gather_cuda_version()
        assert result == "12.4"


def test_gather_cuda_version_with_patch():
    """Test CUDA version with patch number."""
    with patch("pynvml.nvmlSystemGetCudaDriverVersion", return_value=12345):
        result = gather_cuda_version()
        assert result == "12.34.5"


def test_gather_package_versions():
    """Test package version gathering."""
    result = gather_package_versions()
    assert isinstance(result, dict)
    assert len(result) > 0
    # Check that rapids-cli is in the installed packages
    assert "rapids-cli" in result


def test_gather_command_output_success():
    """Test successful command output gathering."""
    result = gather_command_output(["echo", "test"])
    assert result == "test"


def test_gather_command_output_with_fallback():
    """Test command output with fallback."""
    result = gather_command_output(["nonexistent_command"], fallback_output="fallback")
    assert result == "fallback"


def test_gather_command_output_no_fallback():
    """Test command output without fallback."""
    result = gather_command_output(["nonexistent_command"])
    assert result is None


def test_gather_tools():
    """Test tools gathering."""
    with (
        patch(
            "rapids_cli.debug.debug.gather_command_output",
            side_effect=lambda cmd, **kwargs: f"{cmd[0]} version",
        ),
    ):
        result = gather_tools()
        assert isinstance(result, dict)
        assert "pip" in result
        assert "conda" in result
        assert "g++" in result


def test_run_debug_console(capsys):
    """Test run_debug with console output."""
    mock_vm = MagicMock()
    mock_vm.total = 32 * 1024**3

    with (
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlSystemGetDriverVersion", return_value="550.54.15"),
        patch("pynvml.nvmlSystemGetCudaDriverVersion", return_value=12040),
        patch(
            "cuda.pathfinder.find_nvidia_header_directory",
            return_value="/usr/local/cuda/include",
        ),
        patch("pathlib.Path.glob", return_value=[]),
        patch("rapids_cli.debug.debug.gather_package_versions", return_value={}),
        patch("rapids_cli.debug.debug.gather_command_output", return_value=None),
        patch("rapids_cli.debug.debug.gather_tools", return_value={}),
        patch("pathlib.Path.read_text", return_value='NAME="Ubuntu"\nVERSION="22.04"'),
    ):
        run_debug(output_format="console")

    captured = capsys.readouterr()
    assert "RAPIDS Debug Information" in captured.out


def test_run_debug_json(capsys):
    """Test run_debug with JSON output."""
    with (
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlSystemGetDriverVersion", return_value="550.54.15"),
        patch("pynvml.nvmlSystemGetCudaDriverVersion", return_value=12040),
        patch(
            "cuda.pathfinder.find_nvidia_header_directory",
            return_value="/usr/local/cuda/include",
        ),
        patch("pathlib.Path.glob", return_value=[]),
        patch(
            "rapids_cli.debug.debug.gather_package_versions",
            return_value={"test": "1.0"},
        ),
        patch(
            "rapids_cli.debug.debug.gather_command_output", return_value="test output"
        ),
        patch("rapids_cli.debug.debug.gather_tools", return_value={"pip": "pip 23.0"}),
        patch("pathlib.Path.read_text", return_value='NAME="Ubuntu"\nVERSION="22.04"'),
    ):
        run_debug(output_format="json")

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert isinstance(output, dict)
    assert "date" in output
    assert "platform" in output
    assert "driver_version" in output
    assert "cuda_version" in output
    assert "package_versions" in output
