# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import patch

from rapids_cli.debug.debug import (
    gather_command_output,
    gather_cuda_version,
    gather_package_versions,
    gather_tools,
    run_debug,
)
from rapids_cli.hardware import FakeGpuInfo, FakeSystemInfo


def test_gather_cuda_version():
    gpu_info = FakeGpuInfo(cuda_driver_version=12040)
    result = gather_cuda_version(gpu_info=gpu_info)
    assert result == "12.4"


def test_gather_cuda_version_with_patch():
    gpu_info = FakeGpuInfo(cuda_driver_version=12345)
    result = gather_cuda_version(gpu_info=gpu_info)
    assert result == "12.34.5"


def test_gather_package_versions():
    result = gather_package_versions()
    assert isinstance(result, dict)
    assert len(result) > 0
    # Check that rapids-cli is in the installed packages
    assert "rapids-cli" in result


def test_gather_command_output_success():
    result = gather_command_output(["echo", "test"])
    assert result == "test"


def test_gather_command_output_with_fallback():
    result = gather_command_output(["nonexistent_command"], fallback_output="fallback")
    assert result == "fallback"


def test_gather_command_output_no_fallback():
    result = gather_command_output(["nonexistent_command"])
    assert result is None


def test_gather_tools():
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
    gpu_info = FakeGpuInfo(
        device_count=1,
        cuda_driver_version=12040,
        driver_version="550.54.15",
    )
    system_info = FakeSystemInfo(
        total_memory_bytes=32 * 1024**3,
        cuda_runtime_path="/usr/local/cuda/include",
    )

    with (
        patch("pathlib.Path.glob", return_value=[]),
        patch("rapids_cli.debug.debug.gather_package_versions", return_value={}),
        patch("rapids_cli.debug.debug.gather_command_output", return_value=None),
        patch("rapids_cli.debug.debug.gather_tools", return_value={}),
        patch("pathlib.Path.read_text", return_value='NAME="Ubuntu"\nVERSION="22.04"'),
    ):
        run_debug(output_format="console", gpu_info=gpu_info, system_info=system_info)

    captured = capsys.readouterr()
    assert "RAPIDS Debug Information" in captured.out


def test_run_debug_json(capsys):
    gpu_info = FakeGpuInfo(
        device_count=1,
        cuda_driver_version=12040,
        driver_version="550.54.15",
    )
    system_info = FakeSystemInfo(
        total_memory_bytes=32 * 1024**3,
        cuda_runtime_path="/usr/local/cuda/include",
    )

    with (
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
        run_debug(output_format="json", gpu_info=gpu_info, system_info=system_info)

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert isinstance(output, dict)
    assert "date" in output
    assert "platform" in output
    assert "driver_version" in output
    assert "cuda_version" in output
    assert "package_versions" in output
