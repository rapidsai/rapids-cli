# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest

from rapids_cli.doctor.checks.cuda_version_mismatch import (
    check_cuda_major_version_mismatch,
    _get_driver_cuda_major,
    _get_toolkit_cuda_major,
)


def test_versions_match():
    result = check_cuda_major_version_mismatch(
        get_driver_cuda_major=lambda: 12,
        get_toolkit_cuda_major=lambda: 12,
    )
    assert result is True


def test_toolkit_older_than_driver():
    with pytest.raises(ValueError, match="older than the driver"):
        check_cuda_major_version_mismatch(
            get_driver_cuda_major=lambda: 12,
            get_toolkit_cuda_major=lambda: 11,
        )


def test_toolkit_newer_than_driver():
    with pytest.raises(ValueError, match="CUDA toolkit major version"):
        check_cuda_major_version_mismatch(
            get_driver_cuda_major=lambda: 11,
            get_toolkit_cuda_major=lambda: 12,
        )


def test_verbose_output():
    result = check_cuda_major_version_mismatch(
        verbose=True,
        get_driver_cuda_major=lambda: 12,
        get_toolkit_cuda_major=lambda: 12,
    )
    assert isinstance(result, str)
    assert "12" in result


def test_get_driver_cuda_major():
    with (
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlSystemGetCudaDriverVersion", return_value=12040),
    ):
        assert _get_driver_cuda_major() == 12


def test_get_toolkit_cuda_major_no_header_dir():
    with patch("cuda.pathfinder.find_nvidia_header_directory", return_value=None):
        assert _get_toolkit_cuda_major() is None


def test_get_toolkit_cuda_major_file_missing(tmp_path):
    with patch("cuda.pathfinder.find_nvidia_header_directory", return_value=str(tmp_path)):
        assert _get_toolkit_cuda_major() is None


def test_get_toolkit_cuda_major_no_match(tmp_path):
    (tmp_path / "cuda_runtime_version.h").write_text("/* no version define here */")
    with patch("cuda.pathfinder.find_nvidia_header_directory", return_value=str(tmp_path)):
        assert _get_toolkit_cuda_major() is None


def test_get_toolkit_cuda_major_success(tmp_path):
    (tmp_path / "cuda_runtime_version.h").write_text("#define CUDA_VERSION 12040\n")
    with patch("cuda.pathfinder.find_nvidia_header_directory", return_value=str(tmp_path)):
        assert _get_toolkit_cuda_major() == 12
