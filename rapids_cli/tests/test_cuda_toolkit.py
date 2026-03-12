# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import patch

import pytest

from rapids_cli.doctor.checks.cuda_toolkit import (
    CudaToolkitInfo,
    _ctypes_cuda_version,
    _get_toolkit_cuda_major,
    cuda_toolkit_check,
)


def _make_info(**overrides):
    """Build a CudaToolkitInfo with sensible defaults. Override any field."""
    defaults = {
        "found_libs": {"cudart": "conda", "nvrtc": "conda", "nvvm": "conda"},
        "cudart_path": "/usr/lib/libcudart.so",
        "missing_libs": [],
        "driver_major": 12,
        "toolkit_major": 12,
    }
    defaults.update(overrides)
    return CudaToolkitInfo(**defaults)


# Version detection tests


def test_get_toolkit_version_from_headers(tmp_path):
    (tmp_path / "cuda_runtime_version.h").write_text("#define CUDA_VERSION 12040\n")
    with patch(
        "cuda.pathfinder.find_nvidia_header_directory", return_value=str(tmp_path)
    ):
        assert _get_toolkit_cuda_major() == 12


def test_get_toolkit_version_no_headers_falls_back_to_ctypes():
    """When headers unavailable, falls through to ctypes cudaRuntimeGetVersion."""
    import ctypes

    version_obj = ctypes.c_int(0)
    with (
        patch("cuda.pathfinder.find_nvidia_header_directory", return_value=None),
        patch("ctypes.CDLL") as mock_cdll,
        patch("ctypes.c_int", return_value=version_obj),
    ):
        mock_lib = mock_cdll.return_value

        def fake_get_version(ref):
            version_obj.value = 13000
            return 0

        mock_lib.cudaRuntimeGetVersion = fake_get_version
        assert _get_toolkit_cuda_major("/usr/lib/libcudart.so") == 13


def test_get_toolkit_version_returns_none_when_unavailable():
    with patch("cuda.pathfinder.find_nvidia_header_directory", return_value=None):
        assert _get_toolkit_cuda_major() is None


def test_ctypes_cuda_version_oserror():
    """ctypes returns None when the library can't be loaded."""
    with patch("ctypes.CDLL", side_effect=OSError("not found")):
        assert _ctypes_cuda_version("/nonexistent/libcudart.so") is None


# Check function tests


def test_check_success():
    info = _make_info()
    result = cuda_toolkit_check(verbose=True, toolkit_info=info)
    assert isinstance(result, str)
    assert "CUDA 12" in result


@pytest.mark.parametrize(
    "found_libs, missing_libs, expected_match",
    [
        ({}, ["libcudart.so", "libnvrtc.so", "libnvvm.so"], "could not be found"),
        (
            {"cudart": "conda", "nvrtc": "conda"},
            ["libnvvm.so"],
            "conda CUDA installation",
        ),
    ],
    ids=["all_missing", "partial_missing"],
)
def test_check_missing_libs(found_libs, missing_libs, expected_match):
    info = _make_info(
        found_libs=found_libs,
        missing_libs=missing_libs,
        cudart_path=None if not found_libs else "/usr/lib/libcudart.so",
        toolkit_major=None if not found_libs else 12,
    )
    with pytest.raises(ValueError, match=expected_match):
        cuda_toolkit_check(toolkit_info=info)


def test_check_driver_query_fails():
    info = _make_info(driver_major=None)
    with pytest.raises(ValueError, match="Unable to query"):
        cuda_toolkit_check(toolkit_info=info)


def test_check_toolkit_newer_than_driver():
    """CUDA 13 toolkit + CUDA 12 driver = error."""
    info = _make_info(
        found_libs={"cudart": "conda", "nvrtc": "conda", "nvvm": "conda"},
        cudart_path="/usr/lib/libcudart.so.13",
        toolkit_major=13,
        driver_major=12,
    )
    with pytest.raises(ValueError, match="newer than what the GPU driver supports"):
        cuda_toolkit_check(toolkit_info=info)


def test_check_toolkit_older_than_driver_passes():
    """CUDA 12 toolkit + CUDA 13 driver = fine (backward compatible)."""
    info = _make_info(toolkit_major=12, driver_major=13)
    assert cuda_toolkit_check(verbose=False, toolkit_info=info) is True


def test_check_cuda_symlink_newer_than_driver(tmp_path):
    """Only checked when CUDA was found via system paths, not conda/pip."""
    symlink_target = tmp_path / "cuda-13.0"
    symlink_target.mkdir()
    symlink_path = tmp_path / "cuda"
    symlink_path.symlink_to(symlink_target)

    info = _make_info(
        found_libs={
            "cudart": "system-search",
            "nvrtc": "system-search",
            "nvvm": "system-search",
        },
        toolkit_major=12,
        driver_major=12,
    )
    with (
        patch("rapids_cli.doctor.checks.cuda_toolkit._CUDA_SYMLINK", symlink_path),
        patch.dict("os.environ", {}, clear=True),
    ):
        with pytest.raises(ValueError, match="points to CUDA 13"):
            cuda_toolkit_check(toolkit_info=info)


def test_check_cuda_home_newer_than_driver():
    """Only checked when CUDA was found via system paths, not conda/pip."""
    info = _make_info(
        found_libs={
            "cudart": "system-search",
            "nvrtc": "system-search",
            "nvvm": "system-search",
        },
        toolkit_major=12,
        driver_major=12,
    )
    with (
        patch(
            "rapids_cli.doctor.checks.cuda_toolkit._CUDA_SYMLINK", Path("/nonexistent")
        ),
        patch.dict("os.environ", {"CUDA_HOME": "/usr/local/cuda-13.0"}, clear=True),
    ):
        with pytest.raises(ValueError, match="CUDA_HOME"):
            cuda_toolkit_check(toolkit_info=info)
