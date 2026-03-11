# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pynvml
import pytest

from rapids_cli.doctor.checks.cuda_toolkit import cuda_toolkit_check


@dataclass
class FakeLoadedLib:
    """Mimics the return value of cuda.pathfinder.load_nvidia_dynamic_lib()."""

    abs_path: str | None = None
    found_via: str = "conda"
    was_already_loaded_from_elsewhere: bool = False


def _fake_loader(overrides=None):
    """Build a side_effect for load_nvidia_dynamic_lib. All 3 libs found by default."""
    from cuda.pathfinder import DynamicLibNotFoundError

    defaults = {"cudart": FakeLoadedLib(), "nvrtc": FakeLoadedLib(), "nvvm": FakeLoadedLib()}
    if overrides:
        defaults.update(overrides)

    def loader(libname):
        val = defaults[libname]
        if isinstance(val, Exception):
            raise val
        return val

    return loader


def test_check_success(tmp_path):
    (tmp_path / "cuda_runtime_version.h").write_text("#define CUDA_VERSION 12040\n")
    with (
        patch("cuda.pathfinder.load_nvidia_dynamic_lib", side_effect=_fake_loader()),
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlSystemGetCudaDriverVersion", return_value=12040),
        patch("cuda.pathfinder.find_nvidia_header_directory", return_value=str(tmp_path)),
        patch.dict("os.environ", {}, clear=True),
    ):
        result = cuda_toolkit_check(verbose=True)
        assert isinstance(result, str)
        assert "CUDA 12" in result


def test_check_missing_libs():
    from cuda.pathfinder import DynamicLibNotFoundError

    with patch(
        "cuda.pathfinder.load_nvidia_dynamic_lib",
        side_effect=_fake_loader({
            "cudart": DynamicLibNotFoundError("not found"),
            "nvrtc": DynamicLibNotFoundError("not found"),
            "nvvm": DynamicLibNotFoundError("not found"),
        }),
    ):
        with pytest.raises(ValueError, match="libcudart.so"):
            cuda_toolkit_check()


def test_check_driver_query_fails():
    with (
        patch("cuda.pathfinder.load_nvidia_dynamic_lib", side_effect=_fake_loader()),
        patch("pynvml.nvmlInit", side_effect=pynvml.NVMLError(1)),
    ):
        with pytest.raises(ValueError, match="Unable to query"):
            cuda_toolkit_check()


def test_check_toolkit_newer_than_driver(tmp_path):
    """CUDA 13 toolkit + CUDA 12 driver = error."""
    (tmp_path / "cuda_runtime_version.h").write_text("#define CUDA_VERSION 13000\n")
    with (
        patch("cuda.pathfinder.load_nvidia_dynamic_lib", side_effect=_fake_loader()),
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlSystemGetCudaDriverVersion", return_value=12040),
        patch("cuda.pathfinder.find_nvidia_header_directory", return_value=str(tmp_path)),
    ):
        with pytest.raises(ValueError, match="only supports up to CUDA 12"):
            cuda_toolkit_check()


def test_check_toolkit_older_than_driver_passes(tmp_path):
    """CUDA 12 toolkit + CUDA 13 driver = fine (backward compatible)."""
    (tmp_path / "cuda_runtime_version.h").write_text("#define CUDA_VERSION 12040\n")
    with (
        patch("cuda.pathfinder.load_nvidia_dynamic_lib", side_effect=_fake_loader()),
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlSystemGetCudaDriverVersion", return_value=13000),
        patch("cuda.pathfinder.find_nvidia_header_directory", return_value=str(tmp_path)),
        patch.dict("os.environ", {}, clear=True),
    ):
        assert cuda_toolkit_check(verbose=False) is True


def test_check_cuda_symlink_newer_than_driver(tmp_path):
    symlink_target = tmp_path / "cuda-13.0"
    symlink_target.mkdir()
    symlink_path = tmp_path / "cuda"
    symlink_path.symlink_to(symlink_target)

    with (
        patch("cuda.pathfinder.load_nvidia_dynamic_lib", side_effect=_fake_loader()),
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlSystemGetCudaDriverVersion", return_value=12040),
        patch("cuda.pathfinder.find_nvidia_header_directory", return_value=None),
        patch("rapids_cli.doctor.checks.cuda_toolkit._CUDA_SYMLINK", symlink_path),
        patch.dict("os.environ", {}, clear=True),
    ):
        with pytest.raises(ValueError, match="points to CUDA 13"):
            cuda_toolkit_check()


def test_check_cuda_home_newer_than_driver():
    with (
        patch("cuda.pathfinder.load_nvidia_dynamic_lib", side_effect=_fake_loader()),
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlSystemGetCudaDriverVersion", return_value=12040),
        patch("cuda.pathfinder.find_nvidia_header_directory", return_value=None),
        patch("rapids_cli.doctor.checks.cuda_toolkit._CUDA_SYMLINK", Path("/nonexistent")),
        patch.dict("os.environ", {"CUDA_HOME": "/usr/local/cuda-13.0"}, clear=True),
    ):
        with pytest.raises(ValueError, match="CUDA_HOME"):
            cuda_toolkit_check()
