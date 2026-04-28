# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest

from rapids_cli.doctor.checks.gpu import (
    REQUIRED_COMPUTE_CAPABILITY,
    check_gpu_compute_capability,
    gpu_check,
)


def test_gpu_check_success():
    with (
        patch("cuda.bindings.nvml.init_v2"),
        patch("cuda.bindings.nvml.device_get_count_v2", return_value=2),
    ):
        result = gpu_check(verbose=True)
        assert result == "GPU(s) detected: 2"


def test_gpu_check_no_gpus():
    with (
        patch("cuda.bindings.nvml.init_v2"),
        patch("cuda.bindings.nvml.device_get_count_v2", return_value=0),
    ):
        with pytest.raises(AssertionError, match="No GPUs detected"):
            gpu_check(verbose=False)


def test_gpu_check_nvml_error():
    from cuda.bindings import nvml

    with patch("cuda.bindings.nvml.init_v2", side_effect=nvml.NvmlError(1)):
        with pytest.raises(ValueError, match="No available GPUs detected"):
            gpu_check(verbose=False)


def test_check_gpu_compute_capability_success():
    with (
        patch("cuda.bindings.nvml.init_v2"),
        patch("cuda.bindings.nvml.device_get_count_v2", return_value=2),
        patch("cuda.bindings.nvml.device_get_handle_by_index_v2", return_value=0xffffffff),
        patch(
            "cuda.bindings.nvml.device_get_cuda_compute_capability",
            return_value=(REQUIRED_COMPUTE_CAPABILITY, 5),
        ),
    ):
        result = check_gpu_compute_capability(verbose=True)
        assert result is True


def test_check_gpu_compute_capability_insufficient():
    with (
        patch("cuda.bindings.nvml.init_v2"),
        patch("cuda.bindings.nvml.device_get_count_v2", return_value=1),
        patch("cuda.bindings.nvml.device_get_handle_by_index_v2", return_value=0xffffffff),
        patch("cuda.bindings.nvml.device_get_cuda_compute_capability", return_value=(6, 0)),
    ):
        with pytest.raises(
            ValueError,
            match=f"GPU 0 requires compute capability {REQUIRED_COMPUTE_CAPABILITY}",
        ):
            check_gpu_compute_capability(verbose=False)


def test_check_gpu_compute_capability_no_gpu():
    from cuda.bindings import nvml

    with patch("cuda.bindings.nvml.init_v2", side_effect=nvml.NvmlError(1)):
        with pytest.raises(
            ValueError, match="No GPU - cannot determine GPU Compute Capability"
        ):
            check_gpu_compute_capability(verbose=False)
