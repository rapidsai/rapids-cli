# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from rapids_cli.doctor.checks.gpu import (
    REQUIRED_COMPUTE_CAPABILITY,
    check_gpu_compute_capability,
    gpu_check,
)
from rapids_cli.hardware import DeviceInfo, FailingGpuInfo, FakeGpuInfo


def test_gpu_check_success():
    gpu_info = FakeGpuInfo(device_count=2)
    result = gpu_check(verbose=True, gpu_info=gpu_info)
    assert result == "GPU(s) detected: 2"


def test_gpu_check_no_gpus():
    gpu_info = FakeGpuInfo(device_count=0)
    with pytest.raises(AssertionError, match="No GPUs detected"):
        gpu_check(verbose=False, gpu_info=gpu_info)


def test_gpu_check_nvml_error():
    gpu_info = FailingGpuInfo()
    with pytest.raises(ValueError, match="No available GPUs detected"):
        gpu_check(verbose=False, gpu_info=gpu_info)


def test_check_gpu_compute_capability_success():
    devices = [
        DeviceInfo(
            index=0,
            compute_capability=(REQUIRED_COMPUTE_CAPABILITY, 5),
            memory_total_bytes=0,
        ),
        DeviceInfo(
            index=1,
            compute_capability=(REQUIRED_COMPUTE_CAPABILITY, 5),
            memory_total_bytes=0,
        ),
    ]
    gpu_info = FakeGpuInfo(device_count=2, devices=devices)
    result = check_gpu_compute_capability(verbose=True, gpu_info=gpu_info)
    assert result is True


def test_check_gpu_compute_capability_insufficient():
    devices = [
        DeviceInfo(index=0, compute_capability=(6, 0), memory_total_bytes=0),
    ]
    gpu_info = FakeGpuInfo(device_count=1, devices=devices)
    with pytest.raises(
        ValueError,
        match=f"GPU 0 requires compute capability {REQUIRED_COMPUTE_CAPABILITY}",
    ):
        check_gpu_compute_capability(verbose=False, gpu_info=gpu_info)


def test_check_gpu_compute_capability_no_gpu():
    gpu_info = FailingGpuInfo()
    with pytest.raises(
        ValueError, match="No GPU - cannot determine GPU Compute Capability"
    ):
        check_gpu_compute_capability(verbose=False, gpu_info=gpu_info)
