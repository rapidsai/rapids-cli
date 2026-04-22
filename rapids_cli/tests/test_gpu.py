# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from rapids_cli.doctor.checks.gpu import (
    REQUIRED_COMPUTE_CAPABILITY,
    check_gpu_compute_capability,
    gpu_check,
)
from rapids_cli.hardware import DeviceInfo
from rapids_cli.tests.fakes import FailingGpuInfo, FakeGpuInfo


def test_gpu_check_success(set_gpu_info):
    set_gpu_info(FakeGpuInfo(device_count=2))
    assert gpu_check(verbose=True) == "GPU(s) detected: 2"


def test_gpu_check_no_gpus(set_gpu_info):
    set_gpu_info(FakeGpuInfo(device_count=0))
    with pytest.raises(AssertionError, match="No GPUs detected"):
        gpu_check(verbose=False)


def test_gpu_check_nvml_error(set_gpu_info):
    set_gpu_info(FailingGpuInfo())
    with pytest.raises(ValueError, match="No available GPUs detected"):
        gpu_check(verbose=False)


def test_check_gpu_compute_capability_success(set_gpu_info):
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
    set_gpu_info(FakeGpuInfo(device_count=2, devices=devices))
    assert check_gpu_compute_capability(verbose=True) is True


def test_check_gpu_compute_capability_insufficient(set_gpu_info):
    devices = [
        DeviceInfo(index=0, compute_capability=(6, 0), memory_total_bytes=0),
    ]
    set_gpu_info(FakeGpuInfo(device_count=1, devices=devices))
    with pytest.raises(
        ValueError,
        match=f"GPU 0 requires compute capability {REQUIRED_COMPUTE_CAPABILITY}",
    ):
        check_gpu_compute_capability(verbose=False)


def test_check_gpu_compute_capability_no_gpu(set_gpu_info):
    set_gpu_info(FailingGpuInfo())
    with pytest.raises(
        ValueError, match="No GPU - cannot determine GPU Compute Capability"
    ):
        check_gpu_compute_capability(verbose=False)
