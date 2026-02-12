# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from rapids_cli.doctor.checks.memory import (
    check_memory_to_gpu_ratio,
    get_gpu_memory,
    get_system_memory,
)
from rapids_cli.hardware import DeviceInfo, FailingGpuInfo, FakeGpuInfo, FakeSystemInfo


def test_get_system_memory():
    system_info = FakeSystemInfo(total_memory_bytes=32 * 1024**3)
    result = get_system_memory(verbose=False, system_info=system_info)
    assert result == 32.0


def test_get_gpu_memory_single_gpu():
    devices = [
        DeviceInfo(index=0, compute_capability=(7, 0), memory_total_bytes=16 * 1024**3)
    ]
    gpu_info = FakeGpuInfo(device_count=1, devices=devices)
    result = get_gpu_memory(verbose=False, gpu_info=gpu_info)
    assert result == 16.0


def test_get_gpu_memory_multiple_gpus():
    devices = [
        DeviceInfo(index=i, compute_capability=(7, 0), memory_total_bytes=16 * 1024**3)
        for i in range(4)
    ]
    gpu_info = FakeGpuInfo(device_count=4, devices=devices)
    result = get_gpu_memory(verbose=False, gpu_info=gpu_info)
    assert result == 64.0  # 16 GB * 4 GPUs


def test_check_memory_to_gpu_ratio_good_ratio():
    devices = [
        DeviceInfo(index=0, compute_capability=(7, 0), memory_total_bytes=32 * 1024**3)
    ]
    gpu_info = FakeGpuInfo(device_count=1, devices=devices)
    system_info = FakeSystemInfo(total_memory_bytes=64 * 1024**3)
    result = check_memory_to_gpu_ratio(
        verbose=True, gpu_info=gpu_info, system_info=system_info
    )
    assert result is True


def test_check_memory_to_gpu_ratio_warning():
    devices = [
        DeviceInfo(index=0, compute_capability=(7, 0), memory_total_bytes=32 * 1024**3)
    ]
    gpu_info = FakeGpuInfo(device_count=1, devices=devices)
    system_info = FakeSystemInfo(total_memory_bytes=32 * 1024**3)
    with pytest.warns(UserWarning, match="System Memory to total GPU Memory ratio"):
        result = check_memory_to_gpu_ratio(
            verbose=True, gpu_info=gpu_info, system_info=system_info
        )
        assert result is True


def test_check_memory_to_gpu_ratio_no_gpu():
    gpu_info = FailingGpuInfo()
    with pytest.raises(
        ValueError, match="GPU not found. Please ensure GPUs are installed."
    ):
        check_memory_to_gpu_ratio(verbose=False, gpu_info=gpu_info)
