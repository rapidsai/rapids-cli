# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from rapids_cli.doctor.checks.memory import (
    check_memory_to_gpu_ratio,
    get_gpu_memory,
    get_system_memory,
)
from rapids_cli.hardware import DeviceInfo, FailingGpuInfo, FakeGpuInfo, FakeSystemInfo


def test_get_system_memory(set_system_info):
    set_system_info(FakeSystemInfo(total_memory_bytes=32 * 1024**3))
    assert get_system_memory(verbose=False) == 32.0


def test_get_gpu_memory_single_gpu(set_gpu_info):
    devices = [
        DeviceInfo(index=0, compute_capability=(7, 0), memory_total_bytes=16 * 1024**3)
    ]
    set_gpu_info(FakeGpuInfo(device_count=1, devices=devices))
    assert get_gpu_memory(verbose=False) == 16.0


def test_get_gpu_memory_multiple_gpus(set_gpu_info):
    devices = [
        DeviceInfo(index=i, compute_capability=(7, 0), memory_total_bytes=16 * 1024**3)
        for i in range(4)
    ]
    set_gpu_info(FakeGpuInfo(device_count=4, devices=devices))
    assert get_gpu_memory(verbose=False) == 64.0  # 16 GB * 4 GPUs


def test_check_memory_to_gpu_ratio_good_ratio(set_gpu_info, set_system_info):
    devices = [
        DeviceInfo(index=0, compute_capability=(7, 0), memory_total_bytes=32 * 1024**3)
    ]
    set_gpu_info(FakeGpuInfo(device_count=1, devices=devices))
    set_system_info(FakeSystemInfo(total_memory_bytes=64 * 1024**3))
    assert check_memory_to_gpu_ratio(verbose=True) is True


def test_check_memory_to_gpu_ratio_warning(set_gpu_info, set_system_info):
    devices = [
        DeviceInfo(index=0, compute_capability=(7, 0), memory_total_bytes=32 * 1024**3)
    ]
    set_gpu_info(FakeGpuInfo(device_count=1, devices=devices))
    set_system_info(FakeSystemInfo(total_memory_bytes=32 * 1024**3))
    with pytest.warns(UserWarning, match="System Memory to total GPU Memory ratio"):
        assert check_memory_to_gpu_ratio(verbose=True) is True


def test_check_memory_to_gpu_ratio_no_gpu(set_gpu_info):
    set_gpu_info(FailingGpuInfo())
    with pytest.raises(
        ValueError, match="GPU not found. Please ensure GPUs are installed."
    ):
        check_memory_to_gpu_ratio(verbose=False)
