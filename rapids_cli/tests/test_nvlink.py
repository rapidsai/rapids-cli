# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from rapids_cli.doctor.checks.nvlink import check_nvlink_status
from rapids_cli.hardware import DeviceInfo, FailingGpuInfo, FakeGpuInfo


def test_check_nvlink_status_success():
    devices = [
        DeviceInfo(
            index=0,
            compute_capability=(7, 0),
            memory_total_bytes=0,
            nvlink_states=[True],
        ),
        DeviceInfo(
            index=1,
            compute_capability=(7, 0),
            memory_total_bytes=0,
            nvlink_states=[True],
        ),
    ]
    gpu_info = FakeGpuInfo(device_count=2, devices=devices)
    result = check_nvlink_status(verbose=True, gpu_info=gpu_info)
    assert result is True


def test_check_nvlink_status_single_gpu():
    gpu_info = FakeGpuInfo(device_count=1)
    result = check_nvlink_status(verbose=False, gpu_info=gpu_info)
    assert result is False


def test_check_nvlink_status_no_gpu():
    gpu_info = FailingGpuInfo()
    with pytest.raises(
        ValueError, match="GPU not found. Please ensure GPUs are installed."
    ):
        check_nvlink_status(verbose=False, gpu_info=gpu_info)


def test_check_nvlink_status_no_nvlink():
    devices = [
        DeviceInfo(
            index=0, compute_capability=(7, 0), memory_total_bytes=0, nvlink_states=[]
        ),
        DeviceInfo(
            index=1, compute_capability=(7, 0), memory_total_bytes=0, nvlink_states=[]
        ),
    ]
    gpu_info = FakeGpuInfo(device_count=2, devices=devices)
    result = check_nvlink_status(verbose=True, gpu_info=gpu_info)
    assert result is False
