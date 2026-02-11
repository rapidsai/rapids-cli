# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from rapids_cli.doctor.checks.nvlink import check_nvlink_status
from rapids_cli.hardware import DeviceInfo, FailingGpuInfo, FakeGpuInfo


def _make_device(index: int, nvlink_states: list[bool]) -> DeviceInfo:
    return DeviceInfo(
        index=index,
        compute_capability=(7, 0),
        memory_total_bytes=0,
        nvlink_states=nvlink_states,
    )


@pytest.mark.parametrize(
    "verbose, expected",
    [
        (True, "All NVLinks active across 2 GPUs"),
        (False, None),
    ],
)
def test_check_nvlink_status_success(verbose, expected):
    """2 GPUs, all NVLinks active — verbose controls whether a summary string is returned."""
    # Simulate a V100 with 6 NVLink slots, all active.
    devices = [_make_device(0, [True] * 6), _make_device(1, [True] * 6)]
    gpu_info = FakeGpuInfo(device_count=2, devices=devices)
    result = check_nvlink_status(verbose=verbose, gpu_info=gpu_info)
    assert result == expected


def test_check_nvlink_status_single_gpu():
    """Single GPU — NVLink is not applicable, check skips early."""
    gpu_info = FakeGpuInfo(device_count=1, devices=[_make_device(0, [])])
    result = check_nvlink_status(verbose=False, gpu_info=gpu_info)
    assert result is False


def test_check_nvlink_status_no_gpu():
    """GPU info unavailable — surfaces as a GPU-not-found error."""
    gpu_info = FailingGpuInfo()
    with pytest.raises(
        ValueError, match="GPU not found. Please ensure GPUs are installed."
    ):
        check_nvlink_status(verbose=False, gpu_info=gpu_info)


def test_check_nvlink_status_not_supported():
    """NVLink not supported on any device — check skips silently like single-GPU case."""
    # When NVML reports NVLink as not supported, NvmlGpuInfo records an empty list.
    devices = [_make_device(0, []), _make_device(1, [])]
    gpu_info = FakeGpuInfo(device_count=2, devices=devices)
    result = check_nvlink_status(verbose=False, gpu_info=gpu_info)
    assert result is False


def test_check_nvlink_status_link_inactive():
    """A supported link is inactive — check fails and reports which GPU and link."""
    devices = [_make_device(0, [False] * 6), _make_device(1, [False] * 6)]
    gpu_info = FakeGpuInfo(device_count=2, devices=devices)
    with pytest.raises(ValueError, match="NVLink inactive on:"):
        check_nvlink_status(verbose=False, gpu_info=gpu_info)


def test_check_nvlink_status_partial_failure():
    """Some links active, some inactive — all failures are reported in a single error."""
    # V100 with 6 NVLink slots: link 0 active, link 1 inactive, rest active.
    states = [True, False, True, True, True, True]
    devices = [_make_device(0, states), _make_device(1, states)]
    gpu_info = FakeGpuInfo(device_count=2, devices=devices)
    with pytest.raises(ValueError, match="NVLink inactive on:") as exc_info:
        check_nvlink_status(verbose=False, gpu_info=gpu_info)
    # Both GPUs should have link 1 reported as failed.
    assert "GPU 0 link 1" in str(exc_info.value)
    assert "GPU 1 link 1" in str(exc_info.value)


def test_check_nvlink_status_mixed_link_counts():
    """Links of differing counts (e.g. A100=12) iterate fully and succeed when all active."""
    devices = [_make_device(0, [True] * 12), _make_device(1, [True] * 12)]
    gpu_info = FakeGpuInfo(device_count=2, devices=devices)
    result = check_nvlink_status(verbose=True, gpu_info=gpu_info)
    assert result == "All NVLinks active across 2 GPUs"
