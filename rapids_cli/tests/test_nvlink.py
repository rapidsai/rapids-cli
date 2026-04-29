# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from rapids_cli.doctor.checks.nvlink import check_nvlink_status
from rapids_cli.hardware import DeviceInfo
from rapids_cli.tests.fakes import FailingGpuInfo, FakeGpuInfo


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
def test_check_nvlink_status_success(set_gpu_info, verbose, expected):
    """2 GPUs, all NVLinks active — verbose controls whether a summary string is returned."""
    # Simulate a V100 with 6 NVLink slots, all active.
    devices = [_make_device(0, [True] * 6), _make_device(1, [True] * 6)]
    set_gpu_info(FakeGpuInfo(device_count=2, devices=devices))
    assert check_nvlink_status(verbose=verbose) == expected


def test_check_nvlink_status_single_gpu(set_gpu_info):
    """Single GPU — NVLink is not applicable, check skips early."""
    set_gpu_info(FakeGpuInfo(device_count=1, devices=[_make_device(0, [])]))
    assert check_nvlink_status(verbose=False) is False


def test_check_nvlink_status_no_gpu(set_gpu_info):
    """GPU info unavailable — surfaces as a GPU-not-found error."""
    set_gpu_info(FailingGpuInfo())
    with pytest.raises(
        ValueError, match="GPU not found. Please ensure GPUs are installed."
    ):
        check_nvlink_status(verbose=False)


def test_check_nvlink_status_not_supported(set_gpu_info):
    """NVLink not supported on any device — check skips silently like single-GPU case."""
    # When NVML reports NVLink as not supported, NvmlGpuInfo records an empty list.
    devices = [_make_device(0, []), _make_device(1, [])]
    set_gpu_info(FakeGpuInfo(device_count=2, devices=devices))
    assert check_nvlink_status(verbose=False) is False


def test_check_nvlink_status_link_inactive(set_gpu_info):
    """A supported link is inactive — check fails and reports which GPU and link."""
    devices = [_make_device(0, [False] * 6), _make_device(1, [False] * 6)]
    set_gpu_info(FakeGpuInfo(device_count=2, devices=devices))
    with pytest.raises(ValueError, match="NVLink inactive on:"):
        check_nvlink_status(verbose=False)


def test_check_nvlink_status_partial_failure(set_gpu_info):
    """Some links active, some inactive — all failures are reported in a single error."""
    # V100 with 6 NVLink slots: link 0 active, link 1 inactive, rest active.
    states = [True, False, True, True, True, True]
    devices = [_make_device(0, states), _make_device(1, states)]
    set_gpu_info(FakeGpuInfo(device_count=2, devices=devices))
    with pytest.raises(ValueError, match="NVLink inactive on:") as exc_info:
        check_nvlink_status(verbose=False)
    assert "GPU 0 link 1" in str(exc_info.value)
    assert "GPU 1 link 1" in str(exc_info.value)


def test_check_nvlink_status_mixed_link_counts(set_gpu_info):
    """Links of differing counts (e.g. A100=12) iterate fully and succeed when all active."""
    devices = [_make_device(0, [True] * 12), _make_device(1, [True] * 12)]
    set_gpu_info(FakeGpuInfo(device_count=2, devices=devices))
    assert check_nvlink_status(verbose=True) == "All NVLinks active across 2 GPUs"
