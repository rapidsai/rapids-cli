# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU checks for the doctor command."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rapids_cli.hardware import GpuInfoProvider

REQUIRED_COMPUTE_CAPABILITY = 7


def gpu_check(verbose=False, *, gpu_info: GpuInfoProvider | None = None, **kwargs):
    """Check GPU availability."""
    if gpu_info is None:  # pragma: no cover
        from rapids_cli.hardware import NvmlGpuInfo

        gpu_info = NvmlGpuInfo()

    try:
        num_gpus = gpu_info.device_count
    except ValueError as e:
        raise ValueError("No available GPUs detected") from e
    assert num_gpus > 0, "No GPUs detected"
    return f"GPU(s) detected: {num_gpus}"


def check_gpu_compute_capability(
    verbose=False, *, gpu_info: GpuInfoProvider | None = None, **kwargs
):
    """Check the system for GPU Compute Capability."""
    if gpu_info is None:  # pragma: no cover
        from rapids_cli.hardware import NvmlGpuInfo

        gpu_info = NvmlGpuInfo()

    try:
        devices = gpu_info.devices
    except ValueError as e:
        raise ValueError("No GPU - cannot determine GPU Compute Capability") from e

    for dev in devices:
        if dev.compute_capability[0] >= REQUIRED_COMPUTE_CAPABILITY:
            continue
        else:
            raise ValueError(
                f"GPU {dev.index} requires compute capability {REQUIRED_COMPUTE_CAPABILITY} "
                f"or higher but only has {dev.compute_capability[0]}.{dev.compute_capability[1]}."
                "See https://developer.nvidia.com/cuda-gpus for more information."
            )
    return True
