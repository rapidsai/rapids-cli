# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory checks."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rapids_cli.hardware import GpuInfoProvider, SystemInfoProvider


def get_system_memory(
    verbose=False, *, system_info: SystemInfoProvider | None = None, **kwargs
):
    """Get the total system memory."""
    if system_info is None:  # pragma: no cover
        from rapids_cli.hardware import DefaultSystemInfo

        system_info = DefaultSystemInfo()

    total_memory = system_info.total_memory_bytes / (
        1024**3
    )  # converts bytes to gigabytes
    return total_memory


def get_gpu_memory(verbose=False, *, gpu_info: GpuInfoProvider | None = None, **kwargs):
    """Get the total GPU memory."""
    if gpu_info is None:  # pragma: no cover
        from rapids_cli.hardware import NvmlGpuInfo

        gpu_info = NvmlGpuInfo()

    gpu_memory_total = sum(dev.memory_total_bytes for dev in gpu_info.devices) / (
        1024**3
    )  # converts to gigabytes
    return gpu_memory_total


def check_memory_to_gpu_ratio(
    verbose=True,
    *,
    gpu_info: GpuInfoProvider | None = None,
    system_info: SystemInfoProvider | None = None,
    **kwargs,
):
    """Check the system for a 2:1 ratio of system Memory to total GPU Memory.

    This is especially useful for Dask.

    """
    if gpu_info is None:  # pragma: no cover
        from rapids_cli.hardware import NvmlGpuInfo

        gpu_info = NvmlGpuInfo()
    if system_info is None:  # pragma: no cover
        from rapids_cli.hardware import DefaultSystemInfo

        system_info = DefaultSystemInfo()

    try:
        _ = gpu_info.device_count
    except ValueError as e:
        raise ValueError("GPU not found. Please ensure GPUs are installed.") from e

    system_memory = get_system_memory(verbose, system_info=system_info)
    gpu_memory = get_gpu_memory(verbose, gpu_info=gpu_info)
    ratio = system_memory / gpu_memory
    if ratio < 1.8:
        warnings.warn(
            "System Memory to total GPU Memory ratio not at least 2:1 ratio. "
            "It is recommended to have double the system memory to GPU memory for optimal performance.",
            stacklevel=2,
        )
    return True
