# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory checks."""

import warnings

import psutil

from cuda.core import system


def get_system_memory(verbose=False):
    """Get the total system memory."""
    virtual_memory = psutil.virtual_memory()
    total_memory = virtual_memory.total / (1024**3)  # converts bytes to gigabytes
    return total_memory


def get_gpu_memory(verbose=False):
    """Get the total GPU memory."""

    gpu_memory_total = 0
    for device in system.Device.get_all_devices():
        gpu_memory_total += device.memory_info.total / (1024**3)  # converts to gigabytes

    return gpu_memory_total


def check_memory_to_gpu_ratio(verbose=True):
    """Check the system for a 2:1 ratio of system Memory to total GPU Memory.

    This is especially useful for Dask.

    """
    try:
        if system.Device.get_device_count() == 0:
            raise system.NvmlError(1)
    except system.NvmlError:
        raise ValueError("GPU not found. Please ensure GPUs are installed.")

    system_memory = get_system_memory(verbose)
    gpu_memory = get_gpu_memory(verbose)
    ratio = system_memory / gpu_memory
    if ratio < 1.8:
        warnings.warn(
            "System Memory to total GPU Memory ratio not at least 2:1 ratio. "
            "It is recommended to have double the system memory to GPU memory for optimal performance.",
            stacklevel=2,
        )
    return True
