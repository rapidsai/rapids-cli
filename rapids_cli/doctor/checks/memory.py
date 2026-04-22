# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory checks."""

import warnings

from rapids_cli.hardware import HardwareInfoError
from rapids_cli.providers import get_gpu_info, get_system_info


def get_system_memory(verbose=False):
    """Get the total system memory."""
    return get_system_info().total_memory_bytes / (1024**3)


def get_gpu_memory(verbose=False):
    """Get the total GPU memory."""
    return sum(dev.memory_total_bytes for dev in get_gpu_info().devices) / (1024**3)


def check_memory_to_gpu_ratio(verbose=True):
    """Check the system for a 2:1 ratio of system Memory to total GPU Memory.

    This is especially useful for Dask.
    """
    try:
        _ = get_gpu_info().device_count
    except HardwareInfoError as e:
        raise ValueError("GPU not found. Please ensure GPUs are installed.") from e

    ratio = get_system_memory() / get_gpu_memory()
    if ratio < 1.8:
        warnings.warn(
            "System Memory to total GPU Memory ratio not at least 2:1 ratio. "
            "It is recommended to have double the system memory to GPU memory for optimal performance.",
            stacklevel=2,
        )
    return True
