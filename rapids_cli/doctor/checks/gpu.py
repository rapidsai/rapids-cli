# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU checks for the doctor command."""

from rapids_cli.hardware import HardwareInfoError
from rapids_cli.providers import get_gpu_info

REQUIRED_COMPUTE_CAPABILITY = 7


def gpu_check(verbose=False):
    """Check GPU availability."""
    try:
        num_gpus = get_gpu_info().device_count
    except HardwareInfoError as e:
        raise ValueError("No available GPUs detected") from e
    assert num_gpus > 0, "No GPUs detected"
    return f"GPU(s) detected: {num_gpus}"


def check_gpu_compute_capability(verbose=False):
    """Check the system for GPU Compute Capability."""
    try:
        devices = get_gpu_info().devices
    except HardwareInfoError as e:
        raise ValueError("No GPU - cannot determine GPU Compute Capability") from e

    for dev in devices:
        if dev.compute_capability[0] >= REQUIRED_COMPUTE_CAPABILITY:
            continue
        raise ValueError(
            f"GPU {dev.index} requires compute capability {REQUIRED_COMPUTE_CAPABILITY} "
            f"or higher but only has {dev.compute_capability[0]}.{dev.compute_capability[1]}."
            "See https://developer.nvidia.com/cuda-gpus for more information."
        )
    return True
