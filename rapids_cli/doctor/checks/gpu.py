# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU checks for the doctor command."""

from cuda.core import system

REQUIRED_COMPUTE_CAPABILITY = 7


def gpu_check(verbose=False):
    """Check GPU availability."""
    try:
        num_gpus = system.Device.get_device_count()
    except system.NvmlError as e:
        raise ValueError("No available GPUs detected") from e
    assert num_gpus > 0, "No GPUs detected"
    return f"GPU(s) detected: {num_gpus}"


def check_gpu_compute_capability(verbose):
    """Check the system for GPU Compute Capability."""
    try:
        num_gpus = system.Device.get_device_count()
        if num_gpus == 0:
            raise system.NvmlError(1)
    except system.NvmlError as e:
        raise ValueError("No GPU - cannot determine GPU Compute Capability") from e

    for i, device in enumerate(system.Device.get_all_devices()):
        major, minor = device.cuda_compute_capability
        if major >= REQUIRED_COMPUTE_CAPABILITY:
            continue
        else:
            raise ValueError(
                f"GPU {i} requires compute capability {REQUIRED_COMPUTE_CAPABILITY} "
                f"or higher but only has {major}.{minor}."
                "See https://developer.nvidia.com/cuda-gpus for more information."
            )
    return True
