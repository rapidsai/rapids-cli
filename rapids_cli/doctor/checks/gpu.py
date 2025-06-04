# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU checks for the doctor command."""

import pynvml


def gpu_check(verbose=False):
    """Check GPU availability."""
    try:
        pynvml.nvmlInit()
        num_gpus = pynvml.nvmlDeviceGetCount()
    except pynvml.NVMLError as e:
        raise ValueError("No available GPUs detected") from e
    assert num_gpus > 0, "No GPUs detected"
    return f"GPU(s) detected: {num_gpus}"


def check_gpu_compute_capability(verbose):
    """Check the system for GPU Compute Capability."""
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        raise ValueError("No GPU - cannot determineg GPU Compute Capability") from e

    required_capability = 7
    num_gpus = pynvml.nvmlDeviceGetCount()
    for i in range(num_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        compute_capability = major * 10 + minor

        if compute_capability >= required_capability:
            continue
        else:
            raise ValueError(
                f"GPU {i} does not meet the required compute "
                f"capability {required_capability[0]}.{required_capability[1]}."
            )
    return True
