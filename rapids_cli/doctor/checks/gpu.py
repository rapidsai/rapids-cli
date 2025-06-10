# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU checks for the doctor command."""

import pynvml

REQUIRED_COMPUTE_CAPABILITY = 7


def gpu_check(verbose=False):
    """Check GPU availability."""
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        raise ValueError("No available GPUs detected") from e
    num_gpus = pynvml.nvmlDeviceGetCount()
    return f"GPU(s) detected: {num_gpus}"


def check_gpu_compute_capability(verbose):
    """Check the system for GPU Compute Capability."""
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        raise ValueError("No GPU - cannot determineg GPU Compute Capability") from e

    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        if major >= REQUIRED_COMPUTE_CAPABILITY:
            continue
        else:
            raise ValueError(
                f"GPU {i} requires compute capability {REQUIRED_COMPUTE_CAPABILITY} "
                f"or higher but only has {major}.{minor}."
                "See https://developer.nvidia.com/cuda-gpus for more information."
            )
    return True
