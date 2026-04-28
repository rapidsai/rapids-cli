# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check for CUDA and driver compatibility."""

from cuda.core import system


def cuda_check(verbose=False):
    """Check CUDA availability."""
    
    try:
        cuda_version = system.get_driver_version_full(kernel_mode=True)
        return cuda_version[0] * 1000 + cuda_version[1] * 10 + cuda_version[2]
    except system.NvmlError as e:
        raise ValueError("Unable to look up CUDA version") from e
