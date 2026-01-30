# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check for CUDA and driver compatibility."""

import pynvml


def cuda_check(verbose=False):
    """Check CUDA availability."""
    try:
        pynvml.nvmlInit()
        try:
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
            return cuda_version
        except pynvml.NVMLError as e:
            raise ValueError("Unable to look up CUDA version") from e
    except pynvml.NVMLError as e:
        raise ValueError("Unable to look up CUDA version") from e
