# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check for CUDA and driver compatibility."""

from __future__ import annotations

from rapids_cli.providers import get_gpu_info


def cuda_check(verbose=False, **kwargs):
    """Check CUDA availability."""
    try:
        return get_gpu_info().cuda_driver_version
    except ValueError as e:
        raise ValueError("Unable to look up CUDA version") from e
