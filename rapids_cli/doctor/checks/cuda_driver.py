# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check for CUDA and driver compatibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rapids_cli.hardware import GpuInfoProvider


def cuda_check(verbose=False, *, gpu_info: GpuInfoProvider | None = None, **kwargs):
    """Check CUDA availability."""
    if gpu_info is None:  # pragma: no cover
        from rapids_cli.hardware import NvmlGpuInfo

        gpu_info = NvmlGpuInfo()

    try:
        return gpu_info.cuda_driver_version
    except ValueError as e:
        raise ValueError("Unable to look up CUDA version") from e
