# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check for NVLink status."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rapids_cli.hardware import GpuInfoProvider


def check_nvlink_status(
    verbose=True, *, gpu_info: GpuInfoProvider | None = None, **kwargs
):
    """Check the system for NVLink with 2 or more GPUs."""
    if gpu_info is None:  # pragma: no cover
        from rapids_cli.hardware import NvmlGpuInfo

        gpu_info = NvmlGpuInfo()

    try:
        device_count = gpu_info.device_count
    except ValueError as e:
        raise ValueError("GPU not found. Please ensure GPUs are installed.") from e

    if device_count < 2:
        return False

    for dev in gpu_info.devices:
        if any(dev.nvlink_states):
            return True

    return False
