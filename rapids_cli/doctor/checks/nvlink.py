# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check for NVLink status."""

from rapids_cli.hardware import HardwareInfoError
from rapids_cli.providers import get_gpu_info


def check_nvlink_status(verbose=True, **kwargs):
    """Check NVLink status across all GPUs."""
    gpu_info = get_gpu_info()
    try:
        device_count = gpu_info.device_count
    except HardwareInfoError as e:
        raise ValueError("GPU not found. Please ensure GPUs are installed.") from e

    # NVLink requires at least 2 GPUs to be meaningful. A single GPU has nothing
    # to link to, so there is nothing to check.
    if device_count < 2:
        return False

    # Note: this check assumes a homogeneous GPU environment (all GPUs of the same
    # model). Mixed configurations — e.g. some NVLink-capable GPUs alongside some
    # that are not — are not handled and may produce misleading results.

    devices = gpu_info.devices

    # An empty nvlink_states means the driver reported NVLink as unsupported (or
    # no links were enumerated) for that device. Treat a system where no device
    # advertises links the same as the single-GPU case — nothing to check.
    if all(not dev.nvlink_states for dev in devices):
        return False

    failed_links: list[tuple[int, int]] = [
        (dev.index, link_id)
        for dev in devices
        for link_id, active in enumerate(dev.nvlink_states)
        if not active
    ]

    if failed_links:
        details = ", ".join(f"GPU {gpu} link {link}" for gpu, link in failed_links)
        raise ValueError(f"NVLink inactive on: {details}")

    if verbose:
        return f"All NVLinks active across {device_count} GPUs"
