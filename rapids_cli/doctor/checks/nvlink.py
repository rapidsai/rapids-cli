# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check for NVLink status."""

import pynvml


def check_nvlink_status(verbose=True, **kwargs):
    """Check NVLink status across all GPUs."""
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        raise ValueError("GPU not found. Please ensure GPUs are installed.") from e

    device_count = pynvml.nvmlDeviceGetCount()

    # NVLink requires at least 2 GPUs to be meaningful. A single GPU has nothing
    # to link to, so there is nothing to check.
    if device_count < 2:
        return False

    # Note: this check assumes a homogeneous GPU environment (all GPUs of the same
    # model). Mixed configurations — e.g. some NVLink-capable GPUs alongside some
    # that are not — are not handled and may produce misleading results.

    failed_links: list[tuple[int, int]] = []

    for gpu_idx in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        # NVML provides no API to query the number of NVLink slots on a device
        # (e.g. V100=6, A100=12, H100=18). The only way to discover the real count
        # is to iterate up to NVML_NVLINK_MAX_LINKS and stop when the driver signals
        # that link_id is out of range via NVMLError_InvalidArgument.
        for link_id in range(pynvml.NVML_NVLINK_MAX_LINKS):
            try:
                # nvmlDeviceGetNvLinkState(device, link) returns NVML_FEATURE_ENABLED
                # if the link is active, or NVML_FEATURE_DISABLED if it is not.
                state = pynvml.nvmlDeviceGetNvLinkState(handle, link_id)
                if state == pynvml.NVML_FEATURE_DISABLED:
                    failed_links.append((gpu_idx, link_id))
            except pynvml.NVMLError_NotSupported:
                # The driver reports NVLink is not supported on this system.
                # There is nothing to check — skip like the single-GPU case above.
                return False
            except pynvml.NVMLError_InvalidArgument:
                # link_id exceeds the number of NVLink slots on this device.
                # Stop iterating links for this GPU.
                break

    if failed_links:
        details = ", ".join(f"GPU {gpu} link {link}" for gpu, link in failed_links)
        raise ValueError(f"NVLink inactive on: {details}")

    if verbose:
        return f"All NVLinks active across {device_count} GPUs"
