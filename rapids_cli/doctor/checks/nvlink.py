# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check for NVLink status."""

import pynvml


def check_nvlink_status(verbose=True):
    """Check the system for NVLink with 2 or more GPUs."""
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count < 2:
            return False

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            for nvlink_id in range(pynvml.NVML_NVLINK_MAX_LINKS):
                try:
                    pynvml.nvmlDeviceGetNvLinkState(handle, 0)
                    return True
                except pynvml.NVMLError as e:
                    raise ValueError(f"NVLink {nvlink_id} Status Check Failed") from e
    except pynvml.NVMLError as e:
        raise ValueError("GPU not found. Please ensure GPUs are installed.") from e
