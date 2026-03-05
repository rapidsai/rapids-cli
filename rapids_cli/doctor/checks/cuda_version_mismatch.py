# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check for CUDA toolkit vs driver major version mismatch."""

import re
import warnings
from pathlib import Path

import cuda.pathfinder
import pynvml


def _get_driver_cuda_major() -> int:
    """Return the CUDA major version supported by the installed driver via pynvml."""
    pynvml.nvmlInit()
    return pynvml.nvmlSystemGetCudaDriverVersion() // 1000


def _get_toolkit_cuda_major() -> int | None:
    """Return the CUDA major version of the toolkit found via cuda-pathfinder, or None."""
    header_dir = cuda.pathfinder.find_nvidia_header_directory("cudart")
    if header_dir is None:
        return None
    version_file = Path(header_dir) / "cuda_runtime_version.h"
    if not version_file.exists():
        return None
    match = re.search(r"#define\s+CUDA_VERSION\s+(\d+)", version_file.read_text())
    return int(match.group(1)) // 1000 if match else None


def check_cuda_major_version_mismatch(
    verbose=False,
    get_driver_cuda_major=_get_driver_cuda_major,
    get_toolkit_cuda_major=_get_toolkit_cuda_major,
    **kwargs,
) -> bool | str:
    """Check that the CUDA toolkit major version matches the driver's supported CUDA major version.

    Args:
        verbose: If True, return a descriptive string on success.
        get_driver_cuda_major: Callable returning the driver's max CUDA major version.
        get_toolkit_cuda_major: Callable returning the toolkit CUDA major version, or None.
        **kwargs: Accepted for forward compatibility.

    Returns:
        True on success, or a descriptive string if verbose is True.

    Raises:
        ValueError: If no driver is found or if the toolkit and driver major versions differ.
    """
    try:
        driver_major = get_driver_cuda_major()
    except pynvml.NVMLError as e:
        raise ValueError("Unable to determine driver CUDA version. Is an NVIDIA driver installed?") from e

    toolkit_major = get_toolkit_cuda_major()
    if toolkit_major is None:
        warnings.warn("No CUDA toolkit found in the current environment; skipping toolkit version check.")
        return True

    if toolkit_major > driver_major:
        raise ValueError(
            f"CUDA toolkit major version ({toolkit_major}) is newer than what the installed driver supports "
            f"({driver_major}). Update your NVIDIA driver to one that supports CUDA {toolkit_major} or "
            f"downgrade your CUDA toolkit to CUDA {driver_major}."
        )

    if toolkit_major < driver_major:
        raise ValueError(
            f"CUDA toolkit major version ({toolkit_major}) is older than the driver's supported CUDA major version "
            f"({driver_major}). Upgrade your CUDA toolkit to CUDA {driver_major} or "
            f"downgrade your NVIDIA driver to one that supports CUDA {toolkit_major}."
        )

    if verbose:
        return f"CUDA toolkit major version ({toolkit_major}) matches driver CUDA major version ({driver_major})."

    return True
