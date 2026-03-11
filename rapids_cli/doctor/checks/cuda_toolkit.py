# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check CUDA toolkit library availability and version consistency."""

import os
import re
from pathlib import Path

import pynvml

# Core libraries to check for findability.
_CUDA_LIBS = {
    "cudart": "libcudart.so",
    "nvrtc": "libnvrtc.so",
    "nvvm": "libnvvm.so",
}

_CUDA_SYMLINK = Path("/usr/local/cuda")

# Maps cuda-pathfinder's found_via values to human-readable source labels.
_SOURCE_LABELS = {
    "conda": "conda",
    "site-packages": "pip",
    "system": "system",
    "CUDA_HOME": "CUDA_HOME",
}


def _get_source_label(found_via: str | None) -> str | None:
    """Map cuda-pathfinder's found_via to a human-readable source label."""
    if found_via:
        for key, label in _SOURCE_LABELS.items():
            if key in found_via:
                return label
    return None


def _format_mismatch_error(
    toolkit_major: int,
    driver_major: int,
    found_via: str | None,
    cudart_path: str | None,
) -> str:
    """Build a clear error message for toolkit > driver version mismatch."""
    source = _get_source_label(found_via)

    location = f"CUDA {toolkit_major} toolkit"
    if source and cudart_path:
        location += f" (found via {source} at {cudart_path})"
    elif source:
        location += f" (found via {source})"
    elif cudart_path:
        location += f" (at {cudart_path})"

    return (
        f"{location} is newer than what the GPU driver supports (CUDA {driver_major}). "
        f"Either update the GPU driver to one that supports CUDA {toolkit_major}, "
        f"or recreate your environment with CUDA {driver_major} packages."
    )


def _format_missing_error(missing_libs: list[str], found_via: str | None) -> str:
    """Build a clear error message for missing CUDA libraries."""
    source = _get_source_label(found_via)
    missing_str = ", ".join(missing_libs)

    if source:
        return (
            f"A {source} CUDA installation was detected, but {missing_str} could not be found. "
            f"Try reinstalling the CUDA packages in your {source} environment."
        )

    return (
        f"Some CUDA libraries ({missing_str}) could not be found. "
        "Install the CUDA Toolkit, or use conda/pip which manage CUDA automatically."
    )


def _get_toolkit_cuda_major(cudart_path: str | None = None) -> int | None:
    """Return the CUDA major version of the toolkit.

    Tries two strategies in order:
    1. Parse #define CUDA_VERSION from cuda_runtime_version.h (precise, needs dev headers)
    2. Call cudaRuntimeGetVersion via ctypes on the loaded libcudart.so

    Args:
        cudart_path: Absolute path to libcudart.so from cuda-pathfinder, used as fallback.
    """
    import ctypes

    import cuda.pathfinder

    # header parsing
    header_dir = cuda.pathfinder.find_nvidia_header_directory("cudart")
    if header_dir is not None:
        version_file = Path(header_dir) / "cuda_runtime_version.h"
        if version_file.exists():
            match = re.search(
                r"#define\s+CUDA_VERSION\s+(\d+)", version_file.read_text()
            )
            if match:
                return int(match.group(1)) // 1000

    # if header parsing fails, call cudaRuntimeGetVersion via ctypes
    if cudart_path is not None:
        try:
            libcudart = ctypes.CDLL(cudart_path)
            version = ctypes.c_int()
            if libcudart.cudaRuntimeGetVersion(ctypes.byref(version)) == 0:
                return version.value // 1000
        except OSError:
            pass

    return None


def _extract_major_from_cuda_path(path: Path) -> int | None:
    """Extract CUDA major version from a path like /usr/local/cuda-12.4 or its version.txt."""
    match = re.search(r"cuda-(\d+)", str(path))
    if match:
        return int(match.group(1))
    version_file = path / "version.txt"
    if version_file.exists():
        match = re.search(r"(\d+)\.", version_file.read_text())
        if match:
            return int(match.group(1))
    return None


def cuda_toolkit_check(verbose=False):
    """Check CUDA toolkit library availability and version consistency."""
    import cuda.pathfinder
    from cuda.pathfinder import DynamicLibNotFoundError

    # Check library findability
    found_via = {}
    cudart_path = None
    missing = []
    for libname, soname in _CUDA_LIBS.items():
        try:
            loaded = cuda.pathfinder.load_nvidia_dynamic_lib(libname)
            found_via[libname] = loaded.found_via
            if libname == "cudart":
                cudart_path = loaded.abs_path
        except (DynamicLibNotFoundError, RuntimeError):
            missing.append(soname)

    if missing:
        any_found_via = next(iter(found_via.values()), None)
        raise ValueError(_format_missing_error(missing, any_found_via))

    # Get driver CUDA major version
    try:
        pynvml.nvmlInit()
        driver_major = pynvml.nvmlSystemGetCudaDriverVersion() // 1000
    except pynvml.NVMLError as e:
        raise ValueError(
            "Unable to query the GPU driver's CUDA version. "
            "RAPIDS requires a working NVIDIA GPU driver."
        ) from e

    # Get toolkit CUDA major version and compare to driver
    # Only error when toolkit > driver (drivers are backward compatible)
    toolkit_major = _get_toolkit_cuda_major(cudart_path)
    if toolkit_major is not None and toolkit_major > driver_major:
        raise ValueError(
            _format_mismatch_error(
                toolkit_major, driver_major, found_via.get("cudart"), cudart_path
            )
        )

    # Check /usr/local/cuda symlink
    if _CUDA_SYMLINK.exists():
        sym_major = _extract_major_from_cuda_path(_CUDA_SYMLINK.resolve())
        if sym_major is not None and sym_major > driver_major:
            raise ValueError(
                f"/usr/local/cuda points to CUDA {sym_major} but the GPU driver "
                f"only supports up to CUDA {driver_major}. "
                f"Update the symlink to a CUDA {driver_major}.x installation."
            )

    # Check CUDA_HOME / CUDA_PATH
    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        env_val = os.environ.get(env_var)
        if env_val:
            env_major = _extract_major_from_cuda_path(Path(env_val))
            if env_major is not None and env_major > driver_major:
                raise ValueError(
                    f"{env_var}={env_val} (CUDA {env_major}) but the GPU driver "
                    f"only supports up to CUDA {driver_major}. "
                    f"Set {env_var} to a CUDA {driver_major}.x path."
                )

    if verbose:
        version_str = f"CUDA {toolkit_major}" if toolkit_major else "unknown version"
        return f"CUDA toolkit OK ({version_str}). Driver supports CUDA {driver_major}."
    return True
