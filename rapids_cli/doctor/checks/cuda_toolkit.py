# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check CUDA toolkit library availability and version consistency."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

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
    "site-packages": "pip/uv",
    "system": "system",
    "CUDA_HOME": "CUDA_HOME",
}


@dataclass
class CudaToolkitInfo:
    """Gathered CUDA toolkit and driver information for the check to evaluate."""

    found_libs: dict[str, str] = field(default_factory=dict)  # libname -> found_via
    cudart_path: str | None = None
    missing_libs: list[str] = field(default_factory=list)
    driver_major: int | None = None
    toolkit_major: int | None = None


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
    details = [
        v
        for v in (
            f"found via {source}" if source else None,
            f"at {cudart_path}" if cudart_path else None,
        )
        if v
    ]
    if details:
        location += f" ({', '.join(details)})"
    return (
        f"{location} is newer than what the GPU driver supports (CUDA {driver_major}). "
        f"Either update the GPU driver to one that supports CUDA {toolkit_major}, "
        f"or recreate your environment with CUDA {driver_major} packages. "
        f"See https://docs.nvidia.com/deploy/cuda-compatibility/ for details."
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
        "Install the CUDA Toolkit from https://developer.nvidia.com/cuda-downloads, "
        "or use a conda environment which manages CUDA dependencies automatically. "
        "pip also bundles CUDA for cudf and cuml (recent versions)."
    )


def _ctypes_cuda_version(cudart_path: str) -> int | None:
    """Get CUDA major version by calling cudaRuntimeGetVersion via ctypes."""
    import ctypes

    try:
        libcudart = ctypes.CDLL(cudart_path)
        version = ctypes.c_int()
        if libcudart.cudaRuntimeGetVersion(ctypes.byref(version)) == 0:
            return version.value // 1000
    except OSError:
        # ctypes.CDLL raises OSError when the .so file cannot be loaded
        # (e.g. file missing, broken symlink, or incompatible architecture).
        pass
    return None


def _get_toolkit_cuda_major(cudart_path: str | None = None) -> int | None:
    """Return the CUDA major version of the toolkit.

    Tries two different methods:
    1. Parse #define CUDA_VERSION from cuda_runtime_version.h (precise, needs dev headers)
    2. Call cudaRuntimeGetVersion via ctypes on the loaded libcudart.so

    Args:
        cudart_path: Absolute path to libcudart.so from cuda-pathfinder, used as fallback.
    """
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
        return _ctypes_cuda_version(cudart_path)

    return None


def _extract_major_from_cuda_path(path: Path) -> int | None:
    """Extract CUDA major version from a path like /usr/local/cuda-12.4."""
    match = re.search(r"cuda-(\d+)", str(path))
    return int(match.group(1)) if match else None


def _check_path_version(label: str, path: Path, driver_major: int) -> None:
    """Raise if a CUDA path points to a version newer than the driver supports."""
    major = _extract_major_from_cuda_path(path)
    if major is not None and major > driver_major:
        raise ValueError(
            f"{label} points to CUDA {major} but the GPU driver "
            f"only supports up to CUDA {driver_major}. "
            f"Update {label} to a CUDA {driver_major}.x installation."
        )


def _gather_toolkit_info() -> CudaToolkitInfo:  # pragma: no cover
    """Gather CUDA toolkit and driver information from the real system."""
    import cuda.pathfinder
    from cuda.core.system import get_driver_version
    from cuda.pathfinder import DynamicLibNotFoundError

    info = CudaToolkitInfo()

    # Discover libraries
    for libname, soname in _CUDA_LIBS.items():
        try:
            loaded = cuda.pathfinder.load_nvidia_dynamic_lib(libname)
            info.found_libs[libname] = loaded.found_via
            if libname == "cudart":
                info.cudart_path = loaded.abs_path
        except (DynamicLibNotFoundError, RuntimeError):
            info.missing_libs.append(soname)

    # Get driver version
    try:
        info.driver_major = get_driver_version(kernel_mode=True)[0]
    except Exception:
        info.driver_major = None

    # Get toolkit version
    if not info.missing_libs:
        info.toolkit_major = _get_toolkit_cuda_major(info.cudart_path)

    return info


def cuda_toolkit_check(verbose=False, **kwargs):
    """Check CUDA toolkit library availability and version consistency."""
    from rapids_cli.providers import get_toolkit_info

    info = get_toolkit_info()

    # Check library findability
    if info.missing_libs:
        any_found_via = next(iter(info.found_libs.values()), None)
        raise ValueError(_format_missing_error(info.missing_libs, any_found_via))

    # Check driver availability
    if info.driver_major is None:
        raise ValueError(
            "Unable to query the GPU driver's CUDA version. "
            "RAPIDS requires a working NVIDIA GPU driver."
        )

    driver_major = info.driver_major
    toolkit_major = info.toolkit_major

    # Compare toolkit to driver (only error when toolkit > driver, drivers are backward compatible)
    if toolkit_major is not None and toolkit_major > driver_major:
        raise ValueError(
            _format_mismatch_error(
                toolkit_major,
                driver_major,
                info.found_libs.get("cudart"),
                info.cudart_path,
            )
        )

    # Only check system paths if CUDA was found via system/CUDA_HOME.
    # When found via conda or pip, RAPIDS uses those libs and ignores system paths.
    cudart_source = info.found_libs.get("cudart", "")
    if cudart_source not in ("conda", "site-packages"):
        if _CUDA_SYMLINK.exists():
            _check_path_version(
                "/usr/local/cuda", _CUDA_SYMLINK.resolve(), driver_major
            )

        for env_var in ("CUDA_HOME", "CUDA_PATH"):
            env_val = os.environ.get(env_var)
            if env_val:
                _check_path_version(f"{env_var}={env_val}", Path(env_val), driver_major)

    if verbose:
        version_str = f"CUDA {toolkit_major}" if toolkit_major else "unknown version"
        return f"CUDA toolkit OK ({version_str}). Driver supports CUDA {driver_major}."
    return True
