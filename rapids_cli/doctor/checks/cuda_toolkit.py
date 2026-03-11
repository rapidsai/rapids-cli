# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check CUDA toolkit library availability and version consistency."""

import os
import re
from pathlib import Path

import pynvml

# Core libraries to check for findability.
# cudart: universal — everything needs it.
# nvrtc: JIT compilation — cupy, cudf UDFs. Frequently missing with pip (pre-cupy 14).
# nvvm: numba-cuda JIT — cudf string UDFs. Was moved/renamed in CUDA 13.1.
_CUDA_LIBS = {
    "cudart": "libcudart.so",
    "nvrtc": "libnvrtc.so",
    "nvvm": "libnvvm.so",
}

_INSTALL_ADVICE = {
    "conda": (
        "Update your conda environment with the correct CUDA toolkit version, "
        "e.g. 'conda install cuda-toolkit' in your active environment."
    ),
    "pip": (
        "Update the CUDA pip packages in your environment, "
        "e.g. 'pip install --upgrade nvidia-cuda-toolkit'."
    ),
}
_DEFAULT_ADVICE = (
    "Install the CUDA Toolkit matching your driver, "
    "or use conda which manages CUDA automatically."
)

_CUDA_SYMLINK = Path("/usr/local/cuda")

def _get_advice(found_via: str | None) -> str:
    """Return install advice based on how cuda-pathfinder found the library."""
    if found_via:
        for key, advice in _INSTALL_ADVICE.items():
            if key in found_via:
                return advice
    return _DEFAULT_ADVICE


def _get_toolkit_cuda_major() -> int | None:
    """Return the CUDA major version of the toolkit via cuda-pathfinder headers.

    Parses #define CUDA_VERSION from cuda_runtime_version.h.
    Returns None if headers are not available.
    """
    import cuda.pathfinder

    header_dir = cuda.pathfinder.find_nvidia_header_directory("cudart")
    if header_dir is None:
        return None
    version_file = Path(header_dir) / "cuda_runtime_version.h"
    if not version_file.exists():
        return None
    match = re.search(r"#define\s+CUDA_VERSION\s+(\d+)", version_file.read_text())
    return int(match.group(1)) // 1000 if match else None


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
    missing = []
    for libname, soname in _CUDA_LIBS.items():
        try:
            loaded = cuda.pathfinder.load_nvidia_dynamic_lib(libname)
            found_via[libname] = loaded.found_via
        except (DynamicLibNotFoundError, RuntimeError):
            missing.append(soname)

    if missing:
        advice = _get_advice(next(iter(found_via.values()), None))
        raise ValueError(
            f"{', '.join(missing)} could not be found. "
            f"RAPIDS will not be able to run GPU operations. {advice}"
        )

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
    toolkit_major = _get_toolkit_cuda_major()
    if toolkit_major is not None and toolkit_major > driver_major:
        advice = _get_advice(found_via.get("cudart"))
        raise ValueError(
            f"CUDA toolkit is version {toolkit_major} but the GPU driver "
            f"only supports up to CUDA {driver_major}. {advice}"
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
