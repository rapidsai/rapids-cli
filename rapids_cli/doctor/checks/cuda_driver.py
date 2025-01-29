# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check for CUDA and driver compatibility."""

import platform
import subprocess

import pynvml


def cuda_check(verbose=False):
    """Check CUDA availability."""
    try:
        pynvml.nvmlInit()
        try:
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
            return cuda_version
        except pynvml.NVMLError as e:
            raise ValueError("Unable to look up CUDA version") from e
    except pynvml.NVMLError as e:
        raise ValueError("Unable to look up CUDA version") from e


# CUDA Version : NVIDIA DRIVER Version
SUPPORTED_VERSIONS = {
    "11.2": "470.42.01",
    "11.4": "470.42.01",
    "11.5": "495.29.05",
    "11.8": "520.61.05",
    "12.0": "525.60.13",
    "12.1": "530.30.02",
    "12.2": "535.86.10",
}


def get_cuda_version(verbose=False):
    """Get the CUDA version."""
    try:
        output = subprocess.check_output(["nvcc", "--version"])
        version_line = output.decode("utf-8").strip().split("\n")[-1]
        return version_line.split()[-1].split("/")[0][-4:]  # Extract the version number
    except Exception:
        return None


def get_driver_version(verbose=False):
    """Get the driver version."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )

        result_chain = result.stdout.strip()
        return result_chain.split("\n")[0]
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError:
        return None


# https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
def check_driver_compatibility(verbose=False):
    """Check if the driver is compatible with the installed CUDA toolkit."""
    platform.system()
    driver_compatible = True
    cuda_version = get_cuda_version()
    driver_version = get_driver_version()

    if not driver_version or not cuda_version:
        driver_compatible = False
    elif cuda_version >= "12.3":
        driver_compatible = True
    elif cuda_version < "11.2":
        driver_compatible = False
    else:
        if driver_version < SUPPORTED_VERSIONS[cuda_version]:
            driver_compatible = False

    if driver_compatible:
        return True
    else:
        raise ValueError(
            "CUDA & Driver is not compatible with RAPIDS. "
            "Please see https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html "
            "for CUDA compatability guidance."
        )
