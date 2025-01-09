"""Check the system for compatibility with RAPIDS dependencies."""

import json
import platform
import subprocess

from rapids_cli.config import config
from rapids_cli.doctor.checks.cuda_driver import get_cuda_version
from rapids_cli.doctor.checks.os import detect_os


def check_conda(verbose=False):
    """Check the system for Conda compatibility."""
    conda_requirement = config["min_supported_versions"]["conda_requirement"]
    result = subprocess.check_output(
        ["conda", "info", "--json"], stderr=subprocess.DEVNULL
    )
    result_json = json.loads(result.decode("utf-8"))
    version_num = result_json["conda_version"]

    if version_num >= conda_requirement:
        return True
    else:
        raise ValueError(
            "CONDA Version is not compatible with RAPIDS - "
            f"please upgrade to conda {conda_requirement}"
        )


def check_pip(verbose=False):
    """Check the system for Pip compatibility."""
    system_cuda_version = get_cuda_version()
    if not system_cuda_version:
        return
    result = subprocess.check_output(
        ["pip", "show", "cuda-python"], stderr=subprocess.DEVNULL
    )
    pip_cuda_version = result.decode("utf-8").strip().split("\n")[1].split(" ")[-1]
    system_cuda_version_major, pip_cuda_version_major = (
        system_cuda_version.split(".")[0],
        pip_cuda_version.split(".")[0],
    )
    if system_cuda_version_major >= pip_cuda_version_major:
        return True
    else:
        raise ValueError(
            f"Please upgrade pip CUDA version to {system_cuda_version_major}"
        )


def check_docker(verbose=False):
    """Check the system for Docker compatibility."""
    docker_requirement = config["min_supported_versions"]["docker_requirement"]
    docker_version_data = json.loads(
        subprocess.check_output(
            ["docker", "version", "-f", "json"], stderr=subprocess.DEVNULL
        )
    )
    if docker_version_data["Server"]["Version"] >= docker_requirement:
        return True
    else:
        raise ValueError(
            f"Docker Version is not compatible with RAPIDS - "
            f"please upgrade to Docker {docker_requirement}"
        )


def check_glb(verbose=False):
    """Check the system for GLB compatibility."""
    if detect_os() != "Ubuntu":
        return True

    glb_compatible = False
    result = subprocess.check_output(["ldd", "--version"])
    glb_version = result.decode("utf-8").strip().split("\n")[0].split(" ")[-1]

    machine = platform.machine()

    if machine == "x86_64":
        if glb_version >= "2.17":
            glb_compatible = True
    elif machine == "aarch64" or machine == "arm64":
        if glb_version >= "2.32":
            glb_compatible = True
    else:
        raise ValueError("Please only use x86_64 or arm64 architectures")
    if glb_compatible:
        return True
    else:
        if machine == "x86_64":
            raise ValueError("Please upgrade glb to 2.17 and above")
        elif machine == "aarch64" or machine == "arm64":
            raise ValueError("Please upgrade glb to 2.32 and above")
