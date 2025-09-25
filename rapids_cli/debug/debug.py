# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This module contains the debug subcommand for the Rapids CLI."""
import json
import shutil
import subprocess
import sys
from importlib.metadata import distributions, version

import pynvml
from rich.console import Console
from rich.table import Table

console = Console()

pynvml.nvmlInit()


def gather_nvidia_smi_output():
    """Gather NVIDIA-SMI output."""
    try:
        return subprocess.run(
            ["nvsdazsdidia-smi"], capture_output=True, text=True
        ).stdout
    except FileNotFoundError:
        return "Nvidia-smi not installed"


def gather_cuda_version():
    """Return CUDA driver version as a string, similar to nvidia-smi output."""
    version = pynvml.nvmlSystemGetCudaDriverVersion()
    # pynvml returns an int like 12040 for 12.4, so format as string
    major = version // 1000
    minor = (version % 1000) // 10
    patch = version % 10
    if patch == 0:
        return f"{major}.{minor}"
    else:
        return f"{major}.{minor}.{patch}"


def gather_driver_version():
    """Return CUDA driver version as a string, similar to nvidia-smi output."""
    return pynvml.nvmlSystemGetDriverVersion()


def gather_pip_packages():
    """Return pip packages."""
    try:
        return subprocess.check_output(["pip", "freeze"], text=True)
    except FileNotFoundError:
        return "Pip not installed"


def gather_python_version_full():
    """Return full Python version."""
    return sys.version


def gather_python_version():
    """Return Python version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def gather_package_versions():
    """Return package version."""
    installed_packages = sorted(
        distributions(), key=lambda pkg: pkg.metadata["Name"].lower()
    )
    package_versions = {}
    for package in installed_packages:
        package_name = package.metadata["Name"]
        package_version = version(package_name)
        package_versions[package_name] = package_version
    return package_versions


def gather_conda_packages():
    """Return conda packages."""
    try:
        return subprocess.check_output(["conda", "list"], text=True)
    except FileNotFoundError:
        return "Conda not installed"


def gather_package_managers():
    """Return package managers."""
    return {
        "pip": shutil.which("pip") is not None,
        "conda": shutil.which("conda") is not None,
        "uv": shutil.which("uv") is not None,
        "pixi": shutil.which("pixi") is not None,
    }


def run_debug(output_format="console"):
    """Run debug."""
    debug_info = {
        "nvidia_smi_output": gather_nvidia_smi_output(),
        "driver_version": gather_driver_version(),
        "cuda_version": gather_cuda_version(),
        "python_version_full": gather_python_version_full(),
        "python_version": gather_python_version(),
        "package_versions": gather_package_versions(),
        "pip_packages": gather_pip_packages(),
        "conda_packages": gather_conda_packages(),
        "package_managers": gather_package_managers(),
    }

    if output_format == "json":
        console.print(json.dumps(debug_info, indent=4))
    else:
        console.print("[bold purple]RAPIDS Debug Information[/bold purple]")
        for key, value in debug_info.items():
            console.print(f"[bold green]{key.replace('_', ' ').title()}[/bold green]")
            if isinstance(value, str):
                console.print(value)
            elif isinstance(value, dict):
                table = Table(show_header=False, header_style="bold magenta")
                for k, v in value.items():
                    table.add_row(str(k), str(v))
                console.print(table)
            else:
                console.print(value)
            console.print()
