# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This module contains the debug subcommand for the Rapids CLI."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from importlib.metadata import distributions, version
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from rapids_cli.hardware import GpuInfoProvider, SystemInfoProvider

console = Console()


def gather_cuda_version(*, gpu_info: GpuInfoProvider | None = None):
    """Return CUDA driver version as a string, similar to nvidia-smi output."""
    if gpu_info is None:  # pragma: no cover
        from rapids_cli.hardware import NvmlGpuInfo

        gpu_info = NvmlGpuInfo()

    ver = gpu_info.cuda_driver_version
    # pynvml returns an int like 12040 for 12.4, so format as string
    major = ver // 1000
    minor = (ver % 1000) // 10
    patch = ver % 10
    if patch == 0:
        return f"{major}.{minor}"
    else:
        return f"{major}.{minor}.{patch}"


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


def gather_command_output(
    command: list[str], fallback_output: str | None = None
) -> str | None:
    """Return command output."""
    try:
        return subprocess.check_output(command, text=True).strip()
    except FileNotFoundError:
        return fallback_output


def gather_tools():
    """Return tools."""
    return {
        "pip": gather_command_output(["pip", "--version"]),
        "conda": gather_command_output(["conda", "--version"]),
        "uv": gather_command_output(["uv", "--version"]),
        "pixi": gather_command_output(["pixi", "--version"]),
        "g++": gather_command_output(["g++", "--version"]),
        "cmake": gather_command_output(["cmake", "--version"]),
        "nvcc": gather_command_output(["nvcc", "--version"]),
    }


def run_debug(
    output_format="console",
    *,
    gpu_info: GpuInfoProvider | None = None,
    system_info: SystemInfoProvider | None = None,
):
    """Run debug."""
    if gpu_info is None:  # pragma: no cover
        from rapids_cli.hardware import NvmlGpuInfo

        gpu_info = NvmlGpuInfo()
    if system_info is None:  # pragma: no cover
        from rapids_cli.hardware import DefaultSystemInfo

        system_info = DefaultSystemInfo()

    debug_info = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "nvidia_smi_output": gather_command_output(
            ["nvidia-smi"], "Nvidia-smi not installed"
        ),
        "driver_version": gpu_info.driver_version,
        "cuda_version": gather_cuda_version(gpu_info=gpu_info),
        "cuda_runtime_path": system_info.cuda_runtime_path,
        "system_ctk": sorted(
            [str(p) for p in Path("/usr/local").glob("cuda*") if p.is_dir()]
        ),
        "python_version_full": sys.version,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        # cast sys.hash_info to str as repr is most useful https://github.com/rapidsai/rapids-cli/pull/127#discussion_r2397926022
        "python_hash_info": str(sys.hash_info),
        "package_versions": gather_package_versions(),
        "pip_packages": gather_command_output(["pip", "freeze"], "Pip not installed"),
        "conda_packages": gather_command_output(
            ["conda", "list"], "Conda not installed"
        ),
        "conda_info": gather_command_output(["conda", "info"], "Conda not installed"),
        "tools": gather_tools(),
        "os_info": {
            v.split("=")[0]: v.split("=")[1].strip('"')
            for v in Path("/etc/os-release").read_text().splitlines()
        },
    }

    if output_format == "json":
        print(json.dumps(debug_info, indent=4))
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
