# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware abstraction layer for GPU and system information."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class DeviceInfo:
    """Per-GPU device information."""

    index: int
    compute_capability: tuple[int, int]
    memory_total_bytes: int
    nvlink_states: list[bool] = field(default_factory=list)


class HardwareInfoError(Exception):
    """Raised when hardware information cannot be obtained."""


@runtime_checkable
class GpuInfoProvider(Protocol):
    """Read-only interface for GPU information."""

    @property
    def device_count(self) -> int:
        """Return number of GPU devices."""
        ...

    @property
    def devices(self) -> list[DeviceInfo]:
        """Return list of device information."""
        ...

    @property
    def cuda_driver_version(self) -> int:
        """Return CUDA driver version as integer."""
        ...

    @property
    def driver_version(self) -> str:
        """Return driver version string."""
        ...


@runtime_checkable
class SystemInfoProvider(Protocol):
    """Read-only interface for system information."""

    @property
    def total_memory_bytes(self) -> int:
        """Return total system memory in bytes."""
        ...

    @property
    def cuda_runtime_path(self) -> str | None:
        """Return path to CUDA runtime headers."""
        ...


class NvmlGpuInfo:
    """Real GPU info provider backed by cuda.core.system.

    Lazily loads all device information on first property access and caches results.
    """

    def __init__(self) -> None:
        """Initialize with empty cached state."""
        self._loaded = False
        self._device_count = 0
        self._devices: list[DeviceInfo] = []
        self._cuda_driver_version = 0
        self._driver_version = ""

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        from cuda.core import system

        try:
            self._device_count = system.get_num_devices()
        except system.NvmlError as e:
            raise HardwareInfoError("Unable to initialize GPU driver (NVML)") from e

        cuda_driver_version = system.get_user_mode_driver_version()
        self._cuda_driver_version = cuda_driver_version[0] * 1000 + cuda_driver_version[1] * 10
        driver_version = system.get_kernel_mode_driver_version()
        self._driver_version = ".".join(str(x) for x in driver_version[:2])

        self._devices = []
        for device in system.Device.get_all_devices():
            major, minor = device.cuda_compute_capability
            memory_info = device.memory_info

            nvlink_states: list[bool] = []
            for link_id in range(system.NvlinkInfo.max_links):
                try:
                    state = device.get_nvlink(link_id).state
                    nvlink_states.append(bool(state))
                except (
                    system.InvalidArgumentError,
                    system.NotSupportedError,
                ):
                    break

            self._devices.append(
                DeviceInfo(
                    index=device.index,
                    compute_capability=(major, minor),
                    memory_total_bytes=memory_info.total,
                    nvlink_states=nvlink_states,
                )
            )

        self._loaded = True

    @property
    def device_count(self) -> int:
        """Return number of GPU devices."""
        self._ensure_loaded()
        return self._device_count

    @property
    def devices(self) -> list[DeviceInfo]:
        """Return list of device information."""
        self._ensure_loaded()
        return self._devices

    @property
    def cuda_driver_version(self) -> int:
        """Return CUDA driver version as integer (e.g. 12040)."""
        self._ensure_loaded()
        return self._cuda_driver_version

    @property
    def driver_version(self) -> str:
        """Return driver version string."""
        self._ensure_loaded()
        return self._driver_version


class DefaultSystemInfo:
    """Real system info provider backed by psutil and cuda.pathfinder.

    Lazily loads each piece of information on first access.
    """

    def __init__(self) -> None:
        """Initialize with empty cached state."""
        self._memory_loaded = False
        self._total_memory_bytes = 0
        self._cuda_path_loaded = False
        self._cuda_runtime_path: str | None = None

    @property
    def total_memory_bytes(self) -> int:
        """Return total system memory in bytes."""
        if not self._memory_loaded:
            import psutil

            self._total_memory_bytes = psutil.virtual_memory().total
            self._memory_loaded = True
        return self._total_memory_bytes

    @property
    def cuda_runtime_path(self) -> str | None:
        """Return path to CUDA runtime headers."""
        if not self._cuda_path_loaded:
            import cuda.pathfinder

            self._cuda_runtime_path = cuda.pathfinder.find_nvidia_header_directory(
                "cudart"
            )
            self._cuda_path_loaded = True
        return self._cuda_runtime_path
