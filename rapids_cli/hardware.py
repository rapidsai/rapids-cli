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
    """Real GPU info provider backed by pynvml.

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

        import pynvml

        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as e:
            raise ValueError("Unable to initialize GPU driver (NVML)") from e

        self._device_count = pynvml.nvmlDeviceGetCount()
        self._cuda_driver_version = pynvml.nvmlSystemGetCudaDriverVersion()
        self._driver_version = pynvml.nvmlSystemGetDriverVersion()

        self._devices = []
        for i in range(self._device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            nvlink_states: list[bool] = []
            for link_id in range(pynvml.NVML_NVLINK_MAX_LINKS):
                try:
                    state = pynvml.nvmlDeviceGetNvLinkState(handle, link_id)
                    nvlink_states.append(bool(state))
                except pynvml.NVMLError:
                    break

            self._devices.append(
                DeviceInfo(
                    index=i,
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


@dataclass
class FakeGpuInfo:
    """Test fake for GPU information with pre-set data."""

    device_count: int = 0
    devices: list[DeviceInfo] = field(default_factory=list)
    cuda_driver_version: int = 0
    driver_version: str = ""


@dataclass
class FakeSystemInfo:
    """Test fake for system information with pre-set data."""

    total_memory_bytes: int = 0
    cuda_runtime_path: str | None = None


class FailingGpuInfo:
    """Test fake that raises ValueError on any property access."""

    @property
    def device_count(self) -> int:
        """Raise ValueError."""
        raise ValueError("No GPU available")

    @property
    def devices(self) -> list[DeviceInfo]:
        """Raise ValueError."""
        raise ValueError("No GPU available")

    @property
    def cuda_driver_version(self) -> int:
        """Raise ValueError."""
        raise ValueError("No GPU available")

    @property
    def driver_version(self) -> str:
        """Raise ValueError."""
        raise ValueError("No GPU available")


class FailingSystemInfo:
    """Test fake that raises ValueError on any property access."""

    @property
    def total_memory_bytes(self) -> int:
        """Raise ValueError."""
        raise ValueError("System info unavailable")

    @property
    def cuda_runtime_path(self) -> str | None:
        """Raise ValueError."""
        raise ValueError("System info unavailable")
