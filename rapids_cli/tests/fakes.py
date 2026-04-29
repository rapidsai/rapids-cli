# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test fakes for hardware providers."""

from __future__ import annotations

from dataclasses import dataclass, field

from rapids_cli.hardware import DeviceInfo, HardwareInfoError


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
    """Test fake that raises HardwareInfoError on any property access."""

    @property
    def device_count(self) -> int:
        """Raise HardwareInfoError."""
        raise HardwareInfoError("No GPU available")

    @property
    def devices(self) -> list[DeviceInfo]:
        """Raise HardwareInfoError."""
        raise HardwareInfoError("No GPU available")

    @property
    def cuda_driver_version(self) -> int:
        """Raise HardwareInfoError."""
        raise HardwareInfoError("No GPU available")

    @property
    def driver_version(self) -> str:
        """Raise HardwareInfoError."""
        raise HardwareInfoError("No GPU available")


class FailingSystemInfo:
    """Test fake that raises HardwareInfoError on any property access."""

    @property
    def total_memory_bytes(self) -> int:
        """Raise HardwareInfoError."""
        raise HardwareInfoError("System info unavailable")

    @property
    def cuda_runtime_path(self) -> str | None:
        """Raise HardwareInfoError."""
        raise HardwareInfoError("System info unavailable")
