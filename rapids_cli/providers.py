# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-wide hardware provider registry.

The doctor orchestrator installs real providers once per run via
``set_providers``; check and debug functions read them via the ``get_*``
accessors. Tests swap in fakes with ``monkeypatch.setattr`` on the
``_providers`` dataclass instance (or via the fixtures in
``rapids_cli/tests/conftest.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rapids_cli.doctor.checks.cuda_toolkit import CudaToolkitInfo
    from rapids_cli.hardware import GpuInfoProvider, SystemInfoProvider


@dataclass
class _Providers:
    """Container for the process-wide hardware providers."""

    gpu_info: GpuInfoProvider | None = field(default=None)
    system_info: SystemInfoProvider | None = field(default=None)
    toolkit_info: CudaToolkitInfo | None = field(default=None)


_providers = _Providers()


def set_providers(
    *,
    gpu_info: GpuInfoProvider | None = None,
    system_info: SystemInfoProvider | None = None,
    toolkit_info: CudaToolkitInfo | None = None,
) -> None:
    """Install providers for the current run. Only non-None args are applied."""
    if gpu_info is not None:
        _providers.gpu_info = gpu_info
    if system_info is not None:
        _providers.system_info = system_info
    if toolkit_info is not None:
        _providers.toolkit_info = toolkit_info


def get_gpu_info() -> GpuInfoProvider:
    """Return the installed GPU info provider, lazily creating a real one."""
    if _providers.gpu_info is None:  # pragma: no cover
        from rapids_cli.hardware import NvmlGpuInfo

        _providers.gpu_info = NvmlGpuInfo()
    return _providers.gpu_info


def get_system_info() -> SystemInfoProvider:
    """Return the installed system info provider, lazily creating a real one."""
    if _providers.system_info is None:  # pragma: no cover
        from rapids_cli.hardware import DefaultSystemInfo

        _providers.system_info = DefaultSystemInfo()
    return _providers.system_info


def get_toolkit_info() -> CudaToolkitInfo:
    """Return the installed toolkit info, lazily gathering it from the system."""
    if _providers.toolkit_info is None:  # pragma: no cover
        from rapids_cli.doctor.checks.cuda_toolkit import _gather_toolkit_info

        _providers.toolkit_info = _gather_toolkit_info()
    return _providers.toolkit_info
