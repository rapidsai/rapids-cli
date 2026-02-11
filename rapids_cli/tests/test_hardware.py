# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

import pynvml
import pytest

from rapids_cli.hardware import (
    DefaultSystemInfo,
    DeviceInfo,
    FailingGpuInfo,
    FailingSystemInfo,
    FakeGpuInfo,
    FakeSystemInfo,
    GpuInfoProvider,
    NvmlGpuInfo,
    SystemInfoProvider,
)

# --- NvmlGpuInfo tests ---


def test_nvml_gpu_info_init_failure():
    with patch(
        "pynvml.nvmlInit",
        side_effect=pynvml.NVMLError(pynvml.NVML_ERROR_DRIVER_NOT_LOADED),
    ):
        gpu_info = NvmlGpuInfo()
        with pytest.raises(ValueError, match="Unable to initialize GPU driver"):
            _ = gpu_info.device_count


def test_nvml_gpu_info_loads_once():
    mock_handle = MagicMock()
    mock_memory = MagicMock()
    mock_memory.total = 16 * 1024**3

    with (
        patch("pynvml.nvmlInit") as mock_init,
        patch("pynvml.nvmlDeviceGetCount", return_value=1),
        patch("pynvml.nvmlSystemGetCudaDriverVersion", return_value=12050),
        patch("pynvml.nvmlSystemGetDriverVersion", return_value="550.54"),
        patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle),
        patch("pynvml.nvmlDeviceGetCudaComputeCapability", return_value=(7, 5)),
        patch("pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_memory),
        patch(
            "pynvml.nvmlDeviceGetNvLinkState", side_effect=pynvml.NVMLError_NotSupported
        ),
    ):
        gpu_info = NvmlGpuInfo()
        # Access multiple properties to verify caching
        _ = gpu_info.device_count
        _ = gpu_info.devices
        _ = gpu_info.cuda_driver_version
        _ = gpu_info.driver_version
        # nvmlInit should be called exactly once
        mock_init.assert_called_once()


def test_nvml_gpu_info_device_data():
    mock_handle = MagicMock()
    mock_memory = MagicMock()
    mock_memory.total = 24 * 1024**3

    with (
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlDeviceGetCount", return_value=2),
        patch("pynvml.nvmlSystemGetCudaDriverVersion", return_value=12060),
        patch("pynvml.nvmlSystemGetDriverVersion", return_value="560.10"),
        patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle),
        patch("pynvml.nvmlDeviceGetCudaComputeCapability", return_value=(9, 0)),
        patch("pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_memory),
        patch("pynvml.nvmlDeviceGetNvLinkState", return_value=1),
    ):
        gpu_info = NvmlGpuInfo()
        assert gpu_info.device_count == 2
        assert len(gpu_info.devices) == 2
        assert gpu_info.devices[0].compute_capability == (9, 0)
        assert gpu_info.devices[0].memory_total_bytes == 24 * 1024**3
        assert gpu_info.cuda_driver_version == 12060
        assert gpu_info.driver_version == "560.10"


def test_nvml_gpu_info_nvlink_states():
    mock_handle = MagicMock()
    mock_memory = MagicMock()
    mock_memory.total = 16 * 1024**3

    def nvlink_side_effect(handle, link_id):
        if link_id < 2:
            return 1
        raise pynvml.NVMLError_NotSupported()

    with (
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlDeviceGetCount", return_value=1),
        patch("pynvml.nvmlSystemGetCudaDriverVersion", return_value=12050),
        patch("pynvml.nvmlSystemGetDriverVersion", return_value="550.54"),
        patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle),
        patch("pynvml.nvmlDeviceGetCudaComputeCapability", return_value=(7, 5)),
        patch("pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_memory),
        patch("pynvml.nvmlDeviceGetNvLinkState", side_effect=nvlink_side_effect),
    ):
        gpu_info = NvmlGpuInfo()
        assert gpu_info.devices[0].nvlink_states == [True, True]


def test_nvml_gpu_info_no_nvlink():
    mock_handle = MagicMock()
    mock_memory = MagicMock()
    mock_memory.total = 16 * 1024**3

    with (
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlDeviceGetCount", return_value=1),
        patch("pynvml.nvmlSystemGetCudaDriverVersion", return_value=12050),
        patch("pynvml.nvmlSystemGetDriverVersion", return_value="550.54"),
        patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle),
        patch("pynvml.nvmlDeviceGetCudaComputeCapability", return_value=(7, 5)),
        patch("pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_memory),
        patch(
            "pynvml.nvmlDeviceGetNvLinkState", side_effect=pynvml.NVMLError_NotSupported
        ),
    ):
        gpu_info = NvmlGpuInfo()
        assert gpu_info.devices[0].nvlink_states == []


# --- DefaultSystemInfo tests ---


def test_default_system_info_total_memory():
    mock_vm = MagicMock()
    mock_vm.total = 64 * 1024**3
    with patch("psutil.virtual_memory", return_value=mock_vm):
        sys_info = DefaultSystemInfo()
        assert sys_info.total_memory_bytes == 64 * 1024**3


def test_default_system_info_cuda_runtime_path():
    with patch(
        "cuda.pathfinder.find_nvidia_header_directory",
        return_value="/usr/local/cuda/include",
    ):
        sys_info = DefaultSystemInfo()
        assert sys_info.cuda_runtime_path == "/usr/local/cuda/include"


def test_default_system_info_caches():
    mock_vm = MagicMock()
    mock_vm.total = 64 * 1024**3
    with patch("psutil.virtual_memory", return_value=mock_vm) as mock_psutil:
        sys_info = DefaultSystemInfo()
        _ = sys_info.total_memory_bytes
        _ = sys_info.total_memory_bytes
        mock_psutil.assert_called_once()


# --- FakeGpuInfo tests ---


def test_fake_gpu_info_defaults():
    fake = FakeGpuInfo()
    assert fake.device_count == 0
    assert fake.devices == []
    assert fake.cuda_driver_version == 0
    assert fake.driver_version == ""


def test_fake_gpu_info_custom():
    devices = [
        DeviceInfo(index=0, compute_capability=(8, 0), memory_total_bytes=32 * 1024**3)
    ]
    fake = FakeGpuInfo(
        device_count=1,
        devices=devices,
        cuda_driver_version=12040,
        driver_version="550.0",
    )
    assert fake.device_count == 1
    assert len(fake.devices) == 1
    assert fake.cuda_driver_version == 12040


def test_fake_gpu_info_satisfies_protocol():
    assert isinstance(FakeGpuInfo(), GpuInfoProvider)


# --- FakeSystemInfo tests ---


def test_fake_system_info_defaults():
    fake = FakeSystemInfo()
    assert fake.total_memory_bytes == 0
    assert fake.cuda_runtime_path is None


def test_fake_system_info_satisfies_protocol():
    assert isinstance(FakeSystemInfo(), SystemInfoProvider)


# --- FailingGpuInfo tests ---


def test_failing_gpu_info_device_count():
    with pytest.raises(ValueError, match="No GPU available"):
        _ = FailingGpuInfo().device_count


def test_failing_gpu_info_devices():
    with pytest.raises(ValueError, match="No GPU available"):
        _ = FailingGpuInfo().devices


def test_failing_gpu_info_cuda_driver_version():
    with pytest.raises(ValueError, match="No GPU available"):
        _ = FailingGpuInfo().cuda_driver_version


def test_failing_gpu_info_driver_version():
    with pytest.raises(ValueError, match="No GPU available"):
        _ = FailingGpuInfo().driver_version


# --- FailingSystemInfo tests ---


def test_failing_system_info_total_memory():
    with pytest.raises(ValueError, match="System info unavailable"):
        _ = FailingSystemInfo().total_memory_bytes


def test_failing_system_info_cuda_runtime_path():
    with pytest.raises(ValueError, match="System info unavailable"):
        _ = FailingSystemInfo().cuda_runtime_path
