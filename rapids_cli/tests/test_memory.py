# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

import pytest

from rapids_cli.doctor.checks.memory import (
    check_memory_to_gpu_ratio,
    get_gpu_memory,
    get_system_memory,
)


def test_get_system_memory():
    mock_vm = MagicMock()
    mock_vm.total = 32 * 1024**3  # 32 GB in bytes
    with patch("psutil.virtual_memory", return_value=mock_vm):
        result = get_system_memory(verbose=False)
        assert result == 32.0


def test_get_gpu_memory_single_gpu():
    mock_memory_info = MagicMock()
    mock_memory_info.total = 16 * 1024**3  # 16 GB in bytes

    with (
        patch("cuda.bindings.nvml.init_v2"),
        patch("cuda.bindings.nvml.device_get_count_v2", return_value=1),
        patch("cuda.bindings.nvml.device_get_handle_by_index_v2", return_value=0xffffffff),
        patch("cuda.bindings.nvml.device_get_memory_info_v2", return_value=mock_memory_info),
        patch("cuda.bindings.nvml.shutdown"),
    ):
        result = get_gpu_memory(verbose=False)
        assert result == 16.0


def test_get_gpu_memory_multiple_gpus():
    mock_memory_info = MagicMock()
    mock_memory_info.total = 16 * 1024**3  # 16 GB per GPU

    with (
        patch("cuda.bindings.nvml.init_v2"),
        patch("cuda.bindings.nvml.device_get_count_v2", return_value=4),
        patch("cuda.bindings.nvml.device_get_handle_by_index_v2", return_value=0xffffffff),
        patch("cuda.bindings.nvml.device_get_memory_info_v2", return_value=mock_memory_info),
        patch("cuda.bindings.nvml.shutdown"),
    ):
        result = get_gpu_memory(verbose=False)
        assert result == 64.0  # 16 GB * 4 GPUs


def test_check_memory_to_gpu_ratio_good_ratio():
    with (
        patch("cuda.bindings.nvml.init_v2"),
        patch("cuda.bindings.nvml.device_get_count_v2", return_value=2),
        patch("rapids_cli.doctor.checks.memory.get_system_memory", return_value=64.0),
        patch("rapids_cli.doctor.checks.memory.get_gpu_memory", return_value=32.0),
    ):
        result = check_memory_to_gpu_ratio(verbose=True)
        assert result is True


def test_check_memory_to_gpu_ratio_warning():
    with (
        patch("cuda.bindings.nvml.init_v2"),
        patch("cuda.bindings.nvml.device_get_count_v2", return_value=2),
        patch("rapids_cli.doctor.checks.memory.get_system_memory", return_value=32.0),
        patch("rapids_cli.doctor.checks.memory.get_gpu_memory", return_value=32.0),
    ):
        with pytest.warns(UserWarning, match="System Memory to total GPU Memory ratio"):
            result = check_memory_to_gpu_ratio(verbose=True)
            assert result is True


def test_check_memory_to_gpu_ratio_no_gpu():
    from cuda.bindings import nvml

    with (
        patch("cuda.bindings.nvml.init_v2"),
        patch("cuda.bindings.nvml.device_get_count_v2", return_value=0),
    ):
        with pytest.raises(
            ValueError, match="GPU not found. Please ensure GPUs are installed."
        ):
            check_memory_to_gpu_ratio(verbose=False)
