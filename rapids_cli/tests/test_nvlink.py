# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

import pytest

from rapids_cli.doctor.checks.nvlink import check_nvlink_status


def test_check_nvlink_status_success():
    mock_handle = MagicMock()
    with (
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlDeviceGetCount", return_value=2),
        patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle),
        patch("pynvml.nvmlDeviceGetNvLinkState", return_value=1),
    ):
        result = check_nvlink_status(verbose=True)
        assert result is True


def test_check_nvlink_status_single_gpu():
    with (
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlDeviceGetCount", return_value=1),
    ):
        result = check_nvlink_status(verbose=False)
        assert result is False


def test_check_nvlink_status_no_gpu():
    import pynvml

    with patch("pynvml.nvmlInit", side_effect=pynvml.NVMLError(1)):
        with pytest.raises(
            ValueError, match="GPU not found. Please ensure GPUs are installed."
        ):
            check_nvlink_status(verbose=False)


def test_check_nvlink_status_nvml_error():
    import pynvml

    mock_handle = MagicMock()
    with (
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlDeviceGetCount", return_value=2),
        patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle),
        patch(
            "pynvml.nvmlDeviceGetNvLinkState", side_effect=pynvml.NVMLError_NotSupported
        ),
    ):
        with pytest.raises(ValueError, match="NVLink 0 Status Check Failed"):
            check_nvlink_status(verbose=False)
