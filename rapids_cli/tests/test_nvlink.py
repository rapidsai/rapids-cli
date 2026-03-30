# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

import pytest

from rapids_cli.doctor.checks.nvlink import check_nvlink_status


@pytest.mark.parametrize(
    "verbose, expected",
    [
        (True, "All NVLinks active across 2 GPUs"),
        (False, None),
    ],
)
def test_check_nvlink_status_success(verbose, expected):
    """2 GPUs, all NVLinks active — verbose controls whether a summary string is returned."""
    import pynvml

    mock_handle = MagicMock()
    with (
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlDeviceGetCount", return_value=2),
        patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle),
        patch(
            "pynvml.nvmlDeviceGetNvLinkState", return_value=pynvml.NVML_FEATURE_ENABLED
        ),
    ):
        result = check_nvlink_status(verbose=verbose)
        assert result == expected


def test_check_nvlink_status_single_gpu():
    """Single GPU — NVLink is not applicable, check skips early."""
    with (
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlDeviceGetCount", return_value=1),
    ):
        result = check_nvlink_status(verbose=False)
        assert result is False


def test_check_nvlink_status_no_gpu():
    """nvmlInit fails — no GPUs installed."""
    import pynvml

    with patch("pynvml.nvmlInit", side_effect=pynvml.NVMLError(1)):
        with pytest.raises(
            ValueError, match="GPU not found. Please ensure GPUs are installed."
        ):
            check_nvlink_status(verbose=False)


def test_check_nvlink_status_not_supported():
    """NVLink is not supported on this system — check skips silently like single-GPU case."""
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
        result = check_nvlink_status(verbose=False)
        assert result is False


def test_check_nvlink_status_link_inactive():
    """A supported link is inactive — check fails and reports which GPU and link."""
    import pynvml

    mock_handle = MagicMock()
    with (
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlDeviceGetCount", return_value=2),
        patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle),
        patch(
            "pynvml.nvmlDeviceGetNvLinkState", return_value=pynvml.NVML_FEATURE_DISABLED
        ),
    ):
        with pytest.raises(ValueError, match="NVLink inactive on:"):
            check_nvlink_status(verbose=False)


def test_check_nvlink_status_partial_failure():
    """Some links active, some inactive — all failures are reported in a single error."""
    import pynvml

    mock_handle = MagicMock()

    # Simulate: link 0 active, link 1 inactive, rest active
    def mock_link_state(handle, link_id):
        if link_id == 1:
            return pynvml.NVML_FEATURE_DISABLED
        return pynvml.NVML_FEATURE_ENABLED

    with (
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlDeviceGetCount", return_value=2),
        patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle),
        patch("pynvml.nvmlDeviceGetNvLinkState", side_effect=mock_link_state),
    ):
        with pytest.raises(ValueError, match="NVLink inactive on:") as exc_info:
            check_nvlink_status(verbose=False)
        # Both GPUs should have link 1 reported as failed
        assert "GPU 0 link 1" in str(exc_info.value)
        assert "GPU 1 link 1" in str(exc_info.value)
