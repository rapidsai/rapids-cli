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
    from cuda.bindings import nvml

    # Simulate a V100 with 6 NVLink slots; link_id >= 6 is out of range.
    def mock_link_state(handle, link_id):
        if link_id >= 6:
            raise nvml.InvalidArgumentError(0)
        return nvml.EnableState.FEATURE_ENABLED

    with (
        patch("cuda.bindings.nvml.init_v2"),
        patch("cuda.bindings.nvml.device_get_count_v2", return_value=2),
        patch("cuda.bindings.nvml.device_get_handle_by_index_v2", return_value=0xffffffff),
        patch("cuda.bindings.nvml.device_get_nvlink_state", side_effect=mock_link_state),
    ):
        result = check_nvlink_status(verbose=verbose)
        assert result == expected


def test_check_nvlink_status_single_gpu():
    """Single GPU — NVLink is not applicable, check skips early."""
    with (
        patch("cuda.bindings.nvml.init_v2"),
        patch("cuda.bindings.nvml.device_get_count_v2", return_value=1),
    ):
        result = check_nvlink_status(verbose=False)
        assert result is False


def test_check_nvlink_status_no_gpu():
    """nvmlInit fails — no GPUs installed."""
    from cuda.bindings import nvml

    with patch("cuda.bindings.nvml.init_v2", side_effect=nvml.NvmlError(1)):
        with pytest.raises(
            ValueError, match="GPU not found. Please ensure GPUs are installed."
        ):
            check_nvlink_status(verbose=False)


def test_check_nvlink_status_not_supported():
    """NVLink is not supported on this system — check skips silently like single-GPU case."""
    from cuda.bindings import nvml

    with (
        patch("cuda.bindings.nvml.init_v2"),
        patch("cuda.bindings.nvml.device_get_count_v2", return_value=2),
        patch("cuda.bindings.nvml.device_get_handle_by_index_v2", return_value=0xffffffff),
        patch(
            "cuda.bindings.nvml.device_get_nvlink_state", side_effect=nvml.NotSupportedError(1)
        ),
    ):
        result = check_nvlink_status(verbose=False)
        assert result is False


def test_check_nvlink_status_link_inactive():
    """A supported link is inactive — check fails and reports which GPU and link."""
    from cuda.bindings import nvml

    # Simulate a V100 with 6 NVLink slots, all inactive.
    def mock_link_state(handle, link_id):
        if link_id >= 6:
            raise nvml.InvalidArgumentError(0)
        return nvml.EnableState.FEATURE_DISABLED

    with (
        patch("cuda.bindings.nvml.init_v2"),
        patch("cuda.bindings.nvml.device_get_count_v2", return_value=2),
        patch("cuda.bindings.nvml.device_get_handle_by_index_v2", return_value=0xffffffff),
        patch("cuda.bindings.nvml.device_get_nvlink_state", side_effect=mock_link_state),
    ):
        with pytest.raises(ValueError, match="NVLink inactive on:"):
            check_nvlink_status(verbose=False)


def test_check_nvlink_status_partial_failure():
    """Some links active, some inactive — all failures are reported in a single error."""
    from cuda.bindings import nvml

    # Simulate a V100 with 6 NVLink slots: link 0 active, link 1 inactive, rest active.
    def mock_link_state(handle, link_id):
        if link_id >= 6:
            raise nvml.InvalidArgumentError(0)
        if link_id == 1:
            return nvml.EnableState.FEATURE_DISABLED
        return nvml.EnableState.FEATURE_ENABLED

    with (
        patch("cuda.bindings.nvml.init_v2"),
        patch("cuda.bindings.nvml.device_get_count_v2", return_value=2),
        patch("cuda.bindings.nvml.device_get_handle_by_index_v2", return_value=0xffffffff),
        patch("cuda.bindings.nvml.device_get_nvlink_state", side_effect=mock_link_state),
    ):
        with pytest.raises(ValueError, match="NVLink inactive on:") as exc_info:
            check_nvlink_status(verbose=False)
        # Both GPUs should have link 1 reported as failed
        assert "GPU 0 link 1" in str(exc_info.value)
        assert "GPU 1 link 1" in str(exc_info.value)


def test_check_nvlink_status_invalid_argument():
    """NVMLError_InvalidArgument stops link iteration early — check succeeds for valid links."""
    from cuda.bindings import nvml

    # Simulate an A100 with 12 NVLink slots; link_id >= 12 is out of range.
    def mock_link_state(handle, link_id):
        if link_id >= 12:
            raise nvml.InvalidArgumentError(0)
        return nvml.EnableState.FEATURE_ENABLED

    with (
        patch("cuda.bindings.nvml.init_v2"),
        patch("cuda.bindings.nvml.device_get_count_v2", return_value=2),
        patch("cuda.bindings.nvml.device_get_handle_by_index_v2", return_value=0xffffffff),
        patch("cuda.bindings.nvml.device_get_nvlink_state", side_effect=mock_link_state),
    ):
        result = check_nvlink_status(verbose=True)
        assert result == "All NVLinks active across 2 GPUs"
