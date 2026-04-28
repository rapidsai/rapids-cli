# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest

from rapids_cli.doctor.checks.cuda_driver import cuda_check


def test_cuda_check_success():
    with (
        patch("cuda.core.system.get_driver_version_full", return_value=(12, 5, 0)),
    ):
        assert cuda_check(verbose=True) == 12050


def test_cuda_check_init_fails():
    from cuda.bindings import nvml  

    with patch("cuda.bindings.nvml.init_v2", side_effect=nvml.NvmlError(1)):
        with pytest.raises(ValueError, match="Unable to look up CUDA version"):
            cuda_check()


def test_cuda_check_version_query_fails():
    from cuda.bindings import nvml

    with (
        patch("cuda.bindings.nvml.init_v2"),
        patch(
            "cuda.bindings.nvml.system_get_cuda_driver_version",
            side_effect=nvml.NvmlError(1),
        ),
    ):
        with pytest.raises(ValueError, match="Unable to look up CUDA version"):
            cuda_check()
