# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pynvml
import pytest

from rapids_cli.doctor.checks.cuda_driver import cuda_check


def test_cuda_check_success():
    with (
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlSystemGetCudaDriverVersion", return_value=12050),
    ):
        assert cuda_check(verbose=True) == 12050


def test_cuda_check_init_fails():
    with patch("pynvml.nvmlInit", side_effect=pynvml.NVMLError(1)):
        with pytest.raises(ValueError, match="Unable to look up CUDA version"):
            cuda_check()


def test_cuda_check_version_query_fails():
    with (
        patch("pynvml.nvmlInit"),
        patch(
            "pynvml.nvmlSystemGetCudaDriverVersion",
            side_effect=pynvml.NVMLError(1),
        ),
    ):
        with pytest.raises(ValueError, match="Unable to look up CUDA version"):
            cuda_check()
