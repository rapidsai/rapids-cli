# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from rapids_cli.doctor.checks.cuda_driver import cuda_check
from rapids_cli.hardware import FailingGpuInfo, FakeGpuInfo


def test_cuda_check_success(set_gpu_info):
    set_gpu_info(FakeGpuInfo(cuda_driver_version=12050))
    assert cuda_check(verbose=True) == 12050


def test_cuda_check_no_gpu(set_gpu_info):
    set_gpu_info(FailingGpuInfo())
    with pytest.raises(ValueError, match="Unable to look up CUDA version"):
        cuda_check(verbose=False)
