# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from rapids_cli.config import config


def test_config_loaded():
    """Test that config is loaded successfully."""
    assert config is not None
    assert isinstance(config, dict)


def test_config_has_min_supported_versions():
    """Test that config contains minimum supported versions."""
    assert "min_supported_versions" in config
    assert "gpu_compute_requirement" in config["min_supported_versions"]


def test_config_has_valid_subcommands():
    """Test that config contains valid subcommands."""
    assert "valid_subcommands" in config
    assert "VALID_SUBCOMMANDS" in config["valid_subcommands"]


def test_config_has_os_requirements():
    """Test that config contains OS requirements."""
    assert "os_requirements" in config
    assert "VALID_LINUX_OS_VERSIONS" in config["os_requirements"]
    assert "OS_TO_MIN_SUPPORTED_VERSION" in config["os_requirements"]


def test_config_has_cudf_section():
    """Test that config contains cuDF section."""
    assert "cudf" in config
    assert "cuda_requirement" in config["cudf"]
    assert "driver_requirement" in config["cudf"]
    assert "compute_requirement" in config["cudf"]
    assert "links" in config["cudf"]
    assert "description" in config["cudf"]


def test_config_has_cuml_section():
    """Test that config contains cuML section."""
    assert "cuml" in config
    assert "links" in config["cuml"]
    assert "description" in config["cuml"]
