# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures for the rapids-cli test suite."""

from __future__ import annotations

import pytest

from rapids_cli import providers


@pytest.fixture(autouse=True)
def _reset_providers(monkeypatch):
    """Ensure each test starts with a clean provider registry.

    Tests that need specific providers installed use the ``set_gpu_info`` /
    ``set_system_info`` / ``set_toolkit_info`` fixtures, which install fakes
    via ``monkeypatch.setattr`` so they auto-revert after the test.
    """
    monkeypatch.setattr(providers._providers, "gpu_info", None)
    monkeypatch.setattr(providers._providers, "system_info", None)
    monkeypatch.setattr(providers._providers, "toolkit_info", None)


@pytest.fixture
def set_gpu_info(monkeypatch):
    """Install a fake GPU info provider for the duration of the test."""

    def _set(fake):
        monkeypatch.setattr(providers._providers, "gpu_info", fake)

    return _set


@pytest.fixture
def set_system_info(monkeypatch):
    """Install a fake system info provider for the duration of the test."""

    def _set(fake):
        monkeypatch.setattr(providers._providers, "system_info", fake)

    return _set


@pytest.fixture
def set_toolkit_info(monkeypatch):
    """Install a fake CUDA toolkit info for the duration of the test."""

    def _set(fake):
        monkeypatch.setattr(providers._providers, "toolkit_info", fake)

    return _set
