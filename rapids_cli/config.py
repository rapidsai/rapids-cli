# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config loader for the CLI."""

import pathlib

import yaml

_ROOT_DIR = pathlib.Path(__file__).parent


with open(_ROOT_DIR / "config.yml", "r") as file:
    config = yaml.safe_load(file)
