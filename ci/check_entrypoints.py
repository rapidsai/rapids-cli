#! /usr/bin/env python

"""Check entry points between pyproject.toml and recipe.yaml."""
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Check entry points between pyproject.toml and recipe.yaml.
#
# Usage:
#   python check_entrypoints.py
#
# This script will add missing entry points to recipe.yaml.

import sys
from pathlib import Path

import tomli
from ruamel.yaml import YAML

PYPROJECT_PATH = Path(__file__).parent.parent / "pyproject.toml"
RECIPE_PATH = Path(__file__).parent.parent / "conda/recipes/rapids-cli/recipe.yaml"

ENTRYPOINTS_SECTION = "project.entry-points.rapids_doctor_check"
ENTRYPOINTS_YAML_PATH = ["build", "python", "entry_points"]


def get_toml_entrypoints(pyproject_path):
    """Get entry points from pyproject.toml."""
    with open(pyproject_path, "rb") as f:
        data = tomli.load(f)
    # Traverse the nested keys for entry points
    section = data
    for key in ENTRYPOINTS_SECTION.split("."):
        section = section.get(key, {})
    return section


def get_yaml_entrypoints(recipe_path):
    """Get entry points from recipe.yaml."""
    yaml = YAML()
    with open(recipe_path) as f:
        data = yaml.load(f)
    # Traverse the nested keys for entry_points
    section = data
    for key in ENTRYPOINTS_YAML_PATH:
        section = section.get(key, {})
    # section should be a list of strings
    return data, section if isinstance(section, list) else []


def sync_entrypoints():
    """Sync entry points between pyproject.toml and recipe.yaml."""
    toml_entrypoints = get_toml_entrypoints(PYPROJECT_PATH)
    yaml_data, yaml_entrypoints = get_yaml_entrypoints(RECIPE_PATH)

    # Convert YAML entrypoints to a set for easy comparison
    yaml_entrypoints_set = set(yaml_entrypoints)
    # TOML entrypoints are a dict: {name: value}
    toml_entrypoints_set = set(f"{k} = {v}" for k, v in toml_entrypoints.items())

    missing = toml_entrypoints_set - yaml_entrypoints_set
    if not missing:
        print("All entry points are in sync.")
        return 0

    # Add missing entrypoints
    print(f"Adding missing entry points to recipe.yaml: {missing}")
    # Traverse to the entry_points list in the YAML data
    section = yaml_data
    for key in ENTRYPOINTS_YAML_PATH[:-1]:
        section = section.setdefault(key, {})
    entry_points_list = section.setdefault(ENTRYPOINTS_YAML_PATH[-1], yaml_entrypoints)
    for entry in missing:
        entry_points_list.append(entry)

    # Write back with ruamel.yaml to preserve comments
    yaml = YAML()
    with open(RECIPE_PATH, "w") as f:
        yaml.dump(yaml_data, f)
    print(f"Updated {RECIPE_PATH} with missing entry points.")
    return 1


if __name__ == "__main__":
    sys.exit(sync_entrypoints())
