# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the real Conda build entrypoint across Python/Conda version boundaries."""

import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "build_python.sh"
WHEEL_SCRIPT = Path(__file__).parents[1] / "build_wheel.sh"


@pytest.fixture
def build_environment(tmp_path):
    """Isolate runtime commands while retaining the real build entrypoint."""
    tools = tmp_path / "tools"
    tools.mkdir()
    scripts = {
        "rapids-datetime-string": "export RAPIDS_DATETIME_STRING=000000000000\n",
        "rapids-print-env": "true\n",
        "rapids-mamba-retry": "true\n",
        "hatchling": 'printf "called\\n" > "${HATCHLING_CALLED}"\nprintf "26.12\\n"\n',
        "rapids-rattler-channel-string": "RATTLER_CHANNELS=()\nRATTLER_ARGS=()\n",
        "rapids-logger": "true\n",
        "rattler-build": 'printf "%s" "${RAPIDS_PACKAGE_VERSION}" > "${BUILT_VERSION}"\n',
        "rapids-artifact-name": 'printf "rapids-cli-artifact\\n"\n',
        "python": 'printf "%s" "${SETUPTOOLS_SCM_PRETEND_VERSION:-unset}" > "${BUILT_WHEEL_VERSION}"\n',
    }
    for name, contents in scripts.items():
        executable = tools / name
        executable.write_text("#!/bin/bash\nset -euo pipefail\n" + contents)
        executable.chmod(0o700)
    validation = tmp_path / "ci" / "validate_wheel.sh"
    validation.parent.mkdir()
    validation.write_text("#!/bin/bash\ntrue\n")
    validation.chmod(0o700)
    environment = dict(os.environ)
    environment.pop("BUILD_TYPE", None)
    environment.pop("RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS", None)
    environment.update(
        PATH=f"{tools}:{environment['PATH']}",
        HATCHLING_CALLED=str(tmp_path / "hatchling-called"),
        BUILT_VERSION=str(tmp_path / "built-version"),
        RAPIDS_CONDA_BLD_OUTPUT_DIR=str(tmp_path / "build-output"),
        # Reproduce the shared setup's major.minor alias; it is not Conda identity.
        RAPIDS_RELEASE_CANDIDATE_VERSION="26.12",
        SETUPTOOLS_SCM_PRETEND_VERSION="26.12",
        BUILT_WHEEL_VERSION=str(tmp_path / "wheel-version"),
        RAPIDS_WHEEL_BLD_OUTPUT_DIR=str(tmp_path / "wheels"),
        TEST_WORKSPACE=str(tmp_path),
    )
    return environment


def run_build(environment):
    """Source the checked-in build script with test-owned command substitutes."""
    return (
        subprocess.run(  # noqa: S603 - fixed checked-in script, isolated test commands.
            ["bash", str(SCRIPT)],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
    )


def run_wheel(environment):
    """Execute the real wheel builder and capture the version seen by pip."""
    return (
        subprocess.run(  # noqa: S603 - fixed checked-in script, isolated test commands.
            ["bash", str(WHEEL_SCRIPT)],
            cwd=environment["TEST_WORKSPACE"],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
    )


def test_candidate_wheel_keeps_exact_frozen_profile(build_environment):
    """Correct the second V5 catalog mismatch before invoking the backend."""
    build_environment["BUILD_TYPE"] = "release-candidate"
    build_environment["RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS"] = json.dumps(
        {
            "package_versions": {"conda": "26.12.00", "wheel": "26.12.0"},
            "unit_package_versions": {},
        }
    )
    result = run_wheel(build_environment)
    assert result.returncode == 0, result.stderr
    assert Path(build_environment["BUILT_WHEEL_VERSION"]).read_text() == "26.12.0"


def test_candidate_wheel_unit_override_precedes_profile(build_environment):
    """Use the wheel unit's version without borrowing the Conda profile."""
    build_environment["BUILD_TYPE"] = "release-candidate"
    build_environment["RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS"] = json.dumps(
        {
            "package_versions": {"wheel": "26.12.0"},
            "unit_package_versions": {"wheel:rapids-cli": "26.12.7"},
        }
    )
    result = run_wheel(build_environment)
    assert result.returncode == 0, result.stderr
    assert Path(build_environment["BUILT_WHEEL_VERSION"]).read_text() == "26.12.7"


@pytest.mark.parametrize("build_type", [None, "branch", "nightly", "pull-request"])
def test_normal_wheel_does_not_override_metadata(build_environment, build_type):
    """Keep ordinary wheel backend configuration exactly as supplied."""
    if build_type is not None:
        build_environment["BUILD_TYPE"] = build_type
    result = run_wheel(build_environment)
    assert result.returncode == 0, result.stderr
    assert Path(build_environment["BUILT_WHEEL_VERSION"]).read_text() == "26.12"


@pytest.mark.parametrize(
    "versions",
    [
        None,
        "not-json",
        "{}",
        '{"package_versions": {}, "unit_package_versions": {}}',
        '{"package_versions": {"wheel": "26.12.0"}, "unit_package_versions": {"wheel:rapids-cli": false}}',
        '{"package_versions": {"wheel": "26.12.0"}, "unit_package_versions": {"wheel:rapids-cli": ""}}',
    ],
)
def test_candidate_invalid_wheel_identity_fails_before_backend(
    build_environment, versions
):
    """Refuse missing identity instead of producing another unadmittable wheel."""
    build_environment["BUILD_TYPE"] = "release-candidate"
    if versions is not None:
        build_environment["RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS"] = versions
    result = run_wheel(build_environment)
    assert result.returncode != 0
    assert not Path(build_environment["BUILT_WHEEL_VERSION"]).exists()


def test_candidate_conda_keeps_exact_frozen_version_despite_python_alias(
    build_environment,
):
    """Reproduce V5's normalized metadata without changing Conda identity."""
    build_environment["BUILD_TYPE"] = "release-candidate"
    build_environment["RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS"] = json.dumps(
        {
            "package_versions": {"conda": "26.12.00", "wheel": "26.12.0"},
            "unit_package_versions": {},
        }
    )
    result = run_build(build_environment)
    assert result.returncode == 0, result.stderr
    assert Path(build_environment["BUILT_VERSION"]).read_text() == "26.12.00"
    assert not Path(build_environment["HATCHLING_CALLED"]).exists()


def test_candidate_unit_override_precedes_family_version(build_environment):
    """Honor explicit unit versions before the family's default version."""
    build_environment["BUILD_TYPE"] = "release-candidate"
    build_environment["RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS"] = json.dumps(
        {
            "package_versions": {"conda": "26.12.00"},
            "unit_package_versions": {"conda:rapids-cli": "26.12.07"},
        }
    )
    result = run_build(build_environment)
    assert result.returncode == 0, result.stderr
    assert Path(build_environment["BUILT_VERSION"]).read_text() == "26.12.07"


@pytest.mark.parametrize("build_type", [None, "branch", "nightly", "pull-request"])
def test_normal_build_keeps_hatchling_behavior(build_environment, build_type):
    """Leave ordinary source-derived package versions unchanged."""
    if build_type is not None:
        build_environment["BUILD_TYPE"] = build_type
    result = run_build(build_environment)
    assert result.returncode == 0, result.stderr
    assert Path(build_environment["BUILT_VERSION"]).read_text() == "26.12"
    assert Path(build_environment["HATCHLING_CALLED"]).exists()


@pytest.mark.parametrize(
    "versions",
    [
        None,
        "not-json",
        "{}",
        '{"package_versions": {}, "unit_package_versions": {}}',
        '{"package_versions": {"conda": "26.12.00"}, "unit_package_versions": {"conda:rapids-cli": false}}',
        '{"package_versions": {"conda": "26.12.00"}, "unit_package_versions": {"conda:rapids-cli": ""}}',
    ],
)
def test_candidate_missing_or_invalid_identity_fails_before_build(
    build_environment, versions
):
    """Never substitute guessed Python metadata for missing train identity."""
    build_environment["BUILD_TYPE"] = "release-candidate"
    if versions is not None:
        build_environment["RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS"] = versions
    result = run_build(build_environment)
    assert result.returncode != 0
    assert not Path(build_environment["BUILT_VERSION"]).exists()
    assert not Path(build_environment["HATCHLING_CALLED"]).exists()
