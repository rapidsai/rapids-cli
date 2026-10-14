#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ "${BUILD_TYPE:-}" == "release-candidate" ]]; then
  # Preserve the wheel profile before Hatch VCS reads the generic release alias.
  SETUPTOOLS_SCM_PRETEND_VERSION=$(jq -er '
    .unit_package_versions as $units | .package_versions as $families
    | if ($units | type) != "object" or ($families | type) != "object"
      then error("candidate package version maps are required")
      else (if $units | has("wheel:rapids-cli")
        then $units["wheel:rapids-cli"] else $families.wheel end)
        | select(type == "string" and length > 0)
      end
  ' <<< "${RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS:?verified candidate package versions are required}")
  export SETUPTOOLS_SCM_PRETEND_VERSION
fi

python -m pip wheel                     \
    -v                                  \
    --no-deps                           \
    --disable-pip-version-check         \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    .

ci/validate_wheel.sh "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name wheel_python rapids-cli rapids-cli --pure --arch any)"
export RAPIDS_PACKAGE_NAME
