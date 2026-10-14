#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-datetime-string

rapids-print-env

rapids-mamba-retry install \
  --yes \
  'hatchling' 'hatch-vcs'
if [[ "${BUILD_TYPE:-}" == "release-candidate" ]]; then
  # Conda identity comes from the verified train, not normalized Python metadata.
  RAPIDS_PACKAGE_VERSION=$(jq -er '
    .unit_package_versions as $units | .package_versions as $families
    | if ($units | type) != "object" or ($families | type) != "object"
      then error("candidate package version maps are required")
      else (if $units | has("conda:rapids-cli")
        then $units["conda:rapids-cli"] else $families.conda end)
        | select(type == "string" and length > 0)
      end
  ' <<< "${RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS:?verified candidate package versions are required}")
else
  RAPIDS_PACKAGE_VERSION=$(hatchling version)
fi
export RAPIDS_PACKAGE_VERSION

# populates `RATTLER_CHANNELS` array and `RATTLER_ARGS` array
source rapids-rattler-channel-string

rapids-logger "Building rapids-cli"

rattler-build build                   \
    --recipe conda/recipes/rapids-cli \
    "${RATTLER_ARGS[@]}"              \
    "${RATTLER_CHANNELS[@]}"

# remove build_cache directory to avoid uploading the entire source tree
# tracked in https://github.com/prefix-dev/rattler-build/issues/1424
rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache

# set package name so it is picked up by the shared-workflow for artifact naming
RAPIDS_PACKAGE_NAME="$(rapids-artifact-name conda_python rapids-cli rapids-cli --pure --arch any)"
export RAPIDS_PACKAGE_NAME
