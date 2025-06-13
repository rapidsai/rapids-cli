#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

set -euo pipefail

source rapids-date-string

rapids-print-env

pip install hatch
RAPIDS_PACKAGE_VERSION=$(hatch version)
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
