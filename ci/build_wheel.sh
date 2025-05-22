#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

python -m pip wheel                     \
    -v                                  \
    --no-deps                           \
    --disable-pip-version-check         \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    .

RAPIDS_PY_WHEEL_NAME="rapids-cli" rapids-upload-wheels-to-s3 python "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
