#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

python -m pip wheel                     \
    -v                                  \
    --no-deps                           \
    --disable-pip-version-check         \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    .
