#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

python -m pip wheel                     \
    -v                                  \
    --no-deps                           \
    --disable-pip-version-check         \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    .

ci/validate_wheel.sh "${RAPIDS_WHEEL_BLD_OUTPUT_DR}"
