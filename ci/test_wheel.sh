#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# nx-cugraph is a pure wheel, which is part of generating the download path
WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="rapids-cli" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-github python)

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install \
    "$(echo "${WHEELHOUSE}"/rapids_cli*.whl)[test]"

python -m pytest
