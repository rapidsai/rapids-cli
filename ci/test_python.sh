#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Downloading artifacts from previous jobs"
PYTHON_CHANNEL=$(rapids-download-conda-from-github python)

rapids-logger "Generating Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${PYTHON_CHANNEL}" \
| tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test-env

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test-env
set -u

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "running tests"
python -m pytest
