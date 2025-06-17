#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Downloading artifacts from previous jobs"
PYTHON_CHANNEL=$(rapids-download-conda-from-github python)

# ensure we always install the just-built-in-CI package, instead of falling back to earlier ones
conda config --set channel_priority strict

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

rapids-logger "Checking CLI is available and check discovery is working"

# Sometimes we choose not to explore further.
# If this is left as a "clean" (non-echo'd) command, the output doesn't get
# properly grepped
# shellcheck disable=SC2005
if ! echo "$(rapids doctor --verbose --dry-run)" | grep -q "Found check"; then
  echo "Error: 'Found check' not found in output"
  exit 1
fi

rapids-logger "running tests"
python -m pytest
