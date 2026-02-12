.. SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Troubleshooting
===============

No GPUs Detected
----------------

``rapids doctor`` reports "No available GPUs detected".

1. Verify NVIDIA drivers are installed:

   .. code-block:: bash

      nvidia-smi

2. Check that GPU is visible from Python:

   .. code-block:: bash

      python -c "import pynvml; pynvml.nvmlInit(); print(pynvml.nvmlDeviceGetCount())"

3. If running in a container, ensure GPU passthrough is enabled:

   .. code-block:: bash

      docker run --gpus all ...

Insufficient Compute Capability
--------------------------------

"GPU requires compute capability 7 or higher".

RAPIDS requires Volta-generation GPUs or newer (compute capability 7.0+).
Supported GPUs include V100, A100, H100, and RTX 20xx/30xx/40xx series.
See https://developer.nvidia.com/cuda-gpus for a full list.

CUDA Version Issues
-------------------

"Unable to look up CUDA version".

1. Check your CUDA driver version:

   .. code-block:: bash

      nvidia-smi | grep "CUDA Version"

2. Ensure RAPIDS packages match your CUDA version:

   .. code-block:: bash

      # For CUDA 12.x
      pip install cudf-cu12

      # For CUDA 11.x
      pip install cudf-cu11

Low Memory Warning
------------------

"System Memory to total GPU Memory ratio not at least 2:1 ratio."

This is a warning, not a failure. RAPIDS recommends system RAM be at least
twice total GPU memory for optimal performance, particularly with Dask.
RAPIDS will still function with a lower ratio.

Custom Checks Not Discovered
-----------------------------

If ``rapids doctor --verbose`` doesn't show your custom check:

1. Verify the entry point is registered:

   .. code-block:: bash

      python -c "from importlib.metadata import entry_points; \
          print([ep.name for ep in entry_points(group='rapids_doctor_check')])"

2. Reinstall the package that provides the check:

   .. code-block:: bash

      pip install -e . --force-reinstall --no-deps

3. Check for import errors by importing the check function directly:

   .. code-block:: bash

      python -c "from my_package.checks import my_check"

   Import errors during discovery are silently suppressed
   (see ``contextlib.suppress`` in ``doctor.py``).

General Debugging Steps
-----------------------

1. Run with verbose output:

   .. code-block:: bash

      rapids doctor --verbose

2. Gather full environment information:

   .. code-block:: bash

      rapids debug --json > debug_info.json

3. Report issues at https://github.com/rapidsai/rapids-cli/issues with the
   debug output attached.
