.. SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

User Guide
==========

The RAPIDS CLI provides two commands: ``rapids doctor`` for health checks and
``rapids debug`` for gathering system information.

rapids doctor
-------------

The ``doctor`` command performs health checks to ensure your RAPIDS environment
is properly configured.

.. code-block:: bash

   rapids doctor

Built-in checks verify:

- GPU availability and compute capability (7.0+)
- CUDA driver version
- System memory to GPU memory ratio (recommends 2:1 for Dask)
- NVLink status (multi-GPU systems)

Any installed RAPIDS library can register additional checks via the plugin system
(see :doc:`plugin_development`).

Verbose Output
^^^^^^^^^^^^^^

The ``--verbose`` flag shows check discovery details and per-check output:

.. code-block:: bash

   $ rapids doctor --verbose
   Discovering checks
   Found check 'gpu' provided by 'rapids_cli.doctor.checks.gpu:gpu_check'
   ...
   Discovered 5 checks
   Running checks
   gpu_check: GPU(s) detected: 2
   All checks passed!

Dry Run
^^^^^^^

The ``--dry-run`` flag discovers checks without executing them, useful for
verifying plugin registration:

.. code-block:: bash

   rapids doctor --dry-run

Filtering
^^^^^^^^^

Pass filter arguments to run only matching checks. Filters match against
the check's module path:

.. code-block:: bash

   # Run only cuDF-related checks
   rapids doctor cudf

   # Run checks from multiple packages
   rapids doctor cudf cuml

Exit Codes
^^^^^^^^^^

- ``0``: All checks passed
- ``1``: One or more checks failed

This makes ``rapids doctor`` suitable for scripting:

.. code-block:: bash

   rapids doctor || exit 1

rapids debug
------------

The ``debug`` command gathers comprehensive system information for troubleshooting.

.. code-block:: bash

   rapids debug

Output includes: platform, NVIDIA driver version, CUDA version, CUDA runtime
path, system CTK locations, Python version, all installed package versions,
pip/conda package lists, available tools (pip, conda, uv, pixi, g++, cmake,
nvcc), and OS information.

JSON Output
^^^^^^^^^^^

The ``--json`` flag produces machine-readable output:

.. code-block:: bash

   rapids debug --json > debug_info.json

This is useful for attaching to bug reports or comparing environments.

CI/CD Integration
-----------------

Example GitHub Actions usage:

.. code-block:: yaml

   - name: Verify RAPIDS Environment
     run: |
       pip install rapids-cli
       rapids doctor --verbose || exit 1

   - name: Save Debug Info on Failure
     if: failure()
     run: rapids debug --json > debug.json
