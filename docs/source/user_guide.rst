.. SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

User Guide
==========

This guide provides detailed information on using the RAPIDS CLI.

Overview
--------

The RAPIDS CLI provides two main commands:

- ``rapids doctor`` - Health checks for your RAPIDS installation
- ``rapids debug`` - Gather debugging information about your system

rapids doctor
-------------

The ``doctor`` command performs health checks to ensure your RAPIDS environment is properly configured.

Basic Usage
^^^^^^^^^^^

Run all health checks:

.. code-block:: bash

   rapids doctor

This will check:

- GPU availability and compatibility
- CUDA driver version
- System memory to GPU memory ratio
- NVLink status (for multi-GPU systems)
- Any checks registered by installed RAPIDS packages

Verbose Output
^^^^^^^^^^^^^^

Get detailed information about each check:

.. code-block:: bash

   rapids doctor --verbose

This shows:

- Which checks are discovered
- Detailed output from each check
- Additional diagnostic information

Dry Run
^^^^^^^

See which checks would run without actually executing them:

.. code-block:: bash

   rapids doctor --dry-run

This is useful for:

- Verifying plugin discovery
- Debugging check registration issues
- Understanding what will be checked

Filtering Checks
^^^^^^^^^^^^^^^^

Run only specific checks by filtering:

.. code-block:: bash

   # Run only cuDF-related checks
   rapids doctor cudf

   # Run multiple filtered checks
   rapids doctor cudf cuml

The filter matches any part of the check's module path.

Exit Codes
^^^^^^^^^^

The ``doctor`` command returns:

- ``0`` - All checks passed
- ``1`` - One or more checks failed

This makes it suitable for use in scripts and CI/CD pipelines:

.. code-block:: bash

   if rapids doctor; then
       echo "Environment is ready!"
   else
       echo "Environment has issues!"
       exit 1
   fi

rapids debug
------------

The ``debug`` command gathers comprehensive information about your system for troubleshooting.

Basic Usage
^^^^^^^^^^^

Generate a debug report:

.. code-block:: bash

   rapids debug

This displays:

- Platform information
- NVIDIA driver version
- CUDA version
- Python version and configuration
- Installed package versions
- System tools (pip, conda, cmake, etc.)
- OS information

JSON Output
^^^^^^^^^^^

Get machine-readable output:

.. code-block:: bash

   rapids debug --json

This is useful for:

- Automated debugging scripts
- Parsing in other tools
- Sharing debug information programmatically

The JSON output includes all information in a structured format:

.. code-block:: json

   {
     "date": "2025-02-11 15:30:00",
     "platform": "Linux-6.8.0-94-generic-x86_64",
     "driver_version": "550.54.15",
     "cuda_version": "12.4",
     "python_version": "3.13.12",
     "package_versions": {
       "rapids-cli": "0.1.0",
       ...
     },
     ...
   }

Saving Debug Output
^^^^^^^^^^^^^^^^^^^

Save debug information to a file:

.. code-block:: bash

   rapids debug --json > debug_info.json

This file can be:

- Shared with support teams
- Attached to bug reports
- Used for comparison across environments

Common Workflows
----------------

Pre-Installation Check
^^^^^^^^^^^^^^^^^^^^^^

Before installing RAPIDS, verify your system meets requirements:

.. code-block:: bash

   # Install just the CLI first
   pip install rapids-cli

   # Check system compatibility
   rapids doctor --verbose

The checks will tell you if your GPU, drivers, and CUDA are suitable for RAPIDS.

Post-Installation Verification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After installing RAPIDS packages, verify everything works:

.. code-block:: bash

   # Install RAPIDS
   pip install cudf-cu12 cuml-cu12

   # Verify installation
   rapids doctor

   # If issues occur, gather debug info
   rapids debug --json > debug_info.json

CI/CD Integration
^^^^^^^^^^^^^^^^^

Use RAPIDS CLI in your CI/CD pipelines:

.. code-block:: yaml

   # GitHub Actions example
   - name: Verify RAPIDS Environment
     run: |
       pip install rapids-cli
       rapids doctor --verbose || exit 1

   - name: Save Debug Info on Failure
     if: failure()
     run: rapids debug --json > ${{ github.workspace }}/debug.json

Troubleshooting Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^

When encountering issues:

1. Run verbose health check:

   .. code-block:: bash

      rapids doctor --verbose

2. Review warning messages and failures

3. Gather full debug information:

   .. code-block:: bash

      rapids debug > debug_output.txt

4. Check troubleshooting guide (see :doc:`troubleshooting`)

5. Report issues with debug output

Best Practices
--------------

Regular Health Checks
^^^^^^^^^^^^^^^^^^^^^

Run ``rapids doctor`` regularly to catch configuration drift:

.. code-block:: bash

   # Add to your shell profile
   alias rapids-check='rapids doctor && echo "✓ RAPIDS environment healthy"'

Environment Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^

Document your environment with debug output:

.. code-block:: bash

   # Save baseline configuration
   rapids debug --json > baseline_env.json

   # Later, compare environments
   rapids debug --json > current_env.json
   diff baseline_env.json current_env.json

Automated Monitoring
^^^^^^^^^^^^^^^^^^^^

Monitor RAPIDS environments automatically:

.. code-block:: bash

   #!/bin/bash
   # daily_rapids_check.sh

   if ! rapids doctor; then
       rapids debug --json | mail -s "RAPIDS Health Check Failed" admin@example.com
   fi

Add to cron:

.. code-block:: bash

   0 9 * * * /path/to/daily_rapids_check.sh
