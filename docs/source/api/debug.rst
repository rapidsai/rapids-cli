.. SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Debug Module
============

The debug module gathers comprehensive system information for troubleshooting.

.. automodule:: rapids_cli.debug.debug
   :members:
   :undoc-members:
   :show-inheritance:

Core Functions
--------------

run_debug
^^^^^^^^^

.. autofunction:: rapids_cli.debug.debug.run_debug

Main function for gathering and displaying debug information.

**Parameters:**

- ``output_format`` (str): Output format, either "console" or "json"

**Collected Information:**

- Date and time
- Platform information
- nvidia-smi output
- NVIDIA driver version
- CUDA version
- CUDA runtime path
- System CUDA toolkit locations
- Python version (full and short)
- Python hash info
- All installed package versions
- pip freeze output
- conda list output (if available)
- conda info output (if available)
- Available development tools
- OS information from /etc/os-release

gather_cuda_version
^^^^^^^^^^^^^^^^^^^

.. autofunction:: rapids_cli.debug.debug.gather_cuda_version

Retrieves and formats CUDA driver version from pynvml.

**Returns:**

- str: CUDA version in format "Major.Minor" or "Major.Minor.Patch"

**Example:**

.. code-block:: python

   >>> gather_cuda_version()
   '12.4'

gather_package_versions
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: rapids_cli.debug.debug.gather_package_versions

Collects versions of all installed Python packages.

**Returns:**

- dict: Mapping of package names to version strings

**Example:**

.. code-block:: python

   >>> versions = gather_package_versions()
   >>> versions['rapids-cli']
   '0.1.0'

gather_command_output
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: rapids_cli.debug.debug.gather_command_output

Executes a command and returns its output, with optional fallback.

**Parameters:**

- ``command`` (list[str]): Command and arguments to execute
- ``fallback_output`` (str | None): Value to return if command fails

**Returns:**

- str | None: Command output or fallback value

**Example:**

.. code-block:: python

   >>> gather_command_output(['pip', '--version'])
   'pip 24.0 from /usr/local/lib/python3.10/site-packages/pip (python 3.10)'

   >>> gather_command_output(['nonexistent'], fallback_output='Not installed')
   'Not installed'

gather_tools
^^^^^^^^^^^^

.. autofunction:: rapids_cli.debug.debug.gather_tools

Gathers version information for common development tools.

**Returns:**

- dict: Tool names mapped to version strings or None

**Checked Tools:**

- pip
- conda
- uv
- pixi
- g++
- cmake
- nvcc

Output Formats
--------------

Console Format
^^^^^^^^^^^^^^

Human-readable output with Rich formatting:

.. code-block:: text

   RAPIDS Debug Information

   Date
   2025-02-11 15:30:00

   Platform
   Linux-6.8.0-94-generic-x86_64

   Driver Version
   550.54.15

   Cuda Version
   12.4

   Package Versions
   ┌─────────────────┬──────────┐
   │ rapids-cli      │ 0.1.0    │
   │ cudf            │ 25.02.0  │
   └─────────────────┴──────────┘

JSON Format
^^^^^^^^^^^

Machine-readable output for automation:

.. code-block:: json

   {
     "date": "2025-02-11 15:30:00",
     "platform": "Linux-6.8.0-94-generic-x86_64",
     "nvidia_smi_output": "...",
     "driver_version": "550.54.15",
     "cuda_version": "12.4",
     "cuda_runtime_path": "/usr/local/cuda/include",
     "system_ctk": ["/usr/local/cuda-12.4"],
     "python_version_full": "3.13.12 (main, ...)",
     "python_version": "3.13.12",
     "python_hash_info": "sys.hash_info(...)",
     "package_versions": {
       "rapids-cli": "0.1.0"
     },
     "pip_packages": "...",
     "conda_packages": "...",
     "conda_info": "...",
     "tools": {
       "pip": "pip 24.0",
       "conda": "conda 24.1.0"
     },
     "os_info": {
       "NAME": "Ubuntu",
       "VERSION": "22.04"
     }
   }

Usage Examples
--------------

Console Output
^^^^^^^^^^^^^^

.. code-block:: python

   from rapids_cli.debug.debug import run_debug

   # Display debug info in console
   run_debug(output_format="console")

JSON Output
^^^^^^^^^^^

.. code-block:: python

   import json
   from rapids_cli.debug.debug import run_debug

   # Get JSON output
   run_debug(output_format="json")

   # Can be captured with redirection
   # rapids debug --json > debug.json

Programmatic Access
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from rapids_cli.debug.debug import gather_package_versions, gather_cuda_version

   # Get specific information
   cuda_ver = gather_cuda_version()
   packages = gather_package_versions()

   print(f"CUDA: {cuda_ver}")
   print(f"Installed packages: {len(packages)}")
