.. SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Health Checks
=============

Built-in health check modules for verifying RAPIDS installation requirements.

GPU Checks
----------

.. automodule:: rapids_cli.doctor.checks.gpu
   :members:
   :undoc-members:
   :show-inheritance:

gpu_check
^^^^^^^^^

.. autofunction:: rapids_cli.doctor.checks.gpu.gpu_check

Verifies that NVIDIA GPUs are available and accessible.

**Parameters:**

- ``verbose`` (bool): Enable detailed output

**Returns:**

- str: Message indicating number of GPUs detected

**Raises:**

- ValueError: If no GPUs are detected
- AssertionError: If GPU count is zero

**Example:**

.. code-block:: python

   >>> gpu_check(verbose=True)
   'GPU(s) detected: 2'

check_gpu_compute_capability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: rapids_cli.doctor.checks.gpu.check_gpu_compute_capability

Verifies that all GPUs meet minimum compute capability requirements.

**Parameters:**

- ``verbose`` (bool): Enable detailed output

**Returns:**

- bool: True if all GPUs meet requirements

**Raises:**

- ValueError: If any GPU has insufficient compute capability

**Required Compute Capability:**

- Minimum: 7.0 (Volta architecture or newer)
- Supported GPUs: V100, A100, H100, RTX 20xx/30xx/40xx series

CUDA Driver Checks
------------------

.. automodule:: rapids_cli.doctor.checks.cuda_driver
   :members:
   :undoc-members:
   :show-inheritance:

cuda_check
^^^^^^^^^^

.. autofunction:: rapids_cli.doctor.checks.cuda_driver.cuda_check

Verifies CUDA driver availability and retrieves version.

**Parameters:**

- ``verbose`` (bool): Enable detailed output

**Returns:**

- int: CUDA driver version code (e.g., 12040 for CUDA 12.4)

**Raises:**

- ValueError: If CUDA driver version cannot be determined

**Example:**

.. code-block:: python

   >>> cuda_check(verbose=True)
   12040

Memory Checks
-------------

.. automodule:: rapids_cli.doctor.checks.memory
   :members:
   :undoc-members:
   :show-inheritance:

get_system_memory
^^^^^^^^^^^^^^^^^

.. autofunction:: rapids_cli.doctor.checks.memory.get_system_memory

Retrieves total system memory in gigabytes.

**Parameters:**

- ``verbose`` (bool): Unused, kept for consistency

**Returns:**

- float: Total system memory in GB

get_gpu_memory
^^^^^^^^^^^^^^

.. autofunction:: rapids_cli.doctor.checks.memory.get_gpu_memory

Calculates total GPU memory across all GPUs in gigabytes.

**Parameters:**

- ``verbose`` (bool): Unused, kept for consistency

**Returns:**

- float: Total GPU memory in GB

check_memory_to_gpu_ratio
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: rapids_cli.doctor.checks.memory.check_memory_to_gpu_ratio

Verifies system-to-GPU memory ratio meets recommendations.

**Parameters:**

- ``verbose`` (bool): Enable detailed output

**Returns:**

- bool: Always returns True (issues warnings instead of failing)

**Warnings:**

Issues warning if ratio is less than 1.8:1 (below recommended 2:1)

**Recommendation:**

For optimal performance, especially with Dask:

- System memory should be at least 2x total GPU memory
- Example: 64GB RAM for 32GB total GPU memory (2x 16GB GPUs)

NVLink Checks
-------------

.. automodule:: rapids_cli.doctor.checks.nvlink
   :members:
   :undoc-members:
   :show-inheritance:

check_nvlink_status
^^^^^^^^^^^^^^^^^^^

.. autofunction:: rapids_cli.doctor.checks.nvlink.check_nvlink_status

Checks for NVLink availability on multi-GPU systems.

**Parameters:**

- ``verbose`` (bool): Enable detailed output

**Returns:**

- bool: False if fewer than 2 GPUs, True if NVLink detected

**Raises:**

- ValueError: If NVLink status check fails on multi-GPU system

**NVLink Benefits:**

- High-bandwidth GPU-to-GPU communication
- Essential for multi-GPU training and processing
- Significantly faster than PCIe transfers

**Note:**

Only relevant for multi-GPU systems with NVLink-capable GPUs.

Check Function Contract
-----------------------

All built-in checks follow these conventions:

Function Signature
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def check_function(verbose=False, **kwargs):
       """Brief description of what this check verifies."""
       pass

**Parameters:**

- ``verbose`` (bool): Whether to provide detailed output
- ``**kwargs``: Reserved for future compatibility

Return Values
^^^^^^^^^^^^^

**Success:**

- Return any value (often True or a status string)
- Returning a string provides information for verbose output

**Failure:**

- Raise an exception with descriptive error message
- Use ValueError for failed checks
- Provide actionable guidance in error message

**Warnings:**

- Use ``warnings.warn()`` for non-fatal issues
- Always set ``stacklevel=2`` for correct source location

Usage in Custom Checks
-----------------------

Reference these built-in checks when creating custom checks:

.. code-block:: python

   # Example: Custom memory check based on built-in pattern
   from rapids_cli.doctor.checks.memory import get_gpu_memory


   def my_memory_check(verbose=False, **kwargs):
       """Check if GPU has enough memory for my workload."""
       gpu_memory = get_gpu_memory()

       required_gb = 16
       if gpu_memory < required_gb:
           raise ValueError(
               f"Insufficient GPU memory: {gpu_memory:.1f}GB available, "
               f"{required_gb}GB required"
           )

       if verbose:
           return f"GPU memory check passed: {gpu_memory:.1f}GB available"
       return True
