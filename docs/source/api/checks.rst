.. SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Health Checks
=============

Built-in health check modules registered via the ``rapids_doctor_check``
entry point group in ``pyproject.toml``.

All check functions follow the contract described in :doc:`../plugin_development`.

GPU Checks
----------

.. automodule:: rapids_cli.doctor.checks.gpu
   :members:
   :undoc-members:
   :show-inheritance:

CUDA Driver Checks
------------------

.. automodule:: rapids_cli.doctor.checks.cuda_driver
   :members:
   :undoc-members:
   :show-inheritance:

Memory Checks
-------------

.. automodule:: rapids_cli.doctor.checks.memory
   :members:
   :undoc-members:
   :show-inheritance:

NVLink Checks
-------------

.. automodule:: rapids_cli.doctor.checks.nvlink
   :members:
   :undoc-members:
   :show-inheritance:
