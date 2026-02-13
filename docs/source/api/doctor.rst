.. SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Doctor Module
=============

The ``rapids_cli.doctor.doctor`` module orchestrates health check discovery
and execution.

The CUDA + Python software stack spans multiple layers — the GPU driver, the
CUDA runtime, C/C++ libraries, and Python packages — each managed by different
package managers (OS packages, CUDA toolkit installers, conda, pip). Because no
single package manager owns the full stack, misconfigurations across layer
boundaries are common and difficult to diagnose. ``rapids doctor`` validates
compatibility across these layers, from the driver through CUDA to the Python
libraries, and provides actionable feedback when an incompatibility is found.

Health checks are bundled with the RAPIDS libraries you install rather than
hard-coded into ``rapids-cli`` itself. When you install a library such as
cuDF or cuML, any checks it ships are automatically available to
``rapids doctor`` — no extra configuration required.

You can run all discovered checks at once, or filter to a specific library
by passing its name as an argument:

.. code-block:: bash

   # Run every available check
   rapids doctor

   # Run only checks related to cudf
   rapids doctor cudf

   # See which checks would run without executing them
   rapids doctor --dry-run --verbose

Results are collected into :class:`~rapids_cli.doctor.doctor.CheckResult`
objects that track pass/fail status, return values, errors, and warnings.
For details on how checks are discovered and executed, or how to write your
own, see :doc:`/plugin_development`.

API
---

.. automodule:: rapids_cli.doctor.doctor
   :members:
   :undoc-members:
   :show-inheritance:
