.. SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Doctor Module
=============

The ``rapids_cli.doctor.doctor`` module orchestrates health check discovery
and execution.

Checks are discovered via Python entry points in the ``rapids_doctor_check``
group. Each check function is called with ``verbose`` as a keyword argument.
Results are collected into :class:`CheckResult` objects that track pass/fail
status, return values, errors, and warnings.

Check Execution Flow
--------------------

1. **Discovery**: Scan ``rapids_doctor_check`` entry points and load check
   functions. ``ImportError`` and ``AttributeError`` during loading are
   silently suppressed via ``contextlib.suppress``.

2. **Filtering**: If filter arguments are provided, only checks whose
   ``ep.value`` contains a filter substring are kept.

3. **Execution**: Each check runs inside ``warnings.catch_warnings(record=True)``
   so warnings are captured. Exceptions are caught and stored rather than
   propagated.

4. **Reporting**: Warnings are printed, verbose output is shown for passing
   checks, and failed checks are listed with their error messages.

API
---

.. automodule:: rapids_cli.doctor.doctor
   :members:
   :undoc-members:
   :show-inheritance:
