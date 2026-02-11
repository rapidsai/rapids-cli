.. SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Doctor Module
=============

The doctor module orchestrates health check execution and plugin discovery.

.. automodule:: rapids_cli.doctor.doctor
   :members:
   :undoc-members:
   :show-inheritance:

Core Functions
--------------

doctor_check
^^^^^^^^^^^^

.. autofunction:: rapids_cli.doctor.doctor.doctor_check

The main orchestration function for running health checks.

**Parameters:**

- ``verbose`` (bool): Enable detailed output
- ``dry_run`` (bool): Discover checks without executing them
- ``filters`` (list[str] | None): Optional filters to match check paths

**Returns:**

- bool: True if all checks passed, False if any failed

**Process:**

1. Discovers all registered checks via entry points
2. Filters checks based on provided filters
3. Executes each check and captures results
4. Collects warnings from checks
5. Displays results and returns success status

CheckResult
^^^^^^^^^^^

.. autoclass:: rapids_cli.doctor.doctor.CheckResult
   :members:

Data class representing the result of a single check execution.

**Attributes:**

- ``name`` (str): Name of the check function
- ``description`` (str): First line of check's docstring
- ``status`` (bool): True if check passed, False if failed
- ``value`` (str | None): Optional return value for verbose output
- ``error`` (Exception | None): Exception if check failed
- ``warnings`` (list[WarningMessage] | None): Any warnings issued during check

Plugin Discovery
----------------

The doctor module discovers plugins using Python entry points:

.. code-block:: python

   from importlib.metadata import entry_points

   for ep in entry_points(group="rapids_doctor_check"):
       check_fn = ep.load()
       # Execute check

Entry Point Group
^^^^^^^^^^^^^^^^^

Plugins register in the ``rapids_doctor_check`` group:

.. code-block:: toml

   [project.entry-points.rapids_doctor_check]
   my_check = "my_package.checks:my_check_function"

Check Execution Flow
--------------------

1. **Discovery Phase**

   - Scan entry points for ``rapids_doctor_check`` group
   - Load check functions
   - Apply filters if specified

2. **Execution Phase**

   - Run each check with ``verbose`` parameter
   - Capture warnings using ``warnings.catch_warnings()``
   - Catch exceptions for failed checks
   - Store results in CheckResult objects

3. **Reporting Phase**

   - Display warnings
   - Show verbose output if requested
   - List failed checks with error messages
   - Return overall success status

Error Handling
--------------

The doctor module handles several error scenarios:

**Import Errors**

Failed imports during discovery are suppressed with ``contextlib.suppress``:

.. code-block:: python

   with contextlib.suppress(AttributeError, ImportError):
       check_fn = ep.load()

**Check Exceptions**

Exceptions raised by checks are caught and stored:

.. code-block:: python

   try:
       value = check_fn(verbose=verbose)
       status = True
   except Exception as e:
       error = e
       status = False

**Warnings**

Python warnings are captured and displayed:

.. code-block:: python

   with warnings.catch_warnings(record=True) as w:
       warnings.simplefilter("always")
       value = check_fn(verbose=verbose)
       caught_warnings = w
