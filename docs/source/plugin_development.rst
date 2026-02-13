.. SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Plugin Development
==================

Any package can add checks to ``rapids doctor`` by exposing a function via a
Python entry point in the ``rapids_doctor_check`` group.

Quick Start
-----------

1. Create a check function:

   .. code-block:: python

      # my_package/health_checks.py


      def my_check(verbose=False, **kwargs):
          """Check that my_package is working correctly."""
          try:
              import my_package
          except ImportError as e:
              raise ImportError(
                  "my_package not found. Install with: pip install my_package"
              ) from e

          if verbose:
              return f"my_package {my_package.__version__} is available"

2. Register it in ``pyproject.toml``:

   .. code-block:: toml

      [project.entry-points.rapids_doctor_check]
      my_check = "my_package.health_checks:my_check"

3. Install and verify:

   .. code-block:: bash

      pip install -e .
      rapids doctor --verbose --dry-run

Check Execution Flow
--------------------

When ``rapids doctor`` runs, checks go through four stages:

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

Check Function Contract
-----------------------

Signature
^^^^^^^^^

.. code-block:: python

   def my_check(verbose=False, **kwargs):
       """First line of docstring is shown in output."""
       ...

- Accept ``verbose`` (bool) and ``**kwargs`` for forward compatibility.
- The first line of the docstring is used as the check description in output.
- New keyword arguments may be added in the future but will never be removed,
  so ``**kwargs`` ensures your check won't break.

Return Values
^^^^^^^^^^^^^

- **Pass**: Return any value. Returning a string provides extra info shown in
  ``--verbose`` mode.
- **Fail**: Raise an exception. The message should tell the user how to fix it.
- **Warn**: Call ``warnings.warn("message", stacklevel=2)`` for non-fatal issues.
  Warnings are captured and displayed but do not cause the check to fail.

Examples
--------

GPU memory requirement check:

.. code-block:: python

   import pynvml


   def gpu_memory_check(verbose=False, **kwargs):
       """Check that GPU has at least 8GB memory."""
       pynvml.nvmlInit()
       handle = pynvml.nvmlDeviceGetHandleByIndex(0)
       mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
       available_gb = mem.total / (1024**3)

       if available_gb < 8:
           raise ValueError(
               f"Insufficient GPU memory: {available_gb:.1f}GB available, 8GB required"
           )

       if verbose:
           return f"GPU memory: {available_gb:.1f}GB"

Non-fatal warning:

.. code-block:: python

   import warnings


   def config_check(verbose=False, **kwargs):
       """Check optional configuration."""
       if not optimal_condition():
           warnings.warn(
               "Suboptimal configuration detected. Performance may be degraded.",
               stacklevel=2,
           )

Multiple checks from one package:

.. code-block:: toml

   [project.entry-points.rapids_doctor_check]
   my_pkg_import = "my_package.checks:import_check"
   my_pkg_gpu = "my_package.checks:gpu_check"
   my_pkg_functional = "my_package.checks:functional_check"

Testing Your Plugin
-------------------

Verify discovery:

.. code-block:: bash

   rapids doctor --verbose --dry-run | grep my_check

Run only your checks:

.. code-block:: bash

   rapids doctor --verbose my_package

Unit test with mocks (following the pattern in ``rapids_cli/tests/``):

.. code-block:: python

   from unittest.mock import patch

   import pytest

   from my_package.health_checks import my_check


   def test_my_check_success():
       result = my_check(verbose=True)
       assert result is not None


   def test_my_check_failure():
       with pytest.raises(ValueError, match="expected error"):
           my_check(verbose=False)

Troubleshooting
---------------

**Check not discovered**: Verify the entry point name is in the output of:

.. code-block:: bash

   python -c "from importlib.metadata import entry_points; \
       print([ep.name for ep in entry_points(group='rapids_doctor_check')])"

If missing, reinstall with ``pip install -e . --force-reinstall --no-deps``.

**Import errors are silent**: The doctor module uses ``contextlib.suppress``
to skip checks that fail to import. Test your import directly:

.. code-block:: bash

   python -c "from my_package.health_checks import my_check"

See the built-in checks in ``rapids_cli/doctor/checks/`` for reference
implementations.
