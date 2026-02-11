.. SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Plugin Development Guide
========================

The RAPIDS CLI uses a plugin system based on Python entry points to allow external packages
to register their own health checks. This guide shows you how to create plugins for your
RAPIDS library.

Overview
--------

Plugins are discovered automatically through Python entry points in the ``rapids_doctor_check``
group. When ``rapids doctor`` runs, it discovers all registered checks and executes them.

Quick Start
-----------

Here's a minimal example of adding a check to your RAPIDS package:

1. Create a check function in your package:

   .. code-block:: python

      # my_rapids_package/health_checks.py


      def my_package_check(verbose=False, **kwargs):
          """Check that my_rapids_package is working correctly."""
          import my_rapids_package

          # Perform your check
          result = my_rapids_package.test_function()

          if not result:
              raise ValueError("my_rapids_package self-test failed")

          return "my_rapids_package is working correctly"

2. Register the check in your ``pyproject.toml``:

   .. code-block:: toml

      [project.entry-points.rapids_doctor_check]
      my_package_check = "my_rapids_package.health_checks:my_package_check"

3. Install your package and test:

   .. code-block:: bash

      pip install -e .
      rapids doctor --verbose

Check Function Contract
-----------------------

Your check function must follow these conventions:

Function Signature
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def my_check(verbose=False, **kwargs):
       """Check description goes here."""
       pass

- Accept ``verbose`` parameter (boolean, default False)
- Accept ``**kwargs`` for forward compatibility
- Provide a clear docstring (first line is used in output)

Return Values
^^^^^^^^^^^^^

**Success**: Return successfully (any return value)

.. code-block:: python

   def check_success(verbose=False, **kwargs):
       """This check always passes."""
       # Option 1: Return None (implicit)
       return


   def check_with_info(verbose=False, **kwargs):
       """This check passes with info."""
       # Option 2: Return a string for verbose output
       return "GPU 0: Tesla V100, 32GB memory"

**Failure**: Raise an exception with a helpful message

.. code-block:: python

   def check_failure(verbose=False, **kwargs):
       """This check fails with helpful message."""
       if not some_condition():
           raise ValueError(
               "Check failed: XYZ is not configured correctly. "
               "To fix this, run: sudo apt-get install xyz"
           )

**Warnings**: Use ``warnings.warn()`` for non-fatal issues

.. code-block:: python

   import warnings


   def check_with_warning(verbose=False, **kwargs):
       """This check passes but issues a warning."""
       if not optimal_condition():
           warnings.warn(
               "Suboptimal configuration detected. " "Performance may be degraded.",
               stacklevel=2,
           )
       return True

Examples
--------

Example 1: Basic Import Check
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Check that your package can be imported:

.. code-block:: python

   def import_check(verbose=False, **kwargs):
       """Check that my_package can be imported."""
       try:
           import my_package
       except ImportError as e:
           raise ImportError(
               "my_package not found. Install with: pip install my_package"
           ) from e

       if verbose:
           return f"my_package version {my_package.__version__}"
       return True

Example 2: GPU Memory Check
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Check GPU memory requirements:

.. code-block:: python

   import pynvml


   def gpu_memory_check(verbose=False, **kwargs):
       """Check that GPU has sufficient memory for my_package."""
       pynvml.nvmlInit()

       required_memory_gb = 8
       handle = pynvml.nvmlDeviceGetHandleByIndex(0)
       memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
       available_gb = memory_info.total / (1024**3)

       if available_gb < required_memory_gb:
           raise ValueError(
               f"Insufficient GPU memory: {available_gb:.1f}GB available, "
               f"{required_memory_gb}GB required"
           )

       if verbose:
           return f"GPU memory: {available_gb:.1f}GB available"
       return True

Example 3: Dependency Version Check
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Check that dependencies meet version requirements:

.. code-block:: python

   import warnings
   from packaging import version


   def dependency_version_check(verbose=False, **kwargs):
       """Check that dependencies meet minimum version requirements."""
       import numpy
       import pandas

       min_numpy = "1.20.0"
       min_pandas = "1.3.0"

       if version.parse(numpy.__version__) < version.parse(min_numpy):
           raise ValueError(
               f"NumPy {min_numpy}+ required, found {numpy.__version__}. "
               f"Upgrade with: pip install 'numpy>={min_numpy}'"
           )

       if version.parse(pandas.__version__) < version.parse(min_pandas):
           warnings.warn(
               f"Pandas {min_pandas}+ recommended for best performance. "
               f"Found {pandas.__version__}.",
               stacklevel=2,
           )

       if verbose:
           return f"NumPy {numpy.__version__}, Pandas {pandas.__version__}"
       return True

Example 4: Functional Test
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run a simple functional test:

.. code-block:: python

   def functional_check(verbose=False, **kwargs):
       """Run a simple functional test."""
       import my_package
       import numpy as np

       try:
           # Create test data
           data = np.random.rand(100, 10)

           # Run simple operation
           result = my_package.process(data)

           # Verify result
           assert result.shape == (100, 10), "Unexpected output shape"
           assert not np.isnan(result).any(), "NaN values in output"

       except Exception as e:
           raise RuntimeError(
               f"Functional test failed: {e}. " "This may indicate a GPU or driver issue."
           ) from e

       if verbose:
           return "Functional test passed: basic operations working"
       return True

Best Practices
--------------

Clear Error Messages
^^^^^^^^^^^^^^^^^^^^

Always provide actionable error messages:

.. code-block:: python

   # Bad: Unclear what to do
   raise ValueError("Check failed")

   # Good: Clear action to fix
   raise ValueError(
       "CUDA 11.2+ required but CUDA 10.2 found. "
       "Upgrade CUDA: https://developer.nvidia.com/cuda-downloads"
   )

Performance
^^^^^^^^^^^

Keep checks fast (< 1 second each):

.. code-block:: python

   # Bad: Slow check
   def slow_check(verbose=False, **kwargs):
       """This check is too slow."""
       result = expensive_computation()  # Takes 30 seconds
       return result


   # Good: Fast check
   def fast_check(verbose=False, **kwargs):
       """This check is appropriately fast."""
       # Just verify configuration, don't run full workload
       config = load_config()
       validate_config(config)
       return True

Verbose Output
^^^^^^^^^^^^^^

Provide useful information in verbose mode:

.. code-block:: python

   def informative_check(verbose=False, **kwargs):
       """Check with informative output."""
       gpu_count = get_gpu_count()
       gpu_memory = get_total_gpu_memory()

       if gpu_count == 0:
           raise ValueError("No GPUs found")

       if verbose:
           return f"Found {gpu_count} GPU(s) " f"with {gpu_memory:.1f}GB total memory"
       return True

Graceful Degradation
^^^^^^^^^^^^^^^^^^^^

Handle optional dependencies gracefully:

.. code-block:: python

   def optional_dependency_check(verbose=False, **kwargs):
       """Check that works with optional dependencies."""
       try:
           import optional_package

           has_optional = True
       except ImportError:
           has_optional = False

       if not has_optional:
           import warnings

           warnings.warn(
               "optional_package not found. " "Some features will be disabled.",
               stacklevel=2,
           )

       # Continue with check anyway
       return True

Testing Your Plugin
-------------------

Test Plugin Discovery
^^^^^^^^^^^^^^^^^^^^^

Verify your check is discovered:

.. code-block:: bash

   rapids doctor --verbose --dry-run | grep my_check

Test Plugin Execution
^^^^^^^^^^^^^^^^^^^^^

Run your check:

.. code-block:: bash

   rapids doctor --verbose my_package

Unit Testing
^^^^^^^^^^^^

Test your check function directly:

.. code-block:: python

   # test_health_checks.py
   import pytest
   from my_package.health_checks import my_check


   def test_my_check_success():
       """Test that check passes in normal conditions."""
       result = my_check(verbose=True)
       assert result is not None


   def test_my_check_failure():
       """Test that check fails appropriately."""
       with pytest.raises(ValueError, match="expected error"):
           my_check_with_bad_config(verbose=False)

Advanced Topics
---------------

Multiple Checks per Package
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Register multiple checks:

.. code-block:: toml

   [project.entry-points.rapids_doctor_check]
   my_pkg_import = "my_package.checks:import_check"
   my_pkg_gpu = "my_package.checks:gpu_check"
   my_pkg_functional = "my_package.checks:functional_check"

Check Dependencies
^^^^^^^^^^^^^^^^^^

If checks have dependencies, handle them gracefully:

.. code-block:: python

   def dependent_check(verbose=False, **kwargs):
       """This check depends on GPU check passing."""
       # Don't fail if dependencies aren't met
       try:
           import pynvml

           pynvml.nvmlInit()
       except Exception:
           warnings.warn("GPU not available, skipping dependent check", stacklevel=2)
           return True

       # Rest of check
       return True

Environment-Specific Checks
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adapt checks to different environments:

.. code-block:: python

   import os


   def environment_aware_check(verbose=False, **kwargs):
       """Check that adapts to environment."""
       is_ci = os.environ.get("CI") == "true"

       if is_ci:
           # Skip expensive checks in CI
           return "Skipped in CI environment"

       # Run full check
       run_expensive_validation()
       return True

Troubleshooting
---------------

Check Not Discovered
^^^^^^^^^^^^^^^^^^^^

If your check isn't showing up:

1. Verify entry point is correct:

   .. code-block:: bash

      python -c "from importlib.metadata import entry_points; print([ep for ep in entry_points(group='rapids_doctor_check')])"

2. Reinstall your package:

   .. code-block:: bash

      pip install -e . --force-reinstall --no-deps

3. Check for import errors:

   .. code-block:: python

      python -c "from my_package.checks import my_check"

Check Always Fails
^^^^^^^^^^^^^^^^^^

Debug the check directly:

.. code-block:: python

   from my_package.checks import my_check

   try:
       result = my_check(verbose=True)
       print(f"Success: {result}")
   except Exception as e:
       print(f"Failed: {e}")
       import traceback

       traceback.print_exc()

Resources
---------

- Entry points documentation: https://packaging.python.org/specifications/entry-points/
- RAPIDS CLI repository: https://github.com/rapidsai/rapids-cli
- Example plugins: See built-in checks in ``rapids_cli/doctor/checks/``
