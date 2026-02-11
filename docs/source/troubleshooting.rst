.. SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Troubleshooting
===============

This guide helps you resolve common issues with the RAPIDS CLI.

Common Issues
-------------

No GPUs Detected
^^^^^^^^^^^^^^^^

**Symptom**: ``rapids doctor`` reports "No available GPUs detected"

**Solutions**:

1. Verify NVIDIA drivers are installed:

   .. code-block:: bash

      nvidia-smi

   If this fails, install NVIDIA drivers:

   .. code-block:: bash

      # Ubuntu/Debian
      sudo apt-get install nvidia-driver-550

2. Check that GPU is visible to Python:

   .. code-block:: bash

      python -c "import pynvml; pynvml.nvmlInit(); print(pynvml.nvmlDeviceGetCount())"

3. Verify you're not in a container without GPU access:

   .. code-block:: bash

      # Docker needs --gpus all flag
      docker run --gpus all ...

CUDA Version Mismatch
^^^^^^^^^^^^^^^^^^^^^

**Symptom**: ``rapids doctor`` reports CUDA version incompatibility

**Solutions**:

1. Check your CUDA driver version:

   .. code-block:: bash

      nvidia-smi | grep "CUDA Version"

2. Install compatible RAPIDS packages:

   .. code-block:: bash

      # For CUDA 11.x
      pip install cudf-cu11 cuml-cu11

      # For CUDA 12.x
      pip install cudf-cu12 cuml-cu12

3. Update NVIDIA drivers if needed:

   .. code-block:: bash

      # Check https://docs.rapids.ai/install for requirements
      sudo apt-get update && sudo apt-get upgrade nvidia-driver

Low Memory Warning
^^^^^^^^^^^^^^^^^^

**Symptom**: Warning about system memory to GPU memory ratio

**Context**: RAPIDS recommends 2:1 ratio of system RAM to GPU memory for optimal performance

**Solutions**:

1. This is a warning, not an error. RAPIDS will still work.

2. For better performance, consider:

   - Adding more system RAM
   - Using data chunking strategies
   - Processing smaller batches

3. For Dask workloads, adjust worker memory limits:

   .. code-block:: python

      from dask_cuda import LocalCUDACluster

      cluster = LocalCUDACluster(
          device_memory_limit="8GB",  # Limit per worker
          memory_limit="16GB",  # System memory per worker
      )

NVLink Not Found
^^^^^^^^^^^^^^^^

**Symptom**: ``rapids doctor`` reports NVLink is not available

**Context**: NVLink is only available on multi-GPU systems with NVLink-capable GPUs

**Solutions**:

1. If you have only one GPU, this is expected. NVLink is not needed.

2. For multi-GPU systems without NVLink:

   - RAPIDS will work but inter-GPU transfers will be slower
   - Consider PCIe topology optimization

3. Verify NVLink status:

   .. code-block:: bash

      nvidia-smi nvlink --status

Insufficient Compute Capability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom**: "GPU requires compute capability 7.0 or higher"

**Context**: RAPIDS requires GPU compute capability 7.0+ (Volta architecture or newer)

**Solutions**:

1. Check your GPU compute capability:

   .. code-block:: bash

      rapids debug | grep "GPU"

2. Supported GPUs include:

   - Tesla V100, A100, H100
   - RTX 20xx, 30xx, 40xx series
   - GTX 1660 and above

3. If your GPU is too old, you'll need to upgrade hardware.

Check Discovery Issues
^^^^^^^^^^^^^^^^^^^^^^

**Symptom**: Custom checks not discovered by ``rapids doctor``

**Solutions**:

1. Verify entry point registration:

   .. code-block:: bash

      python -c "from importlib.metadata import entry_points; \
                 print([ep.name for ep in entry_points(group='rapids_doctor_check')])"

2. Reinstall package with entry points:

   .. code-block:: bash

      pip install -e . --force-reinstall

3. Check for import errors:

   .. code-block:: bash

      rapids doctor --verbose

   Look for "Failed to import" messages.

Import Errors
^^^^^^^^^^^^^

**Symptom**: "ModuleNotFoundError" when running checks

**Solutions**:

1. Verify package is installed:

   .. code-block:: bash

      pip list | grep rapids

2. Check Python environment:

   .. code-block:: bash

      which python
      python --version

3. Ensure you're in the correct virtual environment:

   .. code-block:: bash

      # Conda
      conda activate rapids-env

      # venv
      source venv/bin/activate

Permission Errors
^^^^^^^^^^^^^^^^^

**Symptom**: "Permission denied" when accessing GPU

**Solutions**:

1. Add user to video/render groups:

   .. code-block:: bash

      sudo usermod -a -G video $USER
      sudo usermod -a -G render $USER

      # Log out and back in for changes to take effect

2. Check device permissions:

   .. code-block:: bash

      ls -l /dev/nvidia*

3. For containers, ensure proper device mounting:

   .. code-block:: bash

      docker run --gpus all --device=/dev/nvidia0 ...

Debugging Tips
--------------

Enable Verbose Mode
^^^^^^^^^^^^^^^^^^^

Always start with verbose output:

.. code-block:: bash

   rapids doctor --verbose

This shows:

- Which checks are discovered
- Detailed error messages
- Stack traces for failures

Gather Debug Information
^^^^^^^^^^^^^^^^^^^^^^^^^

Collect comprehensive system information:

.. code-block:: bash

   rapids debug --json > debug_info.json

Share this file when reporting issues.

Test Individual Components
^^^^^^^^^^^^^^^^^^^^^^^^^^

Test NVIDIA stack components:

.. code-block:: bash

   # Test nvidia-smi
   nvidia-smi

   # Test pynvml (Python binding)
   python -c "import pynvml; pynvml.nvmlInit(); print('OK')"

   # Test CUDA
   python -c "import cuda; print(cuda.cudaroot)"

Check Environment Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Verify CUDA-related environment variables:

.. code-block:: bash

   echo $CUDA_HOME
   echo $LD_LIBRARY_PATH
   echo $PATH

Run in Isolation
^^^^^^^^^^^^^^^^

Test in a clean environment:

.. code-block:: bash

   # Create fresh environment
   conda create -n test-rapids python=3.10
   conda activate test-rapids

   # Install only RAPIDS CLI
   pip install rapids-cli

   # Test
   rapids doctor

Enable Python Warnings
^^^^^^^^^^^^^^^^^^^^^^

See all warnings:

.. code-block:: bash

   python -W all -m rapids_cli.cli doctor

Performance Issues
------------------

Slow Check Execution
^^^^^^^^^^^^^^^^^^^^

If checks are slow:

1. Use ``--dry-run`` to verify discovery without execution:

   .. code-block:: bash

      rapids doctor --dry-run

2. Profile individual checks:

   .. code-block:: python

      import time
      from my_package.checks import my_check

      start = time.time()
      my_check(verbose=True)
      print(f"Check took {time.time() - start:.2f}s")

3. Optimize slow checks (keep under 1 second each)

High Memory Usage
^^^^^^^^^^^^^^^^^

If ``rapids doctor`` uses too much memory:

1. This is unexpected - report as a bug

2. Workaround: Run checks individually:

   .. code-block:: bash

      rapids doctor package1
      rapids doctor package2

Reporting Issues
----------------

When reporting issues, include:

1. Output of ``rapids debug --json``

2. Complete error messages from ``rapids doctor --verbose``

3. Steps to reproduce

4. Expected vs actual behavior

5. Environment details:

   .. code-block:: bash

      rapids debug > environment.txt
      python --version
      pip list > packages.txt

Submit issues at: https://github.com/rapidsai/rapids-cli/issues

Getting Help
------------

- GitHub Issues: https://github.com/rapidsai/rapids-cli/issues
- RAPIDS Slack: https://rapids.ai/community
- Documentation: https://docs.rapids.ai
- Stack Overflow: Tag questions with ``rapids`` and ``rapids-cli``

Known Limitations
-----------------

- Windows support is experimental
- WSL2 requires special GPU setup
- Some checks require sudo access
- Docker containers need ``--gpus all`` flag
- Remote GPU monitoring not supported
