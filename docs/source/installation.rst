.. SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Installation
============

Requirements
------------

- Python 3.10 or later
- NVIDIA GPU (for running health checks)
- NVIDIA drivers installed
- CUDA toolkit (optional, for full functionality)

Installation Methods
--------------------

From PyPI
^^^^^^^^^

The simplest way to install RAPIDS CLI is via pip:

.. code-block:: bash

   pip install rapids-cli

From Conda
^^^^^^^^^^

You can also install via conda:

.. code-block:: bash

   conda install -c rapidsai -c conda-forge rapids-cli

From Source
^^^^^^^^^^^

For development or to get the latest features:

.. code-block:: bash

   git clone https://github.com/rapidsai/rapids-cli.git
   cd rapids-cli
   pip install -e .

With Test Dependencies
^^^^^^^^^^^^^^^^^^^^^^

To run tests locally:

.. code-block:: bash

   pip install -e .[test]

Verification
------------

Verify the installation by running:

.. code-block:: bash

   rapids --help

You should see the RAPIDS CLI help message with available commands.

Quick Test
^^^^^^^^^^

Run a quick health check to verify everything is working:

.. code-block:: bash

   rapids doctor --verbose

This will check your GPU availability, CUDA installation, and system configuration.

Upgrading
---------

To upgrade to the latest version:

.. code-block:: bash

   pip install --upgrade rapids-cli

Or with conda:

.. code-block:: bash

   conda update rapids-cli

Uninstalling
------------

To uninstall RAPIDS CLI:

.. code-block:: bash

   pip uninstall rapids-cli

Or with conda:

.. code-block:: bash

   conda remove rapids-cli
