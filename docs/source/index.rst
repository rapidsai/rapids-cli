.. SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

RAPIDS CLI Documentation
========================

The RAPIDS CLI is a command-line tool for performing common RAPIDS operations,
primarily focused on health checks (``rapids doctor``) and debugging (``rapids debug``).
It uses a plugin system that allows RAPIDS libraries to register their own health checks
via Python entry points.

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-Apache%202.0-blue.svg
   :target: https://github.com/rapidsai/rapids-cli/blob/main/LICENSE
   :alt: License

Quick Start
-----------

Install the RAPIDS CLI:

.. code-block:: bash

   pip install rapids-cli

Run health checks:

.. code-block:: bash

   rapids doctor

Gather debugging information:

.. code-block:: bash

   rapids debug --json

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   user_guide
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   plugin_development
   contributing

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/cli
   api/doctor
   api/debug
   api/checks

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
