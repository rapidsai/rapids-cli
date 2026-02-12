.. SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

RAPIDS CLI Documentation
========================

The RAPIDS CLI is a command-line tool for performing common RAPIDS operations,
primarily focused on health checks (``rapids doctor``) and debugging (``rapids debug``).
It uses a plugin system that allows RAPIDS libraries to register their own health checks
via Python entry points.

Quick Start
-----------

.. code-block:: bash

   pip install rapids-cli

   # Run health checks
   rapids doctor

   # Gather system info for debugging
   rapids debug --json

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   plugin_development

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
