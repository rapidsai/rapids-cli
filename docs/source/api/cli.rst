.. SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

CLI Module
==========

The CLI module provides the main command-line interface for RAPIDS CLI using Click.

.. automodule:: rapids_cli.cli
   :members:
   :undoc-members:
   :show-inheritance:

Main Commands
-------------

rapids
^^^^^^

.. autofunction:: rapids_cli.cli.rapids

The main CLI entry point. Provides access to all subcommands.

doctor
^^^^^^

.. autofunction:: rapids_cli.cli.doctor

Run health checks to verify RAPIDS installation.

**Options:**

- ``--verbose``: Enable detailed output
- ``--dry-run``: Show which checks would run without executing them
- ``filters``: Optional filters to run specific checks

**Exit Codes:**

- 0: All checks passed
- 1: One or more checks failed

debug
^^^^^

.. autofunction:: rapids_cli.cli.debug

Gather comprehensive debugging information.

**Options:**

- ``--json``: Output in JSON format for machine parsing

**Output:**

Returns detailed system information including:

- Platform and OS details
- GPU and driver information
- CUDA version
- Python configuration
- Installed packages
- Available tools
