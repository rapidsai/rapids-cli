.. SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Debug Module
============

The ``rapids_cli.debug.debug`` module gathers system and environment information
for troubleshooting RAPIDS installations.

:func:`~rapids_cli.debug.debug.run_debug` is the main entry point. It collects:

- Platform and OS details (from ``platform`` and ``/etc/os-release``)
- NVIDIA driver and CUDA versions (via ``pynvml``)
- CUDA runtime path (via ``cuda-pathfinder``)
- System CUDA toolkit locations (globbing ``/usr/local/cuda*``)
- Python version and hash info
- All installed package versions
- pip freeze and conda list output
- Tool versions: pip, conda, uv, pixi, g++, cmake, nvcc

Output is either a Rich-formatted console table or JSON (``--json``).

API
---

.. automodule:: rapids_cli.debug.debug
   :members:
   :undoc-members:
   :show-inheritance:
