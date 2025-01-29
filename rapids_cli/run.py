# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Run a Python script with RAPIDS Accelerator Modes enabled.

Usage:

rapids run <script.py> <args>
rapids run -m module <args>
"""

import code
import runpy
import sys
import tempfile
from contextlib import contextmanager

from cudf.pandas import install
from cudf.pandas.profiler import Profiler, lines_with_profiling


@contextmanager
def profile_cm(function_profile, line_profile, fn):
    """Context manager for profiling."""
    if fn is None and (line_profile or function_profile):
        raise RuntimeError("Enabling the profiler requires a script name.")
    if line_profile:
        with open(fn) as f:
            lines = f.readlines()

        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".py") as f:
            f.write(lines_with_profiling(lines, function_profile).encode())
            f.seek(0)

            yield f.name
    elif function_profile:
        with Profiler() as profiler:
            yield fn
        profiler.print_per_function_stats()
    else:
        yield fn


def run(module, cmd, profile, line_profile, args):
    """Run a Python script with RAPIDS Accelerator Modes enabled."""
    args = list(args)

    if cmd:
        f = tempfile.NamedTemporaryFile(mode="w+b", suffix=".py")
        f.write(cmd[0].encode())
        f.seek(0)
        args.insert(0, f.name)

    install()

    script_name = args[0] if len(args) > 0 else None
    with profile_cm(profile, line_profile, script_name) as fn:
        if script_name is not None:
            args[0] = fn
        if module:
            (module,) = module
            # run the module passing the remaining arguments
            # as if it were run with python -m <module> <args>
            sys.argv[:] = [module, *args]  # not thread safe?
            runpy.run_module(module, run_name="__main__")
        elif len(args) >= 1:
            # Remove ourself from argv and continue
            sys.argv[:] = args
            runpy.run_path(args[0], run_name="__main__")
        else:
            if sys.stdin.isatty():
                banner = f"Python {sys.version} on {sys.platform}"
                site_import = not sys.flags.no_site
                if site_import:
                    cprt = 'Type "help", "copyright", "credits" or "license" for more information.'
                    banner += "\n" + cprt
            else:
                # Don't show prompts or banners if stdin is not a TTY
                sys.ps1 = ""
                sys.ps2 = ""
                banner = ""

            # Launch an interactive interpreter
            code.interact(banner=banner, exitmsg="")
