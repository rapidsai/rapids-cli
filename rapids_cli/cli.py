# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Rapids CLI is a command-line interface for RAPIDS."""

import rich_click as click
from rich.traceback import install

from rapids_cli.debug import run_debug
from rapids_cli.doctor import doctor_check

install(show_locals=True)


@click.group()
def rapids():
    """The Rapids CLI is a command-line interface for RAPIDS."""
    pass


@rapids.command()
@click.option(
    "--verbose", is_flag=True, help="Enable verbose mode for detailed output."
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Perform a dry run without making any changes.",
)
@click.argument("filters", nargs=-1)
def doctor(verbose, dry_run, filters):
    """Run health checks to ensure RAPIDS is installed correctly."""
    status = doctor_check(verbose, dry_run, filters)
    if not status:
        raise click.ClickException("Health checks failed.")


@rapids.command()
@click.option("--json", is_flag=True, help="Enable JSON mode for detailed output.")
def debug(json):
    """Gather debugging information for RAPIDS."""
    run_debug(output_format="json" if json else "console")


if __name__ == "__main__":
    rapids()
