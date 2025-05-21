# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Rapids CLI is a command-line interface for RAPIDS."""

import rich_click as click
from rich.console import Console
from rich.traceback import install

from rapids_cli.doctor import doctor_check

console = Console()
install(show_locals=True)


@click.group()
def rapids():
    """The Rapids CLI is a command-line interface for RAPIDS."""
    pass


@rapids.command()
@click.option(
    "--verbose", is_flag=True, help="Enable verbose mode for detailed output."
)
@click.argument("filters", nargs=-1)
def doctor(verbose, filters):
    """Run health checks to ensure RAPIDS is installed correctly."""
    status = doctor_check(verbose, filters)
    if not status:
        raise click.ClickException("Health checks failed.")


if __name__ == "__main__":
    rapids()
