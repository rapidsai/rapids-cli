# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Rapids CLI is a command-line interface for RAPIDS."""

import rich_click as click
from rich.console import Console
from rich.traceback import install

from rapids_cli.benchmark import benchmark_run
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
@click.option(
    "--verbose", is_flag=True, help="Enable verbose mode for detailed timing output."
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Perform a dry run without running benchmarks.",
)
@click.option(
    "--runs",
    type=int,
    default=5,
    help="Number of runs to average the benchmark over.",
)
@click.argument("filters", nargs=-1)
def benchmark(verbose, dry_run, runs, filters):
    """Run performance benchmarks comparing CPU vs GPU implementations."""
    status = benchmark_run(verbose, dry_run, runs, filters)
    if not status:
        raise click.ClickException("Some benchmarks failed.")

if __name__ == "__main__":
    rapids()
