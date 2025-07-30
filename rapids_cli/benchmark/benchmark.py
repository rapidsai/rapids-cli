# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark suite for RAPIDS CPU vs GPU performance comparison."""

import contextlib
import statistics
import warnings
from dataclasses import dataclass

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from rapids_cli._compatibility import entry_points
from rapids_cli.constants import BENCHMARK_SYMBOL, SPEEDUP_SYMBOL

console = Console()


@dataclass
class BenchmarkResult:
    name: str
    description: str
    status: bool
    cpu_time: float | None
    gpu_time: float | None
    cpu_std: float | None
    gpu_std: float | None
    speedup: float | None
    error: Exception | None
    warnings: list[warnings.WarningMessage] | None

    @property
    def speedup_display(self) -> str:
        """Format speedup for display."""
        if self.speedup is None:
            return "N/A"
        if self.speedup < 1:
            return f"{1/self.speedup:.1f}x slower"
        return f"{self.speedup:.1f}x faster"

    @property
    def cpu_time_display(self) -> str:
        """Format CPU time with standard deviation."""
        if self.cpu_time is None:
            return "N/A"
        if self.cpu_std is not None and self.cpu_std > 0:
            return f"{self.cpu_time:.3f}s ± {self.cpu_std:.3f}s"
        return f"{self.cpu_time:.3f}s"

    @property
    def gpu_time_display(self) -> str:
        """Format GPU time with standard deviation."""
        if self.gpu_time is None:
            return "N/A"
        if self.gpu_std is not None and self.gpu_std > 0:
            return f"{self.gpu_time:.3f}s ± {self.gpu_std:.3f}s"
        return f"{self.gpu_time:.3f}s"


def benchmark_run(
    verbose: bool, dry_run: bool, runs: int = 5, filters: list[str] | None = None
) -> bool:
    """Run performance benchmarks comparing CPU vs GPU implementations.

    This function runs a series of benchmarks based on the provided arguments.
    If no arguments are provided, it executes all available benchmarks.
    If specific filters are given, it validates them and executes corresponding benchmarks.

    Parameters:
    ----------
    verbose : bool
        Enable verbose output with detailed timing information.
    dry_run : bool
        Perform a dry run without executing benchmarks.
    runs : int
        Number of iterations to run each benchmark for averaging.
    filters : list, optional
        A list of filters to run specific benchmarks.

    Returns:
    -------
    bool
        True if all benchmarks completed successfully, False otherwise.

    Notes:
    -----
    The function discovers and loads benchmark functions defined in entry points
    under the 'rapids_benchmark_check' group. Each benchmark function should
    return a tuple of (cpu_time, gpu_time) in seconds.

    Example:
    --------
    > benchmark_run(verbose=True, dry_run=False, filters=[])  # Run all benchmarks
    > benchmark_run(verbose=False, dry_run=False, filters=['cudf'])  # Run cuDF benchmarks
    """
    filters = [] if not filters else filters
    console.print(
        f"[bold green]{BENCHMARK_SYMBOL} Running RAPIDS benchmarks [/bold green]"
    )

    benchmarks = []
    if verbose:
        console.print("Discovering benchmarks")

    for ep in entry_points(group="rapids_benchmark"):
        with contextlib.suppress(AttributeError, ImportError):
            if verbose:
                console.print(f"Found benchmark '{ep.name}' provided by '{ep.value}'")
            if filters and not any(f in ep.value for f in filters):
                continue
            benchmarks += [ep.load()]

    if verbose:
        console.print(f"Discovered {len(benchmarks)} benchmarks")

    if not dry_run:
        console.print(f"Running benchmarks ({runs} runs each)")
    else:
        console.print("Dry run, skipping benchmarks")
        return True

    if not benchmarks:
        console.print(
            "[yellow]No benchmarks found. Install RAPIDS libraries to enable benchmarks.[/yellow]"
        )
        return True

    results: list[BenchmarkResult] = []

    with Progress(
        TextColumn("[bold blue]{task.fields[benchmark_name]}"),
        BarColumn(bar_width=40),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        expand=False,
    ) as progress:

        for i, benchmark_fn in enumerate(benchmarks):
            error = None
            caught_warnings = None
            all_cpu_times = []
            all_gpu_times = []

            task_id = progress.add_task(
                f"[{i+1}/{len(benchmarks)}]",
                total=runs,
                benchmark_name=f"[{i+1}/{len(benchmarks)}] {benchmark_fn.__name__}",
                completed=0,
            )

            try:
                for run in range(runs):
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")

                        result = benchmark_fn(verbose=verbose)
                        if isinstance(result, tuple) and len(result) == 2:
                            cpu_time, gpu_time = result
                            if cpu_time and gpu_time and cpu_time > 0 and gpu_time > 0:
                                all_cpu_times.append(cpu_time)
                                all_gpu_times.append(gpu_time)

                        if run == 0:
                            caught_warnings = w

                    progress.update(task_id, completed=run + 1)

                if all_cpu_times and all_gpu_times:
                    avg_cpu_time = sum(all_cpu_times) / len(all_cpu_times)
                    avg_gpu_time = sum(all_gpu_times) / len(all_gpu_times)
                    speedup = avg_cpu_time / avg_gpu_time

                    # Calculate standard deviations (only if we have multiple runs)
                    cpu_std = (
                        statistics.stdev(all_cpu_times)
                        if len(all_cpu_times) > 1
                        else 0.0
                    )
                    gpu_std = (
                        statistics.stdev(all_gpu_times)
                        if len(all_gpu_times) > 1
                        else 0.0
                    )

                    status = True

                    # Remove progress task and show completion summary with variance
                    progress.remove_task(task_id)
                    console.print(
                        f"[green]✓[/green] [{i+1}/{len(benchmarks)}] {benchmark_fn.__name__}"
                    )

                    # Show timing details with standard deviation
                    if runs > 1:
                        console.print(
                            f"  CPU Time: [red]{avg_cpu_time:.3f}s ± {cpu_std:.3f}s[/red]  "
                            f"GPU Time: [green]{avg_gpu_time:.3f}s ± {gpu_std:.3f}s[/green]  "
                            f"Speedup: [bold green]{speedup:.1f}x[/bold green]"
                        )
                    else:
                        console.print(
                            f"  CPU Time: [red]{avg_cpu_time:.3f}s[/red]  "
                            f"GPU Time: [green]{avg_gpu_time:.3f}s[/green]  "
                            f"Speedup: [bold green]{speedup:.1f}x[/bold green]"
                        )

                else:
                    avg_cpu_time = None
                    avg_gpu_time = None
                    cpu_std = None
                    gpu_std = None
                    speedup = None
                    status = False

                    # Remove progress and show failure
                    progress.remove_task(task_id)
                    console.print(
                        f"[red]❌[/red] [{i+1}/{len(benchmarks)}] {benchmark_fn.__name__} - "
                        f"[bold red]Failed[/bold red]"
                    )

            except Exception as e:
                error = e
                status = False
                avg_cpu_time = None
                avg_gpu_time = None
                cpu_std = None
                gpu_std = None
                speedup = None

                # Remove progress and show failure
                progress.remove_task(task_id)
                console.print(
                    f"[red]❌[/red] [{i+1}/{len(benchmarks)}] {benchmark_fn.__name__} - "
                    f"[bold red]Error: {str(e)}[/bold red]"
                )

            results.append(
                BenchmarkResult(
                    name=benchmark_fn.__name__,
                    description=(
                        benchmark_fn.__doc__.strip().split("\n")[0]
                        if benchmark_fn.__doc__
                        else "No description"
                    ),
                    status=status,
                    cpu_time=avg_cpu_time,
                    gpu_time=avg_gpu_time,
                    cpu_std=cpu_std,
                    gpu_std=gpu_std,
                    speedup=speedup,
                    error=error,
                    warnings=caught_warnings,
                )
            )

    # Print warnings
    for result in results:
        if result.warnings:
            for warning in result.warnings:
                console.print(f"[bold yellow]Warning[/bold yellow]: {warning.message}")

    # Display results in a table
    if any(result.status for result in results):
        _display_benchmark_results(results, verbose)

    # Check for failures
    failed_benchmarks = [result for result in results if not result.status]
    if failed_benchmarks:
        console.print("\n[bold red]Failed benchmarks:[/bold red]")
        for result in failed_benchmarks:
            console.print(f"  [red]❌ {result.name}[/red]: {result.error}")
            if verbose and result.error:
                try:
                    raise result.error
                except Exception:
                    console.print_exception()
        return False

    return True


def _display_benchmark_results(results: list[BenchmarkResult], verbose: bool) -> None:
    """Display benchmark results in a formatted table."""
    successful_results = [r for r in results if r.status and r.speedup is not None]

    if not successful_results:
        console.print("[yellow]No successful benchmarks to display.[/yellow]")
        return

    table = Table(
        title=f"{SPEEDUP_SYMBOL} CPU vs GPU Performance Comparison", show_header=True
    )
    table.add_column("Benchmark", style="cyan", width=20)
    table.add_column("Description", style="white", width=30)
    table.add_column("CPU Time (mean ± σ)", style="red", justify="right")
    table.add_column("GPU Time (mean ± σ)", style="green", justify="right")
    table.add_column("Speedup", style="bold magenta", justify="right")

    for result in successful_results:
        cpu_time_str = result.cpu_time_display
        gpu_time_str = result.gpu_time_display
        speedup_str = result.speedup_display

        if result.speedup and result.speedup > 1:
            speedup_style = "bold green"
        elif result.speedup and result.speedup < 1:
            speedup_style = "bold red"
        else:
            speedup_style = "yellow"

        table.add_row(
            result.name,
            result.description,
            cpu_time_str,
            gpu_time_str,
            f"[{speedup_style}]{speedup_str}[/{speedup_style}]",
        )

    console.print()
    console.print(table)
