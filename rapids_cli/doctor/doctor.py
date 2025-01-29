"""Health check for RAPIDS."""

import contextlib
import warnings
from dataclasses import dataclass
from typing import Optional

from rich.console import Console

from rapids_cli._compatibility import entry_points
from rapids_cli.constants import DOCTOR_SYMBOL

console = Console()


@dataclass
class CheckResult:
    name: str
    description: str
    status: bool
    value: str = None
    error: Exception = None
    warnings: list[Warning] = None


def doctor_check(verbose: bool, filters: Optional[list[str]] = None) -> bool:
    """Perform a health check for RAPIDS.

    This function runs a series of checks based on the provided arguments.
    If no arguments are provided, it executes all available health checks.
    If specific subcommands are given, it validates them against valid
    subcommands and executes corresponding checks.

    Parameters:
    ----------
    filters : list (optional)
        A list of filters to run specific checks.

    Raises:
    -------
    ValueError:
        If an invalid subcommand is provided.

    Notes:
    -----
    The function discovers and loads check functions defined in entry points
    under the 'rapids_doctor_check' group. It also checks specific
    configurations related to a corresponding subcommand if given.

    Example:
    --------
    > doctor_check([])  # Run all health checks
    > doctor_check(['cudf'])  # Run 'cudf' specific checks
    """
    filters = [] if not filters else filters
    console.print(
        f"[bold green]{DOCTOR_SYMBOL} Performing REQUIRED health check for RAPIDS [/bold green]"
    )

    checks = []
    if verbose:
        console.print("Discovering checks")
    for ep in entry_points(group="rapids_doctor_check"):
        with contextlib.suppress(AttributeError, ImportError):
            if verbose:
                console.print(f"Found check '{ep.name}' provided by '{ep.value}'")
            if filters and not any(f in ep.value for f in filters):
                continue
            checks += [ep.load()]
    if verbose:
        console.print(f"Discovered {len(checks)} checks")
        console.print("Running checks")

    results: list[CheckResult] = []
    with console.status("[bold green]Running checks...") as ui_status:
        for i, check_fn in enumerate(checks):
            error = None
            caught_warnings = None
            ui_status.update(f"Running [{i}/{len(checks)}] {check_fn.__name__}")

            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    status = True
                    check_fn(verbose=verbose)
                    caught_warnings = w

            except Exception as e:
                error = e
                status = False

            results.append(
                CheckResult(
                    name=check_fn.__name__,
                    description=check_fn.__doc__.strip().split("\n")[0],
                    status=bool(status),
                    value=status if isinstance(status, str) else None,
                    error=error,
                    warnings=caught_warnings,
                )
            )

    # Print warnings
    for result in results:
        if result.warnings:
            for warning in result.warnings:
                console.print(f"[bold yellow]Warning[/bold yellow]: {warning.message}")

    if all(result.status for result in results):
        console.print("[bold green]All checks passed![/bold green]")
        return True
    else:
        for result in results:
            if not result.status:
                console.print(f"[bold red]{result.name} failed[/bold red]")
                console.print(f"  {result.error}")
                if verbose and result.error:
                    try:
                        raise result.error
                    except Exception:
                        console.print_exception()
        return False
