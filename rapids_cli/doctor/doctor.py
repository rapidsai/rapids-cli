"""Health check for RAPIDS."""

import contextlib

from rich import print

from rapids_cli._compatibility import entry_points
from rapids_cli.config import config
from rapids_cli.constants import DOCTOR_SYMBOL
from rapids_cli.doctor.checks.cudf import cudf_checks

VALID_SUBCOMMANDS = config["valid_subcommands"]["VALID_SUBCOMMANDS"]


def doctor_check(verbose, arguments):
    """Perform a health check for RAPIDS.

    This function runs a series of checks based on the provided arguments.
    If no arguments are provided, it executes all available health checks.
    If specific subcommands are given, it validates them against valid
    subcommands and executes corresponding checks.

    Parameters:
    ----------
    arguments : list
        A list of subcommands for specific checks. If empty, runs all checks.

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
    if len(arguments) == 0:
        print(
            f"[bold green] {DOCTOR_SYMBOL} Performing REQUIRED health check for RAPIDS [/bold green] \n"
        )

        checks = []
        if verbose:
            print("Discovering checks")
        for ep in entry_points(group="rapids_doctor_check"):
            with contextlib.suppress(AttributeError, ImportError):
                if verbose:
                    print(f"Found check '{ep.name}' provided by '{ep.value}'")
                checks += [ep.load()]
        if verbose:
            print("Running checks")
        for check_fn in checks:
            check_fn(verbose)

    else:
        for argument in arguments:
            if argument not in VALID_SUBCOMMANDS:
                print(
                    f"Not a valid subcommand - please use one of the following: {str(VALID_SUBCOMMANDS)}"
                )
            if argument == "cudf":

                cuda_requirement = config["cudf_requirements"]["cuda_requirement"]
                driver_requirement = config["cudf_requirements"]["driver_requirement"]
                compute_requirement = config["cudf_requirements"]["compute_requirement"]

                cudf_checks(
                    cuda_requirement,
                    driver_requirement,
                    compute_requirement,
                    verbose,
                )
