import yaml
import contextlib
from rich import print
from rapids_cli.doctor.checks.cudf import cudf_checks
from rapids_cli.config import config
from rapids_cli._compatibility import entry_points
from rapids_cli.constants import DOCTOR_SYMBOL


VALID_SUBCOMMANDS = config["valid_subcommands"]["VALID_SUBCOMMANDS"]


def doctor_check(arguments):
    if len(arguments) == 0:
        print(
            f"[bold green] {DOCTOR_SYMBOL} Performing REQUIRED health check for RAPIDS [/bold green] \n"
        )
        checks = []
        print("Discovering checks")
        for ep in entry_points(group="rapids_doctor_check"):
            with contextlib.suppress(AttributeError, ImportError):
                print(f"Found check '{ep.name}' provided by '{ep.value}'")
                checks += [ep.load()]
        print("Running checks")
        for check_fn in checks:
            check_fn()
    else:
        for argument in arguments:
            if argument not in VALID_SUBCOMMANDS:
                print(
                    f"Not a valid subcommand - please use one of the following: {str(VALID_SUBCOMMANDS)}"
                )
            if argument == "cudf":
                with open("config.yml", "r") as file:
                    config = yaml.safe_load(file)
                cuda_requirement = config["cudf_requirements"]["cuda_requirement"]
                driver_requirement = config["cudf_requirements"]["driver_requirement"]
                compute_requirement = config["cudf_requirements"]["compute_requirement"]

                cudf_checks(cuda_requirement, driver_requirement, compute_requirement)
