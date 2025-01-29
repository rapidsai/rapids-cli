"""The Rapids CLI is a command-line interface for RAPIDS."""

import rich_click as click
from rich.console import Console
from rich.traceback import install

import rapids_cli.run
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


@rapids.command(help="Run a Python script with RAPIDS Accelerator Modes enabled.")
@click.option("-m", "module")
@click.option("-c", "cmd")
@click.option(
    "--profile",
    is_flag=True,
    help="Perform per-function profiling of this script.",
)
@click.option(
    "--line-profile",
    is_flag=True,
    help="Perform per-line profiling of this script.",
)
@click.argument("args", nargs=-1)
def run(module, cmd, profile, line_profile, args):
    """Run a Python script with RAPIDS Accelerator Modes enabled."""
    rapids_cli.run.run(module, cmd, profile, line_profile, args)


if __name__ == "__main__":
    rapids()
