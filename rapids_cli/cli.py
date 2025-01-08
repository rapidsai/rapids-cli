import rich_click as click
from rich.console import Console
from rich.traceback import install

from rapids_cli.doctor import doctor_check


console = Console()
install(show_locals=True)


@click.group()
def rapids():
    pass


@rapids.command()
@click.option(
    "--verbose", is_flag=True, help="Enable verbose mode for detailed output."
)
@click.argument("arguments", nargs=-1)
def doctor(verbose, arguments):
    """Run health checks to ensure RAPIDS is installed correctly."""
    console.print("Running Doctor Health Checks\n")
    doctor_check(verbose, arguments)


if __name__ == "__main__":
    rapids()
