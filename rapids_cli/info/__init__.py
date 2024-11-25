from rich import print
from rapids_cli.config import config


def info_check(arguments):
    if not arguments:
        return
    if len(arguments) > 1:
        print("Please only enter one subcommand for rapids info.")
        return
    name = arguments[0]
    print(f"[bold green]{name} [/bold green]")
    description = config[name]["description"]
    print(f"{description}\n \nHere are some helpful links to get started:")
    for link in config[name]["links"]:
        print(f"{link} \n")
