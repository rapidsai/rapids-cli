import click
import pynvml
import platform
import subprocess
import json
import yaml
import psutil
from rich import print 
from rich.console import Console 
from rich.table import Table
import platform

from rapids_cli.dependency_parser import dependency_parser
from rapids_cli.default_checks import default_checks
from rapids_cli.cudf_checks import cudf_checks

CHECK_SYMBOL = "🚨"
OK_MARK = "✅"
X_MARK = "❌"
DOCTOR_SYMBOL = "🧑‍⚕️"
VALID_SUBCOMMANDS = ["cudf"]


@click.group()
def rapids(): 
    #main CLI for RAPIDS 
    print("[bold] Welcome to the RAPIDS CLI [/bold]\n")

    print("[italic green] NVIDIA RAPIDS is a suite of open-source software libraries and APIs designed to accelerate data science and ML workflows on GPUs. It leverages the power of NVIDIA GPUs to enable high-performance computing, allowing users to process large datasets much faster than traditional CPU-based methods. [/italic green] \n")
    print("[italic] For more information on RAPIDS installation, please visit [purple] https://docs.rapids.ai/install?_gl=1*o8b62b*_ga*MTU3MTEzNzgxNC4xNzI0OTc1MzQ1*_ga_RKXFW6CM42*MTcyNjg2NDcyNC42LjEuMTcyNjg2NTIzNS40OC4wLjA [/purple].[/italic] \n")
    print("RAPIDS Installation System Requirements \n")
    
    table = Table(title = "[bold] System Requirements [/bold]")

    table.add_column("-", style = "cyan")
    table.add_column("Requirement", style = "magenta")

    table.add_row("GPU", "NVIDIA Volta™ or higher with compute capability 7.0+", style = "red")
    table.add_row("OS", "Ubuntu 20.04/22.04 or Rocky Linux 8 with gcc/++ 9.0+ \nWindows 11 using a WSL2 specific install \nRHEL 7/8 support is provided through Rocky Linux 8 builds/installs", style = "dark_orange")
    table.add_row("CUDA & Nvidia Drivers", "CUDA 11.2 with Driver 470.42.01 or newer \nCUDA 11.4 with Driver 470.42.01 or newer \nCUDA 11.5 with Driver 495.29.05 or newer\nCUDA 11.8 with Driver 520.61.05 or newer \nCUDA 12.0 with Driver 525.60.13 or newer see CUDA 12 section below for notes on usage \nCUDA 12.2 with Driver 535.86.10 or newer", style = "yellow")
 
    console = Console()
    console.print(table)
    print("\n")

    print("RAPIDS Installation System [italic] Recommendations [/italic] \n")
    table = Table(title = "[bold] System Recommendations [/bold]")

    table.add_column("-", style = "cyan")
    table.add_column("Recommendation", style = "magenta")


    table.add_row("SSD Drive", "NVMe preferred", style = "green")
    table.add_row("System to GPU Memory Ratio", "Approximately 2:1 ratio", style = "cyan1")
    table.add_row("GPU Linkage", "NVLink with 2 or more GPUs", style = "purple")

    console = Console()
    console.print(table)
    print("\n")

 


@rapids.command()
def help():
    """Display help information for RAPIDS CLI."""
    click.echo("RAPIDS CLI Help")
    click.echo("Available commands:")
    click.echo("  rapid       - Run the main RAPIDS command")
    click.echo("  help        - Display this help message")
    click.echo("  info       - Display this help message")

    table = Table(title = "RAPIDS subcommands")

    table.add_column("Subcommand", style = "cyan")
    table.add_column("Description", style = "magenta")

    table.add_row("doctor", "checks that all system and hardware requirements are met")
    table.add_row("help", "instructions on how to use RAPIDS")

    console = Console()
    console.print(table)

    

@rapids.command()
@click.argument('arguments', nargs=-1)
def doctor(arguments):
    click.echo("checking environment")
    print("\n")

    if len(arguments) == 0:
        default_checks()
    else:
        for argument in arguments: 
            if argument not in VALID_SUBCOMMANDS: 
                print(f"Not a valid subcommand - please use one of the following: {str(VALID_SUBCOMMANDS)}")
            if argument == "cudf":
                with open('config.yml', 'r') as file: 
                    config = yaml.safe_load(file)
                cuda_requirement = config['cudf_requirements']['cuda_requirement']
                driver_requirement = config['cudf_requirements']['driver_requirement']
                compute_requirement = config['cudf_requirements']['compute_requirement']
                dependicies = config['cudf_dependencies']
                cudf_checks(cuda_requirement,driver_requirement, compute_requirement, dependicies)
                
    

@rapids.command()
def info():
    click.echo("Information about RAPIDS subcommands \n")
    


    table = Table(title = "[bold] doctor [/bold]")

    table.add_column("function", style = "cyan")
    table.add_column("description", style = "magenta")

    table.add_row("check_gpu_compute_capability()", "checks GPU compute capability", style = "red")
    table.add_row("check_os_compatibility()", "checks OS version compatibility", style = "dark_orange")
    table.add_row("check_driver_compatibility()", "checks Driver & CUDA compatibility", style = "yellow")
    table.add_row("check_sdd_nvme()", "detects if NVMe SSDs exist (recommended)", style = "green")
    table.add_row("check_memory_to_gpu_ratio()", "checks if System Memory to GPU Memory ratio is approximately 2:1 ratio (recommended)", style = "cyan1")
    table.add_row("check_nvlink_status()", "checks if NVLink with 2 or more GPUs exist (recommended)", style = "purple")

    console = Console()
    console.print(table)
    print("\n")

if __name__ == '__main__':
    rapids()


