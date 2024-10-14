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

from rapids_cli.os_checks import detect_os
from rapids_cli.gpu_checks.gpu_memory_checks import check_memory_to_gpu_ratio
from rapids_cli.gpu_checks.nvlink import check_nvlink_status
from rapids_cli.gpu_checks import gpu_check, check_gpu_compute_capability
from rapids_cli.cuda_driver_checks import get_cuda_version, cuda_check, get_driver_version, check_driver_compatibility
from rapids_cli.cuda_driver_checks.sdd_nvme import check_sdd_nvme

CHECK_SYMBOL = "🚨"
OK_MARK = "✅"
X_MARK = "❌"
DOCTOR_SYMBOL = "🧑‍⚕️"
VALID_SUBCOMMANDS = ["cudf"]


def compare_version(version, requirement):
    if str(version) >= str(requirement): 
        return True
    return False 


def check_docker(docker_requirement):
    print(f"   {CHECK_SYMBOL} Checking for [italic red]Docker Version[/italic red]")
    result = str(subprocess.check_output(["docker", "--version"]))
    
    version_num = result.split(",")[0].split(" ")[-1]
    if version_num >= docker_requirement:
        print(f"      {OK_MARK: >6} DOCKER Version is compatible with RAPIDS")
    else:
        print(f"      {X_MARK: >6} DOCKER Version is not compatible with RAPIDS - please upgrade to Docker {docker_requirement}")
    
    try:
        docker_version_data = json.loads(subprocess.check_output(["docker", "version", "-f", "json"]))
        print(docker_version_data)
    except:
        print(f"      {X_MARK: >6} NVIDIA Docker Runtime not available - please install here : https://github.com/NVIDIA/nvidia-container-toolkit")



def check_conda(conda_requirement):
    print(f"   {CHECK_SYMBOL} Checking for [italic red]Conda Version[/italic red]")
    result =  subprocess.check_output(["conda", "info", "--json"], stderr=subprocess.DEVNULL)
    result_json = json.loads(result.decode('utf-8'))
    version_num = result_json["conda_version"]

    
    if version_num >= conda_requirement:
        print(f"      {OK_MARK: >6} CONDA Version is compatible with RAPIDS")
    else:
        print(f"      {X_MARK: >6} CONDA Version is not compatible with RAPIDS - please upgrade to Docker {conda_requirement}")
    

def check_pip():
    print(f"   {CHECK_SYMBOL} Checking for [italic red]Pip Requirements[/italic red]")
    system_cuda_version = get_cuda_version()
    print(f"      System CUDA Tookit Version: {system_cuda_version}")
    result = subprocess.check_output(["pip", "show", "cuda-python"], stderr=subprocess.DEVNULL)
    pip_cuda_version = result.decode('utf-8').strip().split("\n")[1].split(" ")[-1]
    print(f"      pip CUDA  Version: {pip_cuda_version}")
    system_cuda_version_major, pip_cuda_version_major = system_cuda_version.split(".")[0], pip_cuda_version.split(".")[0]
    if system_cuda_version_major == pip_cuda_version_major:
        print(f"      {OK_MARK: >6} System and pip CUDA Versions are compatible with each other")
    elif system_cuda_version_major >  pip_cuda_version_major:
        print(f"      {X_MARK: >6} Please upgrade pip CUDA version to {system_cuda_version_major}")
    else:
        print(f"      {X_MARK: >6} Please upgrade system CUDA version to {pip_cuda_version_major}")



def check_glb():
    print(f"   {CHECK_SYMBOL} Checking for [italic red]glb comp[/italic red]")
    glb_compatible = False
    result = subprocess.check_output(["ldd", "--version"])
    glb_version = result.decode('utf-8').strip().split("\n")[0].split(" ")[-1]
    
    machine = platform.machine()

    if machine == 'x86_64':
        if glb_version >= "2.17":
            glb_compatible = True 
    elif machine == 'aarch64' or machine == "arm64":
        if glb_version >= "2.32":
            glb_compatible =  True 
    else: 
        print(f"      {X_MARK: >6} Please only use x86_64 or arm64 architectures")
    if glb_compatible:
        print(f"      {OK_MARK: >6} GLB version and CPU architecture are compatible with each other. ")
    else:
        print(f"      {X_MARK: >6} GLB version and CPU architecture are NOT compatible with each other.")
        
        if machine == 'x86_64':
            print(f"      Please upgrade glb to 2.17 and above")
        elif machine == 'aarch64' or machine == "arm64":
            print(f"      Please upgrade glb to 2.32 and above")
        


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


def cudf_checks(cuda_requirement, driver_requirement, compute_requirement):

    print(f"[bold green] {DOCTOR_SYMBOL} Performing REQUIRED health check for CUDF [/bold green] \n")
    
    
    print(f"   {CHECK_SYMBOL} Checking for [italic red]CUDA dependencies[/italic red]")
    if compare_version(get_cuda_version(), cuda_requirement): #when the other branch gets merged, will move the magic numbers to their yaml file 
        print(f"{OK_MARK: >6}  CUDA version compatible with CUDF")
    else:
        print(f"{X_MARK: >6}  CUDA version not compatible with CUDF. Please upgrade to {cuda_requirement}")
    

    print(f"   {CHECK_SYMBOL} Checking for [italic red]Driver Availability[/italic red]")
    if cuda_check():
        if compare_version(get_driver_version(), driver_requirement):
            print(f"{OK_MARK: >6}  Nvidia Driver version compatible with CUDF")
        else:
            print(f"{X_MARK: >6}  Nvidia Driver version not compatible with CUDF. Please upgrade to {driver_requirement}")
    else: 
        print(f"{X_MARK: >6} No Nvidia Driver Detected")

    if gpu_check():
        if check_gpu_compute_capability(compute_requirement):
            print(f"{OK_MARK: >6}  GPU compute compatible with CUDF")
        else:
            print(f"{X_MARK: >6}  GPU compute not compatible with CUDF. Please upgrade to compute >={compute_requirement}") 
   
    
def default_checks(): 
    print(f"[bold green] {DOCTOR_SYMBOL} Performing REQUIRED health check for RAPIDS [/bold green] \n")


    
    with open('config.yml', 'r') as file: 
        config = yaml.safe_load(file)
    

    gpu_compute_requirement = config['min_supported_versions']['gpu_compute_requirement']
    docker_requirement = config['min_supported_versions']['docker_requirement']
    conda_requirement = config['min_supported_versions']['conda_requirement']


    gpu_check_return = gpu_check()
    cuda_check_return = cuda_check()
    if gpu_check_return:
        check_gpu_compute_capability(gpu_compute_requirement)
    if cuda_check_return:
        check_driver_compatibility()
    os = detect_os()

    print("\n")
    print(f"[bold green]{DOCTOR_SYMBOL} Performing RECOMMENDED health check for RAPIDS[/bold green] \n")
    check_sdd_nvme()
    if gpu_check_return:
        check_memory_to_gpu_ratio()
        check_nvlink_status()

    print("\n")
    print(f"[bold green]{DOCTOR_SYMBOL} Performing OTHER health checks for RAPIDS[/bold green] \n")
    check_docker(docker_requirement)
    check_conda(conda_requirement)

    if cuda_check_return:
        check_pip()
    
    if os == 'Ubuntu':
        check_glb()


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

                cudf_checks(cuda_requirement,driver_requirement, compute_requirement)
                
    

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