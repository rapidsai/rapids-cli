import subprocess
import json
import platform

from rapids_cli.doctor.checks.os import detect_os
from rapids_cli.doctor.checks.cuda_driver import get_cuda_version
from rapids_cli.constants import CHECK_SYMBOL, OK_MARK, X_MARK, DOCTOR_SYMBOL
from rapids_cli.config import config


def check_conda():
    conda_requirement = config['min_supported_versions']['conda_requirement']
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
    if not system_cuda_version: 
        return 
    print(f"      System CUDA Tookit Version: {system_cuda_version}")
    result = subprocess.check_output(["pip", "show", "cuda-python"], stderr=subprocess.DEVNULL)
    pip_cuda_version = result.decode('utf-8').strip().split("\n")[1].split(" ")[-1]
    print(f"      pip CUDA Version: {pip_cuda_version}")
    system_cuda_version_major, pip_cuda_version_major = system_cuda_version.split(".")[0], pip_cuda_version.split(".")[0]
    if system_cuda_version_major == pip_cuda_version_major:
        print(f"      {OK_MARK: >6} System and pip CUDA Versions are compatible with each other")
    elif system_cuda_version_major >  pip_cuda_version_major:
        print(f"      {X_MARK: >6} Please upgrade pip CUDA version to {system_cuda_version_major}")
    else:
        print(f"      {X_MARK: >6} Please upgrade system CUDA version to {pip_cuda_version_major}")


def check_docker():
    docker_requirement = config['min_supported_versions']['docker_requirement']
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




def check_glb():
    if detect_os() != "Ubuntu":
        return True
    
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
