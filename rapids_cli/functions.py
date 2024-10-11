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

CHECK_SYMBOL = "🚨"
OK_MARK = "✅"
X_MARK = "❌"
DOCTOR_SYMBOL = "🧑‍⚕️"
VALID_SUBCOMMANDS = ["cudf"]


def gpu_check():
    print(f"   {CHECK_SYMBOL} Checking for [italic red]GPU Availability[/italic red]")
    try: 
        pynvml.nvmlInit()
        try: 
            num_gpus = pynvml.nvmlDeviceGetCount()
            print(f"      {OK_MARK: >6} Number of GPUs detected: {num_gpus}")
            return True
        except:
            print(f"      {X_MARK: >6} GPU detected but not available")
            return False 
        
        pynvml.nvmlShutdown()
    except: 
        print(f"      {X_MARK: >6} No available GPUs detected")
        return False


def cuda_check():
    try: 
        pynvml.nvmlInit()
        print(f"   {CHECK_SYMBOL} Checking for [italic red]CUDA Availability[/italic red]")
        try: 
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
            print(f"      {OK_MARK: >6} CUDA detected")
            print(f'           CUDA VERSION:{cuda_version//1000}.{cuda_version % 1000}')
            return True 
        except: 
            print(f"      {X_MARK: >6} No CUDA is available")
            return False 
        pynvml.nvmlShutdown()
    except:
        return False
    
    
def check_gpu_compute_capability(required_capability):
    # Initialize pynvml
    print(f"   {CHECK_SYMBOL} Checking for [italic red]GPU Compute Capability[/italic red]")
    try: 
        pynvml.nvmlInit()
        meets_requirement = False 
        num_gpus = pynvml.nvmlDeviceGetCount()
        for i in range(num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            compute_capability = major * 10 + minor
            
            print(f"      GPU {i} Compute Capability: {major}.{minor}")
            
            if compute_capability >= int(required_capability):
                meets_requirement = True
                print(f"         GPU {i} meets the required compute capability {required_capability[0]}.{required_capability[1]}")
            else:
                print(f"         GPU {i} does not meet the required compute capability {required_capability[0]}.{required_capability[1]}.")
    
        pynvml.nvmlShutdown()
    except: 
        print(f"       {X_MARK: >6} No GPU - cannot determineg GPU Compute Capability")
    
    return meets_requirement


def compare_version(version, requirement):
    if str(version) >= str(requirement): 
        return True
    return False 
VALID_SUBCOMMANDS = ["cudf"]



VALID_LINUX_OS_VERSIONS = ["Ubuntu 20.04", "Ubuntu 22.04", "Rocky Linux 8.7"] 

def check_os_version(os_attributes):
    os_name = os_attributes['NAME'] + " " + os_attributes['VERSION_ID']
    print(f"Current OS Version: {os_name}")
    return os_name in VALID_LINUX_OS_VERSIONS 

    
def get_os_attributes(os_release): 
    os_attributes = {}
    for attribute in os_release.split("\n"):
        if len(attribute) < 2: continue 
        #print(attribute.split("="))
        key, value = attribute.split("=")[0], attribute.split("=")[1]
        os_attributes[key] = value[1:-1]
    
    return os_attributes

def get_linux_os_version():
    try: 
        with open('/etc/os-release') as f: 
            os_release = f.read()
            os_attributes =get_os_attributes(os_release)        
        return os_attributes
    except FileNotFoundError:
        return "OS release file not found."
        
def detect_os():
    print(f"   {CHECK_SYMBOL} Checking for [italic red]OS Capability[/italic red]")
    system = platform.system()
    release = platform.release()
    version = platform.version()
    os = ""

    print(f"        System: {system}")
    print(f"        Release: {release}")
    print(f"        Version: {version}")
    validOS = False 
    if system == "Windows":
        os = "Windows"
        if release == '11': 
            try:
                result = subprocess.check_output(['wsl', '--list', '--verbose'], text=True)
                if "Version 2" in result:
                    validOS = True
            except FileNotFoundError:
                print("WSL is not installed")
            except subprocess.CalledProcessError as e:
                print(f"Error checking WSL version: {e}")
    elif system == "Linux":
        print("Running on Linux")

        # Check for specific Linux distributions
        try: 
            with open('/etc/os-release') as f: 
                os_release = f.read()
                os_attributes =get_os_attributes(os_release)
                os = get_os_attributes(os_release)["NAME"]
                validOS = check_os_version(os_attributes)
        except FileNotFoundError:
            print("/etc/os-release file not found. This might not be a typical Linux environment.")
    else:
        print(f"      {X_MARK: >6} Operating System not recognized")
        os = None

    if validOS: 
        print(f"      {OK_MARK: >6} OS is compatible with RAPIDS")
    else:
        print(f"      {X_MARK: >6} OS is not compatible with RAPIDS. Please see https://docs.rapids.ai/install for system requirements.")

    return os 


#CUDA Version : NVIDIA DRIVER Version
SUPPORTED_VERSIONS = {
    "11.2": "470.42.01",
    "11.4": "470.42.01",
    "11.5": "495.29.05",
    "11.8": "520.61.05",
    "12.0": "525.60.13",
    "12.1": "530.30.02",
    "12.2": "535.86.10"
}

def get_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"])
        #print(output)
        version_line = output.decode("utf-8").strip().split('\n')[-1]
        #print(version_line)
        print(version_line.split()[-1].split("/")[0][-4:])
        return version_line.split()[-1].split("/")[0][-4:]  # Extract the version number
    except Exception as e:
        return str(e)
    

def get_driver_version():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, check=True)
        result_chain =  result.stdout.strip()
        return result_chain.split("\n")[0]
    except subprocess.CalledProcessError:
        return None


#https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
def check_driver_compatibility():
    print(f"   {CHECK_SYMBOL} Checking for [italic red]Driver Capability[/italic red]")
    system = platform.system()
    driver_compatible = True
    cuda_version = get_cuda_version()
    print(f"CUDA Version: {cuda_version}")
    driver_version = get_driver_version()
    print(f"Driver Version: {driver_version}")

    if cuda_version >= "12.3": 
        driver_compatible = True
    elif cuda_version < "11.2": 
        driver_compatible = False
    else:
        if driver_version < SUPPORTED_VERSIONS[cuda_version]:
            driver_compatible = False

    if driver_compatible: 
        print(f"      {OK_MARK: >6} CUDA & Driver is compatible with RAPIDS")
    else:
        print(f"      {X_MARK: >6} CUDA & Driver is not compatible with RAPIDS. Please see https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html for CUDA compatability guidance.")

    

def check_sdd_nvme():
    #checks if the system has NVMe SSDs
    print(f"   {CHECK_SYMBOL} Checking for [italic red]NVME SSDs[/italic red]")
    has_nvme = False 
    for disk in psutil.disk_partitions():
        if 'nvme' in disk.device.lower():
            has_nvme = True 
    if has_nvme:
        print(f"      {OK_MARK: >6} SSD drive with preferred NVMe detected for optimized performance.")
    else:
        print(f"      {X_MARK: >6} SSD drive with preferred NVMe not detected. For optimized performance, consider switching to system with NVMe-SSD drive.")



def get_system_memory():
    virtual_memory = psutil.virtual_memory()
    total_memory = virtual_memory.total / (1024 ** 3) #converts bytes to gigabytes
    print("System Memory Information: \n")
    print(f"Total Virtual Memory: {virtual_memory.total / (1024 ** 3):.2f} GB")
    print(f"Available Virtual Memory: {virtual_memory.available / (1024 ** 3):.2f} GB")
    print(f"Used Virtual Memory: {virtual_memory.used / (1024 ** 3):.2f} GB")
    return total_memory
    
    
def get_gpu_memory():
    pynvml.nvmlInit()
    gpus = pynvml.nvmlDeviceGetCount()
    gpu_memory_total = 0
    for i in range(gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_total += memory_info.total / (1024 ** 3) #converts to gigabytes
        print(f"GPU {i} memory: {memory_info.total / (1024 ** 3):.2f} GB")

    pynvml.nvmlShutdown()

    print(f"Total GPU memory: {gpu_memory_total:.2f} GB")
    return gpu_memory_total


#checks that approximately 2:1 ratio of system Memory to total GPU Memory (especially useful for Dask)
def check_memory_to_gpu_ratio():
    print(f"   {CHECK_SYMBOL} Checking for approximately [italic red]2:1 system Memory to total GPU memory ratio[/italic red]")
    system_memory = get_system_memory()
    gpu_memory = get_gpu_memory()
    ratio = system_memory / gpu_memory
    print(f"      System Memory to GPU Memory Ratio: {ratio:.2f}")
    if ratio >= 1.8 and ratio <=2.2:
        print(f"      {OK_MARK: >6} Approximately 2:1 ratio of system Memory to total GPU Memory (especially useful for Dask).")
    else:
        print(f"      {X_MARK: >6} System Memory to total GPU Memory ratio not approximately 2:1 ratio.")


#check for NVLink with 2 or more GPUs 
def check_nvlink_status():
    print(f"   {CHECK_SYMBOL} Checking for [italic red]NVLink with 2 or more GPUs[/italic red]")

    pynvml.nvmlInit()
    try: 
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count < 2:
            print(f"      {X_MARK: >6} Less than 2 GPUs detected. NVLink status check is not applicable.")
        for i in range(device_count):
            print(device_count)
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            for nvlink_id in range(pynvml.NVML_NVLINK_MAX_LINKS):
                try:
                    nvlink_state = pynvml.nvmlDeviceGetNvLinkState(handle, 0)
                    print(f"  NVLink {nvlink_id} State: {nvlink_state}")
                    print(pynvml.NVML_SUCCESS)
                except pynvml.NVMLError as e:
                    print(f"  NVLink {nvlink_id} Status Check Failed: {e}")

    except pynvml.NVMLError as e:
        print(f"NVML Error: {e}")

    pynvml.nvmlShutdown()



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
        

        