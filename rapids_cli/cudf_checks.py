import yaml
import importlib

with open('config.yml', 'r') as file: 
    config = yaml.safe_load(file)

CHECK_SYMBOL = config['symbols']['CHECK_SYMBOL']
OK_MARK = config['symbols']['OK_MARK']
X_MARK = config['symbols']['X_MARK']
DOCTOR_SYMBOL = config['symbols']['DOCTOR_SYMBOL']


def cudf_checks(cuda_requirement, driver_requirement, compute_requirement, dependencies):

    print(f"[bold green] {DOCTOR_SYMBOL} Performing REQUIRED health check for CUDF [/bold green] \n")
    
    gpu_check = importlib.import_module('rapids_cli.functions').gpu_check
    cuda_check = importlib.import_module('rapids_cli.functions').cuda_check
    get_cuda_version = importlib.import_module('rapids_cli.functions').get_cuda_version
    get_driver_version = importlib.import_module('rapids_cli.functions').get_driver_version
    compare_version = importlib.import_module('rapids_cli.functions').compare_version
    check_gpu_compute_capability = importlib.import_module('rapids_cli.functions').check_gpu_compute_capability
    
    
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
    
