import yaml
import importlib
import pytest 

with open('config.yml', 'r') as file: 
    config = yaml.safe_load(file)

CHECK_SYMBOL = config['symbols']['CHECK_SYMBOL']
OK_MARK = config['symbols']['OK_MARK']
X_MARK = config['symbols']['X_MARK']
DOCTOR_SYMBOL = config['symbols']['DOCTOR_SYMBOL']

def cuml_checks():
    print(f"[bold green] {DOCTOR_SYMBOL} Performing REQUIRED health check for cuML [/bold green] \n")
    
    gpu_check = importlib.import_module('rapids_cli.functions').gpu_check
    cuda_check = importlib.import_module('rapids_cli.functions').cuda_check
    get_cuda_version = importlib.import_module('rapids_cli.functions').get_cuda_version
    get_driver_version = importlib.import_module('rapids_cli.functions').get_driver_version
    compare_version = importlib.import_module('rapids_cli.functions').compare_version
    check_gpu_compute_capability = importlib.import_module('rapids_cli.functions').check_gpu_compute_capability
    
    #check if cudf version is greater than 24.12.0
    

    