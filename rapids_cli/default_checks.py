import pkg_resources
import importlib
import yaml


with open('config.yml', 'r') as file: 
    config = yaml.safe_load(file)

CHECK_SYMBOL = config['symbols']['CHECK_SYMBOL']
OK_MARK = config['symbols']['OK_MARK']
X_MARK = config['symbols']['X_MARK']
DOCTOR_SYMBOL = config['symbols']['DOCTOR_SYMBOL']

gpu_compute_requirement = config['min_supported_versions']['gpu_compute_requirement']
docker_requirement = config['min_supported_versions']['docker_requirement']
conda_requirement = config['min_supported_versions']['conda_requirement']



def default_checks(): 
    print(f"[bold green] {DOCTOR_SYMBOL} Performing REQUIRED health check for RAPIDS [/bold green] \n")
    # Load the entry points

    gpu_check = importlib.import_module('rapids_cli.functions').gpu_check
    cuda_check = importlib.import_module('rapids_cli.functions').cuda_check
    check_gpu_compute_capability = importlib.import_module('rapids_cli.functions').check_gpu_compute_capability
    check_driver_compatibility = importlib.import_module('rapids_cli.functions').check_driver_compatibility
    detect_os = importlib.import_module('rapids_cli.functions').detect_os 
    check_sdd_nvme = importlib.import_module('rapids_cli.functions').check_sdd_nvme
    check_memory_to_gpu_ratio = importlib.import_module('rapids_cli.functions').check_memory_to_gpu_ratio
    check_nvlink_status = importlib.import_module('rapids_cli.functions').check_nvlink_status
    check_docker = importlib.import_module('rapids_cli.functions').check_docker
    check_conda = importlib.import_module('rapids_cli.functions').check_conda
    check_pip = importlib.import_module('rapids_cli.functions').check_pip
    check_glb = importlib.import_module('rapids_cli.functions').check_glb

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

