import yaml

from rapids_cli.doctor.checks.os import detect_os
from rapids_cli.doctor.checks.memory import check_memory_to_gpu_ratio
from rapids_cli.doctor.checks.nvlink import check_nvlink_status
from rapids_cli.doctor.checks.gpu import gpu_check, check_gpu_compute_capability
from rapids_cli.doctor.checks.cuda_driver import get_cuda_version, cuda_check, get_driver_version, check_driver_compatibility
from rapids_cli.doctor.checks.sdd_nvme import check_sdd_nvme
from rapids_cli.doctor.checks.dependencies import check_conda, check_pip,check_docker, check_glb

CHECK_SYMBOL = "🚨"
OK_MARK = "✅"
X_MARK = "❌"
DOCTOR_SYMBOL = "🧑‍⚕️"
VALID_SUBCOMMANDS = ["cudf"]





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