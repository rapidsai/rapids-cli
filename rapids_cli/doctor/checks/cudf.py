from rapids_cli.doctor.checks.cuda_driver import (
    get_cuda_version,
    cuda_check,
    get_driver_version,
)
from rapids_cli.doctor.checks.gpu import gpu_check, check_gpu_compute_capability

from rapids_cli.constants import CHECK_SYMBOL, OK_MARK, X_MARK, DOCTOR_SYMBOL


def compare_version(version, requirement):
    if str(version) >= str(requirement):
        return True
    return False


def cudf_checks(cuda_requirement, driver_requirement, compute_requirement):

    print(
        f"[bold green] {DOCTOR_SYMBOL} Performing REQUIRED health check for CUDF [/bold green] \n"
    )

    print(f"   {CHECK_SYMBOL} Checking for [italic red]CUDA dependencies[/italic red]")
    if compare_version(
        get_cuda_version(), cuda_requirement
    ):  # when the other branch gets merged, will move the magic numbers to their yaml file
        print(f"{OK_MARK: >6}  CUDA version compatible with CUDF")
    else:
        print(
            f"{X_MARK: >6}  CUDA version not compatible with CUDF. Please upgrade to {cuda_requirement}"
        )

    print(
        f"   {CHECK_SYMBOL} Checking for [italic red]Driver Availability[/italic red]"
    )
    if cuda_check():
        if compare_version(get_driver_version(), driver_requirement):
            print(f"{OK_MARK: >6}  Nvidia Driver version compatible with CUDF")
        else:
            print(
                f"{X_MARK: >6}  Nvidia Driver version not compatible with CUDF. Please upgrade to {driver_requirement}"
            )
    else:
        print(f"{X_MARK: >6} No Nvidia Driver Detected")

    if gpu_check():
        if check_gpu_compute_capability(compute_requirement):
            print(f"{OK_MARK: >6}  GPU compute compatible with CUDF")
        else:
            print(
                f"{X_MARK: >6}  GPU compute not compatible with CUDF. Please upgrade to compute >={compute_requirement}"
            )
