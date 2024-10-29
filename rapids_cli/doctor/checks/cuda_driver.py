import pynvml
import subprocess
import platform

from rapids_cli.constants import CHECK_SYMBOL, OK_MARK, X_MARK


def cuda_check(VERBOSE_MODE=False):
    try:
        pynvml.nvmlInit()
        print(
            f"   {CHECK_SYMBOL} Checking for [italic red]CUDA Availability[/italic red]"
        )
        try:
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
            if VERBOSE_MODE:
                print(f"      {OK_MARK: >6} CUDA detected")
                print(
                    f"           CUDA VERSION:{cuda_version//1000}.{cuda_version % 1000}"
                )
            else:
                print(f"{OK_MARK: >6}")
            return True
        except pynvml.NVMLError:
            if VERBOSE_MODE:
                print(f"      {X_MARK: >6} No CUDA its available")
            else:
                print(f"{X_MARK: >6}")
            return False
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        return False


# CUDA Version : NVIDIA DRIVER Version
SUPPORTED_VERSIONS = {
    "11.2": "470.42.01",
    "11.4": "470.42.01",
    "11.5": "495.29.05",
    "11.8": "520.61.05",
    "12.0": "525.60.13",
    "12.1": "530.30.02",
    "12.2": "535.86.10",
}


def get_cuda_version(VERBOSE_MODE=False):

    try:
        output = subprocess.check_output(["nvcc", "--version"])
        # print(output)
        version_line = output.decode("utf-8").strip().split("\n")[-1]
        # print(version_line)
        # print(version_line.split()[-1].split("/")[0][-4:])
        return version_line.split()[-1].split("/")[0][-4:]  # Extract the version number
    except Exception:
        if VERBOSE_MODE:
            print(
                f"{X_MARK: >6} CUDA not found. Please ensure CUDA toolkit is installed."
            )
        return None


def get_driver_version(VERBOSE_MODE=False):

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )

        result_chain = result.stdout.strip()
        return result_chain.split("\n")[0]
    except FileNotFoundError:
        if VERBOSE_MODE:
            print(
                f"{X_MARK: >6} nvidia-smi not found. Please ensure NVIDIA drivers are installed."
            )
        return None
    except subprocess.CalledProcessError:
        return None


# https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
def check_driver_compatibility(VERBOSE_MODE=False):
    print(f"   {CHECK_SYMBOL} Checking for [italic red]Driver Capability[/italic red]")
    platform.system()
    driver_compatible = True
    cuda_version = get_cuda_version()
    if VERBOSE_MODE:
        print(f"CUDA Version: {cuda_version}")
    driver_version = get_driver_version()
    if VERBOSE_MODE:
        print(f"Driver Version: {driver_version}")

    if not driver_version or not cuda_version:
        driver_compatible = False
    elif cuda_version >= "12.3":
        driver_compatible = True
    elif cuda_version < "11.2":
        driver_compatible = False
    else:
        if driver_version < SUPPORTED_VERSIONS[cuda_version]:
            driver_compatible = False

    if driver_compatible:
        if VERBOSE_MODE:
            print(f"      {OK_MARK: >6} CUDA & Driver is compatible with RAPIDS")
        else:
            print(f"{OK_MARK: >6}")
    else:
        if VERBOSE_MODE:
            print(
                f"      {X_MARK: >6} CUDA & Driver is not compatible with RAPIDS. Please see https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html for CUDA compatability guidance."
            )
        else:
            print(f"{X_MARK: >6}")
