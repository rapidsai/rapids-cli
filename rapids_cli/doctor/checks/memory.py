import psutil
import pynvml

from rapids_cli.constants import CHECK_SYMBOL, OK_MARK, X_MARK
from rich import print


def get_system_memory(verbose=False):
    virtual_memory = psutil.virtual_memory()
    total_memory = virtual_memory.total / (1024**3)  # converts bytes to gigabytes
    if verbose:
        print("System Memory Information: \n")
        print(f"Total Virtual Memory: {virtual_memory.total / (1024 ** 3):.2f} GB")
        print(
            f"Available Virtual Memory: {virtual_memory.available / (1024 ** 3):.2f} GB"
        )
        print(f"Used Virtual Memory: {virtual_memory.used / (1024 ** 3):.2f} GB")
    return total_memory


def get_gpu_memory(verbose=False):
    pynvml.nvmlInit()
    gpus = pynvml.nvmlDeviceGetCount()
    gpu_memory_total = 0
    for i in range(gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_total += memory_info.total / (1024**3)  # converts to gigabytes
        if verbose:
            print(f"GPU {i} memory: {memory_info.total / (1024 ** 3):.2f} GB")

    pynvml.nvmlShutdown()
    if verbose:
        print(f"Total GPU memory: {gpu_memory_total:.2f} GB")
    return gpu_memory_total


# checks that approximately 2:1 ratio of system Memory to total GPU Memory (especially useful for Dask)
def check_memory_to_gpu_ratio(verbose=True):
    print(
        f"   {CHECK_SYMBOL} Checking for approximately [italic red]2:1 system Memory to total GPU memory ratio[/italic red]"
    )
    try:
        pynvml.nvmlInit()
        system_memory = get_system_memory(verbose)
        gpu_memory = get_gpu_memory(verbose)
        ratio = system_memory / gpu_memory
        if verbose:
            print(f"      System Memory to GPU Memory Ratio: {ratio:.2f}")
        if ratio >= 1.8 and ratio <= 2.2:
            if verbose:
                print(
                    f"{OK_MARK: >6} Approximately 2:1 ratio of system Memory to total GPU Memory (especially useful for Dask)."
                )
            else:
                print(f"{OK_MARK: >6}")
        else:
            if verbose:
                print(
                    f"{X_MARK: >6} System Memory to total GPU Memory ratio not approximately 2:1 ratio."
                )
            else:
                print(f"{X_MARK: >6}")

        pynvml.nvmlShutdown()
        return True
    except pynvml.NVMLError:
        if verbose:
            print(f"{X_MARK: >6} GPU not found. Please ensure GPUs are installed.")
        else:
            print(f"{X_MARK: >6}")
        return False
