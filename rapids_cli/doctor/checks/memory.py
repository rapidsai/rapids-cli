import psutil
import pynvml

from rapids_cli.constants import CHECK_SYMBOL, OK_MARK, X_MARK

from rich.progress import Progress
import time
from rich import print


def get_system_memory():
    virtual_memory = psutil.virtual_memory()
    total_memory = virtual_memory.total / (1024**3)  # converts bytes to gigabytes
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
        gpu_memory_total += memory_info.total / (1024**3)  # converts to gigabytes
        print(f"GPU {i} memory: {memory_info.total / (1024 ** 3):.2f} GB")

    pynvml.nvmlShutdown()

    print(f"Total GPU memory: {gpu_memory_total:.2f} GB")
    return gpu_memory_total


# checks that approximately 2:1 ratio of system Memory to total GPU Memory (especially useful for Dask)
def check_memory_to_gpu_ratio():
    print(
        f"   {CHECK_SYMBOL} Checking for approximately [italic red]2:1 system Memory to total GPU memory ratio[/italic red]"
    )

    total_steps = 5

    with Progress() as progress:
        task = progress.add_task(
            "       System to GPU Memory Ratio checking...", total=total_steps
        )
        for i in range(total_steps):
            time.sleep(0.2)
            progress.update(task, advance=1)

    try:
        pynvml.nvmlInit()
        system_memory = get_system_memory()
        gpu_memory = get_gpu_memory()
        ratio = system_memory / gpu_memory
        print(f"      System Memory to GPU Memory Ratio: {ratio:.2f}")
        if ratio >= 1.8 and ratio <= 2.2:
            print(
                f"{OK_MARK: >6} Approximately 2:1 ratio of system Memory to total GPU Memory (especially useful for Dask)."
            )
        else:
            print(
                f"{X_MARK: >6} System Memory to total GPU Memory ratio not approximately 2:1 ratio."
            )

        pynvml.nvmlShutdown()
        return True
    except pynvml.NVMLError:
        print(f"{X_MARK: >6} GPU not found. Please ensure GPUs are installed.")
        return False
