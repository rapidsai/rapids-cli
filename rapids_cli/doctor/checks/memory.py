"""Memory checks."""

import warnings

import psutil
import pynvml


def get_system_memory(verbose=False):
    """Get the total system memory."""
    virtual_memory = psutil.virtual_memory()
    total_memory = virtual_memory.total / (1024**3)  # converts bytes to gigabytes
    return total_memory


def get_gpu_memory(verbose=False):
    """Get the total GPU memory."""
    pynvml.nvmlInit()
    gpus = pynvml.nvmlDeviceGetCount()
    gpu_memory_total = 0
    for i in range(gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_total += memory_info.total / (1024**3)  # converts to gigabytes

    pynvml.nvmlShutdown()
    return gpu_memory_total


def check_memory_to_gpu_ratio(verbose=True):
    """Check the system for a 2:1 ratio of system Memory to total GPU Memory.

    This is especially useful for Dask.

    """
    try:
        pynvml.nvmlInit()
        system_memory = get_system_memory(verbose)
        gpu_memory = get_gpu_memory(verbose)
        ratio = system_memory / gpu_memory
        if ratio >= 1.8:
            return True
        else:
            warnings.warn(
                "System Memory to total GPU Memory ratio not at least 2:1 ratio. "
                "It is recommended to have double the system memory to GPU memory for optimal performance.",
                stacklevel=2,
            )
            return True
    except pynvml.NVMLError as e:
        raise ValueError("GPU not found. Please ensure GPUs are installed.") from e
