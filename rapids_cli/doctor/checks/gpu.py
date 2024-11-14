import pynvml

from rapids_cli.constants import CHECK_SYMBOL, OK_MARK, X_MARK
from rich.progress import Progress
import time
from rich import print



def gpu_check():
    print(f"   {CHECK_SYMBOL} Checking for [italic red]GPU Availability[/italic red]")
    total_steps = 5

    with Progress() as progress:
        task = progress.add_task(
            "       GPU Availability checking...", total=total_steps
        )
        for i in range(total_steps):
            time.sleep(0.2)
            progress.update(task, advance=1)
    try:
        pynvml.nvmlInit()
        try:
            num_gpus = pynvml.nvmlDeviceGetCount()
            print(f"      {OK_MARK: >6} Number of GPUs detected: {num_gpus}")
            return True
        except pynvml.NVMLError:
            print(f"      {X_MARK: >6} GPU detected but not available")
            return False

        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        print(f"      {X_MARK: >6} No available GPUs detected")
        return False


def check_gpu_compute_capability(required_capability):
    # Initialize pynvml
    print(
        f"   {CHECK_SYMBOL} Checking for [italic red]GPU Compute Capability[/italic red]"
    )
    total_steps = 5

    with Progress() as progress:
        task = progress.add_task("       GPU Compute checking...", total=total_steps)
        for i in range(total_steps):
            time.sleep(0.2)
            progress.update(task, advance=1)

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
                print(
                    f"         GPU {i} meets the required compute capability {required_capability[0]}.{required_capability[1]}"
                )
            else:
                print(
                    f"         GPU {i} does not meet the required compute capability {required_capability[0]}.{required_capability[1]}."
                )

        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        print(f"       {X_MARK: >6} No GPU - cannot determineg GPU Compute Capability")

    return meets_requirement
