import pynvml

from rapids_cli.constants import CHECK_SYMBOL, OK_MARK, X_MARK


def gpu_check(VERBOSE_MODE=False):
    print(f"   {CHECK_SYMBOL} Checking for [italic red]GPU Availability[/italic red]")
    try:
        pynvml.nvmlInit()
        try:
            num_gpus = pynvml.nvmlDeviceGetCount()
            if VERBOSE_MODE:
                print(f"      {OK_MARK: >6} Number of GPUs detected: {num_gpus}")
            else:
                print(f"{OK_MARK: >6}")
            return True
        except pynvml.NVMLError:
            if VERBOSE_MODE:
                print(f"      {X_MARK: >6} GPU detected but not available")
            else:
                print(f"{X_MARK: >6}")
            return False

        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        if VERBOSE_MODE:
            print(f"      {X_MARK: >6} No available GPUs detected")
        else:
            print(f"{X_MARK: >6}")
        return False


def check_gpu_compute_capability(VERBOSE_MODE):
    # Initialize pynvml
    print(
        f"   {CHECK_SYMBOL} Checking for [italic red]GPU Compute Capability[/italic red]"
    )
    meets_requirement = False
    try:
        required_capability = None
        pynvml.nvmlInit()

        num_gpus = pynvml.nvmlDeviceGetCount()
        for i in range(num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            compute_capability = major * 10 + minor
            if VERBOSE_MODE:
                print(f"      GPU {i} Compute Capability: {major}.{minor}")

            if compute_capability >= int(required_capability):
                meets_requirement = True
                if VERBOSE_MODE:
                    print(
                        f"         GPU {i} meets the required compute capability {required_capability[0]}.{required_capability[1]}"
                    )
                else:
                    print(f"{OK_MARK: >6}")
            else:
                if VERBOSE_MODE:
                    print(
                        f"         GPU {i} does not meet the required compute capability {required_capability[0]}.{required_capability[1]}."
                    )
                else:
                    print(f"{X_MARK: >6}")

        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        if VERBOSE_MODE:
            print(
                f"       {X_MARK: >6} No GPU - cannot determineg GPU Compute Capability"
            )
        else:
            print(f"{X_MARK: >6}")

    return meets_requirement
