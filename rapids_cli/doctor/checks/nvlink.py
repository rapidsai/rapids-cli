"""Check for NVLink status."""

import pynvml
from rich import print

from rapids_cli.constants import CHECK_SYMBOL, OK_MARK, X_MARK


def check_nvlink_status(verbose=True):
    """Check the system for NVLink with 2 or more GPUs."""
    print(
        f"   {CHECK_SYMBOL} Checking for [italic red]NVLink with 2 or more GPUs[/italic red]"
    )

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count < 2:
            if verbose:
                print(
                    f"      {X_MARK: >6} Less than 2 GPUs detected. NVLink status check is not applicable."
                )
            else:
                print(f"{X_MARK: >6}")

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            for nvlink_id in range(pynvml.NVML_NVLINK_MAX_LINKS):
                try:
                    nvlink_state = pynvml.nvmlDeviceGetNvLinkState(handle, 0)
                    if verbose:
                        print(f"  NVLink {nvlink_id} State: {nvlink_state}")
                    else:
                        print(f"{OK_MARK: >6}")
                except pynvml.NVMLError as e:
                    if verbose:
                        print(f"  NVLink {nvlink_id} Status Check Failed: {e}")
                    else:
                        print(f"{X_MARK: >6}")
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        if verbose:
            print(f"{X_MARK: >6} GPU not found. Please ensure GPUs are installed.")
        else:
            print(f"{X_MARK: >6}")
