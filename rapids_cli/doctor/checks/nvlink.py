import pynvml

from rapids_cli.constants import CHECK_SYMBOL, OK_MARK, X_MARK, DOCTOR_SYMBOL


#check for NVLink with 2 or more GPUs 
def check_nvlink_status():
    print(f"   {CHECK_SYMBOL} Checking for [italic red]NVLink with 2 or more GPUs[/italic red]")

    pynvml.nvmlInit()
    try: 
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count < 2:
            print(f"      {X_MARK: >6} Less than 2 GPUs detected. NVLink status check is not applicable.")
        for i in range(device_count):
            print(device_count)
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            for nvlink_id in range(pynvml.NVML_NVLINK_MAX_LINKS):
                try:
                    nvlink_state = pynvml.nvmlDeviceGetNvLinkState(handle, 0)
                    print(f"  NVLink {nvlink_id} State: {nvlink_state}")
                    print(pynvml.NVML_SUCCESS)
                except pynvml.NVMLError as e:
                    print(f"  NVLink {nvlink_id} Status Check Failed: {e}")

    except pynvml.NVMLError as e:
        print(f"NVML Error: {e}")

    pynvml.nvmlShutdown()


