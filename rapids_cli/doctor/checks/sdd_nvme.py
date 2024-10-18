import psutil 

from rapids_cli.constants import CHECK_SYMBOL, OK_MARK, X_MARK, DOCTOR_SYMBOL


def check_sdd_nvme():
    #checks if the system has NVMe SSDs
    print(f"   {CHECK_SYMBOL} Checking for [italic red]NVME SSDs[/italic red]")
    has_nvme = False 
    for disk in psutil.disk_partitions():
        if 'nvme' in disk.device.lower():
            has_nvme = True 
    if has_nvme:
        print(f"      {OK_MARK: >6} SSD drive with preferred NVMe detected for optimized performance.")
    else:
        print(f"      {X_MARK: >6} SSD drive with preferred NVMe not detected. For optimized performance, consider switching to system with NVMe-SSD drive.")

