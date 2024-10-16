import psutil 
import yaml 

with open('config.yml', 'r') as file: 
    config = yaml.safe_load(file)

CHECK_SYMBOL = config['symbols']['CHECK_SYMBOL']
OK_MARK = config['symbols']['OK_MARK']
X_MARK = config['symbols']['X_MARK']
DOCTOR_SYMBOL = config['symbols']['DOCTOR_SYMBOL']



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

