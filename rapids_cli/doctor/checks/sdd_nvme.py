"""Checks for NVMe SSDs."""

import psutil


def check_sdd_nvme(verbose=False):
    """Checks if the system has NVMe SSDs."""
    has_nvme = False
    for disk in psutil.disk_partitions():
        if "nvme" in disk.device.lower():
            has_nvme = True
    if has_nvme:
        return True
    else:
        raise ValueError(
            "SSD drive with preferred NVMe not detected. "
            "For optimized performance, consider switching to system with NVMe-SSD drive."
        )
