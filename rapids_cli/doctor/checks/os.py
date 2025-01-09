"""Check the OS version for compatibility with RAPIDS."""

import platform
import subprocess

from rapids_cli.config import config

VALID_LINUX_OS_VERSIONS = config["os_requirements"]["VALID_LINUX_OS_VERSIONS"]


def check_os_version(os_attributes, verbose=False):
    """Check the OS version for compatibility with RAPIDS."""
    os_name = os_attributes["NAME"] + " " + os_attributes["VERSION_ID"]
    return os_name in VALID_LINUX_OS_VERSIONS


def get_os_attributes(os_release):
    """Get the OS attributes from the /etc/os-release file."""
    os_attributes = {}
    for attribute in os_release.split("\n"):
        if len(attribute) < 2:
            continue
        key, value = attribute.split("=")[0], attribute.split("=")[1]
        os_attributes[key] = value[1:-1]

    return os_attributes


def get_linux_os_version():
    """Get the Linux OS version."""
    try:
        with open("/etc/os-release") as f:
            os_release = f.read()
            os_attributes = get_os_attributes(os_release)
        return os_attributes
    except FileNotFoundError:
        return "OS release file not found."


def detect_os(verbose=False):
    """Detect the OS and check for compatibility with RAPIDS."""
    system = platform.system()
    release = platform.release()
    os = ""

    valid_os = False
    if system == "Windows":
        os = "Windows"
        if release == "11":
            try:
                result = subprocess.check_output(
                    ["wsl", "--list", "--verbose"], text=True
                )
                if "Version 2" in result:
                    valid_os = True
            except FileNotFoundError as e:
                raise ValueError("WSL is not installed") from e
            except subprocess.CalledProcessError as e:
                raise ValueError("Error checking WSL version") from e
    elif system == "Linux":

        # Check for specific Linux distributions
        try:
            with open("/etc/os-release") as f:
                os_release = f.read()
                os_attributes = get_os_attributes(os_release)
                os = get_os_attributes(os_release)["NAME"]
                valid_os = check_os_version(os_attributes, verbose)
        except FileNotFoundError as e:
            raise ValueError(
                "/etc/os-release file not found. This might not be a typical Linux environment."
            ) from e
    else:
        raise ValueError("Operating System not recognized")

    if valid_os:
        return True
    else:
        raise ValueError(
            f"OS {os} is not compatible with RAPIDS. "
            "Please see https://docs.rapids.ai/install for system requirements."
        )
