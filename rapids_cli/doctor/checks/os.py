"""Check the OS version for compatibility with RAPIDS."""

import platform
import subprocess

from rich import print

from rapids_cli.config import config
from rapids_cli.constants import CHECK_SYMBOL, OK_MARK, X_MARK

VALID_LINUX_OS_VERSIONS = config["os_requirements"]["VALID_LINUX_OS_VERSIONS"]


def check_os_version(os_attributes, verbose=False):
    """Check the OS version for compatibility with RAPIDS."""
    os_name = os_attributes["NAME"] + " " + os_attributes["VERSION_ID"]
    if verbose:
        print(f"Current OS Version: {os_name}")
    return os_name in VALID_LINUX_OS_VERSIONS


def get_os_attributes(os_release):
    """Get the OS attributes from the /etc/os-release file."""
    os_attributes = {}
    for attribute in os_release.split("\n"):
        if len(attribute) < 2:
            continue
        # print(attribute.split("="))
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
    print(f"   {CHECK_SYMBOL} Checking for [italic red]OS Capability[/italic red]")
    system = platform.system()
    release = platform.release()
    version = platform.version()
    os = ""

    if verbose:
        print(f"        System: {system}")
        print(f"        Release: {release}")
        print(f"        Version: {version}")
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
            except FileNotFoundError:
                if verbose:
                    print("WSL is not installed")
            except subprocess.CalledProcessError as e:
                if verbose:
                    print(f"Error checking WSL version: {e}")
    elif system == "Linux":

        # Check for specific Linux distributions
        try:
            with open("/etc/os-release") as f:
                os_release = f.read()
                os_attributes = get_os_attributes(os_release)
                os = get_os_attributes(os_release)["NAME"]
                valid_os = check_os_version(os_attributes, verbose)
        except FileNotFoundError:
            if verbose:
                print(
                    "/etc/os-release file not found. This might not be a typical Linux environment."
                )
            else:
                print(f"{X_MARK: >6}")
    else:
        if verbose:
            print(f"      {X_MARK: >6} Operating System not recognized")
        else:
            print(f"{X_MARK: >6}")
        os = None

    if valid_os:
        if verbose:
            print(f"      {OK_MARK: >6} OS is compatible with RAPIDS")
        else:
            print(f"{OK_MARK: >6}")
    else:
        if verbose:
            print(
                f"      {X_MARK: >6} OS is not compatible with RAPIDS. "
                "Please see https://docs.rapids.ai/install for system requirements."
            )
        else:
            print(f"{X_MARK: >6}")

    return os
