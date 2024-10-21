import platform 
import subprocess

from rapids_cli.config import config
from rapids_cli.constants import CHECK_SYMBOL, OK_MARK, X_MARK

VALID_LINUX_OS_VERSIONS = config['os_requirements']['VALID_LINUX_OS_VERSIONS']


def check_os_version(os_attributes):
    os_name = os_attributes['NAME'] + " " + os_attributes['VERSION_ID']
    print(f"Current OS Version: {os_name}")
    return os_name in VALID_LINUX_OS_VERSIONS 

    
def get_os_attributes(os_release): 
    os_attributes = {}
    for attribute in os_release.split("\n"):
        if len(attribute) < 2: continue 
        #print(attribute.split("="))
        key, value = attribute.split("=")[0], attribute.split("=")[1]
        os_attributes[key] = value[1:-1]
    
    return os_attributes

def get_linux_os_version():
    try: 
        with open('/etc/os-release') as f: 
            os_release = f.read()
            os_attributes =get_os_attributes(os_release)        
        return os_attributes
    except FileNotFoundError:
        return "OS release file not found."
        
def detect_os():
    print(f"   {CHECK_SYMBOL} Checking for [italic red]OS Capability[/italic red]")
    system = platform.system()
    release = platform.release()
    version = platform.version()
    os = ""

    print(f"        System: {system}")
    print(f"        Release: {release}")
    print(f"        Version: {version}")
    validOS = False 
    if system == "Windows":
        os = "Windows"
        if release == '11': 
            try:
                result = subprocess.check_output(['wsl', '--list', '--verbose'], text=True)
                if "Version 2" in result:
                    validOS = True
            except FileNotFoundError:
                print("WSL is not installed")
            except subprocess.CalledProcessError as e:
                print(f"Error checking WSL version: {e}")
    elif system == "Linux":
        print("Running on Linux")

        # Check for specific Linux distributions
        try: 
            with open('/etc/os-release') as f: 
                os_release = f.read()
                os_attributes =get_os_attributes(os_release)
                os = get_os_attributes(os_release)["NAME"]
                validOS = check_os_version(os_attributes)
        except FileNotFoundError:
            print("/etc/os-release file not found. This might not be a typical Linux environment.")
    else:
        print(f"      {X_MARK: >6} Operating System not recognized")
        os = None

    if validOS: 
        print(f"      {OK_MARK: >6} OS is compatible with RAPIDS")
    else:
        print(f"      {X_MARK: >6} OS is not compatible with RAPIDS. Please see https://docs.rapids.ai/install for system requirements.")

    return os 
