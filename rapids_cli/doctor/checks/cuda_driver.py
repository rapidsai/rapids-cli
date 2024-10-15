import pynvml  
import yaml 
import subprocess 
import platform 



with open('config.yml', 'r') as file: 
    config = yaml.safe_load(file)

CHECK_SYMBOL = config['symbols']['CHECK_SYMBOL']
OK_MARK = config['symbols']['OK_MARK']
X_MARK = config['symbols']['X_MARK']
DOCTOR_SYMBOL = config['symbols']['DOCTOR_SYMBOL']


def cuda_check():
    try: 
        pynvml.nvmlInit()
        print(f"   {CHECK_SYMBOL} Checking for [italic red]CUDA Availability[/italic red]")
        try: 
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
            print(f"      {OK_MARK: >6} CUDA detected")
            print(f'           CUDA VERSION:{cuda_version//1000}.{cuda_version % 1000}')
            return True 
        except: 
            print(f"      {X_MARK: >6} No CUDA is available")
            return False 
        pynvml.nvmlShutdown()
    except:
        return False
    


#CUDA Version : NVIDIA DRIVER Version
SUPPORTED_VERSIONS = {
    "11.2": "470.42.01",
    "11.4": "470.42.01",
    "11.5": "495.29.05",
    "11.8": "520.61.05",
    "12.0": "525.60.13",
    "12.1": "530.30.02",
    "12.2": "535.86.10"
}

def get_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"])
        #print(output)
        version_line = output.decode("utf-8").strip().split('\n')[-1]
        #print(version_line)
        print(version_line.split()[-1].split("/")[0][-4:])
        return version_line.split()[-1].split("/")[0][-4:]  # Extract the version number
    except Exception as e:
        return str(e)
    

def get_driver_version():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, check=True)
        result_chain =  result.stdout.strip()
        return result_chain.split("\n")[0]
    except subprocess.CalledProcessError:
        return None


#https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
def check_driver_compatibility():
    print(f"   {CHECK_SYMBOL} Checking for [italic red]Driver Capability[/italic red]")
    system = platform.system()
    driver_compatible = True
    cuda_version = get_cuda_version()
    print(f"CUDA Version: {cuda_version}")
    driver_version = get_driver_version()
    print(f"Driver Version: {driver_version}")

    if cuda_version >= "12.3": 
        driver_compatible = True
    elif cuda_version < "11.2": 
        driver_compatible = False
    else:
        if driver_version < SUPPORTED_VERSIONS[cuda_version]:
            driver_compatible = False

    if driver_compatible: 
        print(f"      {OK_MARK: >6} CUDA & Driver is compatible with RAPIDS")
    else:
        print(f"      {X_MARK: >6} CUDA & Driver is not compatible with RAPIDS. Please see https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html for CUDA compatability guidance.")

    


