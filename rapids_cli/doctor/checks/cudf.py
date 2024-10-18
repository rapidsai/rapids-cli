import yaml 
from rapids_cli.doctor.checks.cuda_driver import get_cuda_version, cuda_check, get_driver_version, check_driver_compatibility
from rapids_cli.doctor.checks.gpu import gpu_check, check_gpu_compute_capability

from rapids_cli.constants import CHECK_SYMBOL, OK_MARK, X_MARK, DOCTOR_SYMBOL


def compare_version(version, requirement):
    if str(version) >= str(requirement): 
        return True
    return False 

def import_cudf():
    print(f"   {CHECK_SYMBOL} Checking for successful installation of [italic red]cuDF[/italic red]")
    try: 
        import cudf 
        print(f"{OK_MARK: >6} cuDF successfully imported")
        data = {
        'id': [0, 1, 2, 3, 4],                 
        'value': [10.5, 20.0, 30.1, 40.3, 50.8] 
        }   
        df = cudf.DataFrame(data)
        all_checks_passed = True
        try: 
            assert(df.shape == (5, 2)), f"{X_MARK: >6} cuDF dataframe dimensions are wrong"
        except AssertionError as e: 
            all_checks_passed = False
            print(f"{X_MARK: >6  e}")
        try: 
            expected_columns = ['id', 'value']
            assert all(col in df.columns for col in expected_columns), f"{X_MARK: >6} DataFrame columns do not match expected"
        except AssertionError as e: 
            all_checks_passed = False
            print(f"{X_MARK: >6  e}")

        if all_checks_passed: 
            print(f"{OK_MARK: >6} cuDF tests successful")
        else: 
            print(f"{X_MARK: >6} cuDF tests unsuccessful - please check if cuDF library and dependencies are up to date.")
        
    except ImportError: 
        print(f"{X_MARK: >6}  cuDF could not be imported. Please install cuDF https://docs.rapids.ai/install/")
        


def cudf_checks(cuda_requirement, driver_requirement, compute_requirement):

    print(f"[bold green] {DOCTOR_SYMBOL} Performing REQUIRED health check for CUDF [/bold green] \n")
    
    
    print(f"   {CHECK_SYMBOL} Checking for [italic red]CUDA dependencies[/italic red]")
    if compare_version(get_cuda_version(), cuda_requirement): #when the other branch gets merged, will move the magic numbers to their yaml file 
        print(f"{OK_MARK: >6}  CUDA version compatible with CUDF")
    else:
        print(f"{X_MARK: >6}  CUDA version not compatible with CUDF. Please upgrade to {cuda_requirement}")
    

    print(f"   {CHECK_SYMBOL} Checking for [italic red]Driver Availability[/italic red]")
    if cuda_check():
        if compare_version(get_driver_version(), driver_requirement):
            print(f"{OK_MARK: >6}  Nvidia Driver version compatible with CUDF")
        else:
            print(f"{X_MARK: >6}  Nvidia Driver version not compatible with CUDF. Please upgrade to {driver_requirement}")
    else: 
        print(f"{X_MARK: >6} No Nvidia Driver Detected")

    if gpu_check():
        if check_gpu_compute_capability(compute_requirement):
            print(f"{OK_MARK: >6}  GPU compute compatible with CUDF")
        else:
            print(f"{X_MARK: >6}  GPU compute not compatible with CUDF. Please upgrade to compute >={compute_requirement}") 


    import_cudf()

