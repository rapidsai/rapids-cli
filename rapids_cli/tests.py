import pytest
from unittest.mock import patch
import subprocess 
import json
from rapids_cli.doctor.checks.cuda_driver import cuda_check, get_cuda_version, get_driver_version, check_driver_compatibility
from rapids_cli.doctor.checks.dependencies import check_conda, check_pip, check_docker, check_glb

def mock_nvml_get_cuda_driver_version():
    return 12050
    
#cuda_check() tests
def test_cuda_check_available():
    with patch('pynvml.nvmlInit'), patch('pynvml.nvmlSystemGetCudaDriverVersion',  return_value= mock_nvml_get_cuda_driver_version()):
        assert cuda_check() is True

def test_cuda_check_unavailable():
    with patch('pynvml.nvmlInit', side_effect=Exception("Error")):
        assert cuda_check() is False


#get_cuda_version() tests
def test_get_cuda_version_success():
    with patch('subprocess.check_output', return_value=b'nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2024 NVIDIA Corporation\nBuilt on Thu_Jun__6_02:18:23_PDT_2024\nCuda compilation tools, release 12.5, V12.5.82\nBuild cuda_12.5.r12.5/compiler.34385749_0\n'):
        assert get_cuda_version() == '12.5'

def test_get_cuda_version_not_found(): 
    with patch('subprocess.check_output', side_effect=FileNotFoundError):
        assert get_cuda_version() is None

def test_get_cuda_version_error():
    with patch('subprocess.check_output', side_effect=subprocess.CalledProcessError(1, 'nvcc')):
        assert get_cuda_version() is None


def test_get_driver_version_success():
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.stdout = '470.42.01\n'
        assert get_driver_version() == '470.42.01'

def test_get_driver_version_not_found():
    with patch('subprocess.run', side_effect=FileNotFoundError):
        assert get_driver_version() is None

def test_get_driver_version_error():
    with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'nvidia-smi')):
        assert get_driver_version() is None

    

SUPPORTED_VERSIONS = {
    "11.2": "470.42.01",
    "11.4": "470.42.01",
    "11.5": "495.29.05",
    "11.8": "520.61.05",
    "12.0": "525.60.13",
    "12.1": "530.30.02",
    "12.2": "535.86.10"
}

def mock_get_cuda_version():
    return "11.2"

def mock_get_driver_version():
    return "470.42.01"

def test_check_driver_compatibility_compatible(caplog):
    #print("HIIIIIIIIIIIII")
    #print(dir('rapids_cli.doctor.checks.cuda_driver.get_cuda_version'))
    with patch('rapids_cli.doctor.checks.cuda_driver.get_cuda_version', mock_get_cuda_version), \
         patch('rapids_cli.doctor.checks.cuda_driver.get_driver_version', mock_get_driver_version), \
         patch('builtins.print') as mock_print:
        
        check_driver_compatibility()

        assert "CUDA Version: 11.2" in caplog.text
        mock_print.assert_any_call("CUDA Version: 11.2")
        mock_print.assert_any_call("Driver Version: 460.00")
        mock_print.assert_any_call("      X_MARK CUDA & Driver is not compatible with RAPIDS. Please see https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html for CUDA compatibility guidance.")
    

def test_incompatible_driver():
    with patch('my_module.get_cuda_version', lambda: "11.2"), \
         patch('my_module.get_driver_version', lambda: "460.00"), \
         patch('builtins.print') as mock_print:
        
        check_driver_compatibility()
        
        mock_print.assert_any_call("CUDA Version: 11.2")
        mock_print.assert_any_call("Driver Version: 460.00")
        mock_print.assert_any_call("      X_MARK CUDA & Driver is not compatible with RAPIDS. Please see https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html for CUDA compatibility guidance.")

def test_no_cuda_version():
    with patch('my_module.get_cuda_version', lambda: None), \
         patch('my_module.get_driver_version', lambda: "470.42.01"), \
         patch('builtins.print') as mock_print:
        
        check_driver_compatibility()
        
        mock_print.assert_any_call("CUDA Version: None")
        mock_print.assert_any_call("Driver Version: 470.42.01")
        mock_print.assert_any_call("      X_MARK CUDA & Driver is not compatible with RAPIDS. Please see https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html for CUDA compatibility guidance.")



# Sample config for testing
sample_config = {
    'min_supported_versions': {
        'conda_requirement': '22.11',
        'docker_requirement': '19.03'
    }
}

# Mocking config
@pytest.fixture(autouse=True)
def mock_config(monkeypatch):
    monkeypatch.setattr('rapids_cli.config.config', sample_config)

# Test for check_conda function
def test_check_conda_compatible():
    with patch('subprocess.check_output', return_value=json.dumps({"conda_version": "22.11"}).encode('utf-8')):
        with patch('builtins.print') as mock_print:
            assert check_conda() is True

    
            #mock_print.assert_any_call("      OK_MARK CONDA Version is compatible with RAPIDS")

def test_check_conda_incompatible():
    with patch('subprocess.check_output', return_value=json.dumps({"conda_version": "20.00"}).encode('utf-8')):
        with patch('builtins.print') as mock_print:
            assert check_conda() is False
            #mock_print.assert_any_call("      X_MARK CONDA Version is not compatible with RAPIDS - please upgrade to Docker 4.8")


def test_check_docker_compatible():
    with patch('subprocess.check_output', side_effect=[
        b'Docker version 20.10.0, build 3946899\n',
        b'{"Client":{"Version":"20.10.0"}}'
    ]):
        with patch('builtins.print') as mock_print:
            assert check_docker() is True 
            #mock_print.assert_any_call("      OK_MARK DOCKER Version is compatible with RAPIDS")

def test_check_docker_incompatible():
    with patch('subprocess.check_output', side_effect=[
        b'Docker version 18.09.0, build 4d60db4\n',
        b'{"Client":{"Version":"18.09.0"}}'
    ]):
        with patch('builtins.print') as mock_print:
            assert check_docker() is False
            #mock_print.assert_any_call("      X_MARK DOCKER Version is not compatible with RAPIDS - please upgrade to Docker 19.03")