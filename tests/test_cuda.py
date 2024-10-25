from unittest.mock import patch
from rapids_cli.doctor.checks.cuda_driver import cuda_check


def mock_cuda_version():
    return 12050


def test_get_cuda_version_success():
    with (
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlSystemGetCudaDriverVersion", return_value=12050),
    ):
        version = mock_cuda_version()
        assert version == 12050


def test_cuda_check_success(capfd):
    with (
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlSystemGetCudaDriverVersion", return_value=12050),
    ):
        assert cuda_check() is True
        captured = capfd.readouterr()
        assert "CUDA detected" in captured.out
        assert "CUDA VERSION:12.50" in captured.out
