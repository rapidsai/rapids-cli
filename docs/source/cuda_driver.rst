.. _cuda_driver:

CUDA Driver Checks
==================

This module provides functions to check the availability of CUDA, retrieve CUDA version,
and verify compatibility between the CUDA toolkit and NVIDIA drivers.

Functions
---------
.. autofunction:: rapids_cli.doctor.checks.cuda_driver.cuda_check

   Checks if CUDA is available on the system by initializing the NVML and retrieving the CUDA driver version.

   :return: True if CUDA is available, False otherwise.

.. autofunction:: rapids_cli.doctor.checks.cuda_driver.get_cuda_version

   Retrieves the version of the installed CUDA toolkit.

   :return: A string representing the CUDA version or None if CUDA is not found.

.. autofunction:: rapids_cli.doctor.checks.cuda_driver.get_driver_version

   Retrieves the installed NVIDIA driver version.

   :return: A string representing the NVIDIA driver version or None if the driver is not found.

.. autofunction:: rapids_cli.doctor.checks.cuda_driver.check_driver_compatibility

   Checks the compatibility between the installed CUDA version and the NVIDIA driver version.

   This function prints whether the installed versions are compatible with RAPIDS.

   :return: None
