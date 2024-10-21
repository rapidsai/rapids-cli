# rapids-doctor-example

This is an initial version of RAPIDS doctor, a command line tool used to
perform health checks upon installation of RAPIDS to make sure GPU, Driver, OS etc.
meet RAPIDS requirements. More details can be found [here](https://docs.google.com/document/d/1mNicpQnIcFfPcLdVpewk_UKT56EhhxqPsMLxw0Atx-w/edit#heading=h.2ygxf373qzps)

So far, I have implemented the following checks based off of the [setup requirements](https://docs.rapids.ai/install)

**Required**:

- GPU compute capability : `check_gpu_compute_capability()`
- OS version compatibility : `check_os_compatibility()`
- Driver & CUDA compatibility : `check_driver_compatibility()`

**Recommended**:

- NVMe SSD Detection: `check_sdd_nvme()`
- System Memory to GPU Memory 2:1 ratio: `check_memory_to_gpu_ratio()`
- NVLink with 2 or more GPUs: `check_nvlink_status()`

## Getting started

After cloning the repository, run the following commands in your terminal:

`conda env create -f environment.yml`

`pip install .`

`rapids doctor`

## Subcommands

Currently RAPIDS Doctor supports the following health checks:

- `rapids doctor` (system runthrough of basic dependencies and requirements)
- `rapids doctor cudf` (checks for cuDF specific dependencies)
- `rapids doctor cuml` (checks for cuML specific dependencies)
