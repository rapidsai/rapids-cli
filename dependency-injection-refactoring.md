# Dependency Injection Refactoring

## Context

The check modules (`gpu.py`, `cuda_driver.py`, `memory.py`, `nvlink.py`)
and `debug.py` previously called `pynvml`, `psutil`, and `cuda.pathfinder`
directly. This forced tests to use 50+ `mock.patch` calls with deeply
nested context managers and `MagicMock` objects to simulate hardware
configurations. A thin abstraction layer was introduced so tests can
construct plain dataclasses instead of mocking low-level library internals.

## Approach: Default Parameter Injection with Provider Dataclasses

A single new file `rapids_cli/hardware.py` was created containing:

- **`DeviceInfo`** dataclass -- holds per-GPU data
  (index, compute capability, memory, nvlink states)
- **`GpuInfoProvider`** protocol -- read-only interface for GPU info
  (`device_count`, `devices`, `cuda_driver_version`, `driver_version`)
- **`SystemInfoProvider`** protocol -- read-only interface for system info
  (`total_memory_bytes`, `cuda_runtime_path`)
- **`NvmlGpuInfo`** -- real implementation backed by pynvml
  (lazy-loads on first property access, caches results)
- **`DefaultSystemInfo`** -- real implementation backed by
  psutil + cuda.pathfinder (lazy-loads per property)
- **`FakeGpuInfo`** / **`FakeSystemInfo`** -- test fakes
  (plain dataclasses, no hardware dependency)
- **`FailingGpuInfo`** / **`FailingSystemInfo`** -- test fakes that
  raise `ValueError` on access (simulates missing hardware)

Check functions gained an optional keyword parameter with `None` default:

```python
def gpu_check(verbose=False, *, gpu_info: GpuInfoProvider | None = None, **kwargs):
    if gpu_info is None:  # pragma: no cover
        gpu_info = NvmlGpuInfo()
```

The orchestrator (`doctor.py`) creates a shared `NvmlGpuInfo()` instance
and passes it to all checks via `check_fn(verbose=verbose, gpu_info=gpu_info)`.
Third-party plugins safely ignore the extra keyword argument via their
own `**kwargs`.

## Files Changed

### New file: `rapids_cli/hardware.py`

Contains all provider abstractions:

- `DeviceInfo` dataclass with fields: `index`, `compute_capability`,
  `memory_total_bytes`, `nvlink_states`
- `GpuInfoProvider` and `SystemInfoProvider` protocols
  (runtime-checkable)
- `NvmlGpuInfo` -- calls `nvmlInit()` once on first property access,
  queries all device info (count, compute capability, memory,
  NVLink states), and caches everything
- `DefaultSystemInfo` -- lazily loads system memory via psutil and
  CUDA path via cuda.pathfinder (each cached independently)
- `FakeGpuInfo`, `FakeSystemInfo` -- `@dataclass` test fakes with
  pre-set data
- `FailingGpuInfo`, `FailingSystemInfo` -- test fakes that raise
  `ValueError` on any property access

### Modified: `rapids_cli/doctor/checks/gpu.py`

- Removed `import pynvml`
- Added `gpu_info: GpuInfoProvider | None = None` parameter and
  `**kwargs` to both `gpu_check()` and `check_gpu_compute_capability()`
- Replaced direct `pynvml` calls with `gpu_info.device_count` and
  iteration over `gpu_info.devices`

### Modified: `rapids_cli/doctor/checks/cuda_driver.py`

- Removed `import pynvml`
- Added `gpu_info` parameter and `**kwargs` to `cuda_check()`
- Replaced nested try/except with `gpu_info.cuda_driver_version`

### Modified: `rapids_cli/doctor/checks/memory.py`

- Removed `import pynvml` and `import psutil`
- Added `system_info` parameter to `get_system_memory()`
- Added `gpu_info` parameter to `get_gpu_memory()`
- Added both `gpu_info` and `system_info` parameters to
  `check_memory_to_gpu_ratio()`
- `get_system_memory()` reads `system_info.total_memory_bytes`
- `get_gpu_memory()` sums `dev.memory_total_bytes` from
  `gpu_info.devices`
- `check_memory_to_gpu_ratio()` passes injected providers down
  to helpers

### Modified: `rapids_cli/doctor/checks/nvlink.py`

- Removed `import pynvml`
- Added `gpu_info` parameter and `**kwargs` to `check_nvlink_status()`
- Iterates `dev.nvlink_states` instead of calling
  `nvmlDeviceGetNvLinkState`
- **Side-fix**: the original code always passed `0` instead of
  `nvlink_id` to `nvmlDeviceGetNvLinkState`; the refactored
  `NvmlGpuInfo` queries each link by its actual index

### Modified: `rapids_cli/debug/debug.py`

- Removed `import pynvml` and `import cuda.pathfinder`
- Added `gpu_info` parameter to `gather_cuda_version()`
- Added `gpu_info` and `system_info` parameters to `run_debug()`
- Replaced direct pynvml/cuda.pathfinder calls with provider
  property accesses

### Modified: `rapids_cli/doctor/doctor.py`

- Imports `NvmlGpuInfo` from `rapids_cli.hardware`
- Creates a shared `NvmlGpuInfo()` instance before the check loop
- Passes it via `check_fn(verbose=verbose, gpu_info=gpu_info)`

### Rewritten tests

`test_gpu.py`, `test_cuda.py`, `test_memory.py`, `test_nvlink.py`,
`test_debug.py`:

- Replaced all `patch("pynvml.*")` / `patch("psutil.*")` /
  `patch("cuda.pathfinder.*")` with `FakeGpuInfo` / `FakeSystemInfo` /
  `FailingGpuInfo` construction
- Tests for `debug.py` still use patches for non-hardware concerns
  (subprocess, pathlib, gather_tools)

### New file: `rapids_cli/tests/test_hardware.py`

- Unit tests for `NvmlGpuInfo`
  (init failure, loads once, device data, NVLink states, no NVLink)
- Unit tests for `DefaultSystemInfo`
  (total memory, CUDA runtime path, caching)
- Tests for `FakeGpuInfo` / `FakeSystemInfo`
  (defaults, custom values, protocol satisfaction)
- Tests for `FailingGpuInfo` / `FailingSystemInfo`
  (all properties raise)

## Impact

| Metric                                        | Before  | After                             |
| --------------------------------------------- | ------- | --------------------------------- |
| Hardware library patches in check/debug tests | ~51     | 0 (moved to test_hardware.py)     |
| import pynvml in check/debug modules          | 5 files | 1 file (hardware.py)              |
| MagicMock objects for hardware                | ~11     | 0                                 |
| pynvml.nvmlInit() calls in production         | 7       | 1 (in NvmlGpuInfo._ensure_loaded) |
| Total tests                                   | 53      | 72 (+19 hardware tests)           |
| Coverage                                      | 95%+    | 97.72%                            |

## Verification

1. `pytest` -- all 72 tests pass
2. `pytest --cov-fail-under=95` -- coverage at 97.72%, above threshold
3. `pre-commit run --all-files` -- all checks pass
