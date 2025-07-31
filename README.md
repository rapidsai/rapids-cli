# RAPIDS CLI

The RAPIDS CLI is a command line tool for performing common RAPIDS operations
in a quick and scriptable way.

```console
$ rapids --help

 Usage: rapids [OPTIONS] COMMAND [ARGS]...

 The Rapids CLI is a command-line interface for RAPIDS.

╭─ Options ───────────────────────────────────────────────────────────────────────────╮
│ --help      Show this message and exit.                                             │
╰─────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────╮
│ doctor      Run health checks to ensure RAPIDS is installed correctly.              │
│ benchmark   Run performance benchmarks comparing CPU vs GPU implementations.        │
╰─────────────────────────────────────────────────────────────────────────────────────╯
```

## RAPIDS Doctor

The `rapids doctor` subcommand performs health checks on installations of RAPIDS to ensure versions of the
driver, CUDA, and other compiled dependencies are compatible with each other.

The core command has checks based on the standard [setup requirements](https://docs.rapids.ai/install).
Other packages in the RAPIDS ecosystem can also register additional checks via a plugin system.

```console
$ rapids doctor
🧑‍⚕️ Performing REQUIRED health check for RAPIDS
All checks passed!
```

## RAPIDS Benchmark

The `rapids benchmark` subcommand runs performance benchmarks comparing CPU and GPU implementations of common
data processing operations. This helps users understand the potential speedup they can achieve by using RAPIDS
on their specific hardware configuration.

The command currently includes a benchmark for DataFrame operations like joins and aggregations on cuDF vs Pandas.
Other packages in the RAPIDS ecosystem can also register additional benchmarks via a plugin system.

```console
$ rapids benchmark
🏁 Running RAPIDS CPU vs GPU benchmarks (5 runs each)

✓ [1/1] dataframe_join_operations
  CPU Time: 28.567s ± 1.234s  GPU Time: 3.891s ± 0.156s  Speedup: 7.3x

                                ⚡ CPU vs GPU Performance Comparison
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Benchmark                    ┃ Description              ┃ CPU Time       ┃ GPU Time       ┃ Speedup        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ dataframe_join_operations    │ DataFrame join ops       │ 28.567s ± 1.2s │ 3.891s ± 0.16s │   7.3x faster  │
└──────────────────────────────┴──────────────────────────┴────────────────┴────────────────┴────────────────┘
```

## Developing

```bash
# Install and activate a RAPIDS environment E.g
conda create -y -n rapids-25.06 -c rapidsai -c conda-forge rapids=25.06 python=3.13 'cuda-version>=12.0,<=12.8'
conda activate rapids-25.06

# Install the RAPIDS CLI
git clone git@github.com:rapidsai/rapids-cli.git
cd rapids-cli
pip install -e .

# Run rapids doctor
rapids doctor
# 🧑‍⚕️ Performing REQUIRED health check for RAPIDS
# All checks passed!

# Run rapids benchmark
rapids benchmark
# 🏁 Running RAPIDS CPU vs GPU benchmarks (5 runs each)
# ✓ [1/1] dataframe_join_operations
# ...
```

## Testing

```bash
# Install test dependencies
pip install -e .[test]

# Run pytest
pytest
# ========================= test session starts ==========================
# platform linux -- Python 3.12.8, pytest-8.3.4, pluggy-1.5.0
# rootdir: /home/jtomlinson/Projects/rapids/rapids-cli
# configfile: pyproject.toml
# plugins: anyio-4.8.0
# collected 2 items
#
# rapids_cli/tests/test_cuda.py ..
# ========================== 2 passed in 0.08s ==========================
```

## Check plugins

Any project can add checks to `rapids doctor` by exposing a function via an entrypoint.

These checks would live in existing RAPIDS libraries like `cudf`, `cuml` or `cugraph`,
but for an example let's create a summy package.

```console
$ hatch new doctor-check-example
doctor-check-example
├── src
│   └── doctor_check_example
│       ├── __about__.py
│       └── __init__.py
├── tests
│   └── __init__.py
├── LICENSE.txt
├── README.md
└── pyproject.toml

$ cd doctor-check-example

```

Then let's create a new file in our package and create a simple check function.

```python
# src/doctor_check_example/check.py
def my_awesome_check(**kwargs):
    """A quick check to ensure we can import cudf and create a GPU memory resource."""
    import cudf

    s = cudf.Series([1, 2, 3, None, 4])
    assert isinstance(s, cudf.core.series.Series)
```

Then we need to register this function with `rapids doctor` by adding an entrypoint to `pyproject.toml`.

```toml
# pyproject.toml
# ...

[project.entry-points.rapids_doctor_check]
quick_cudf_check = "doctor_check_example.check:my_awesome_check"
```

Now we can install our new package, which will register the entrypoint.

```console
$ pip install -e .
Successfully installed doctor-check-example-0.0.1
```

Then if we run `rapids doctor` with the `--verbose` flag we can see our new check is discovered and included in our
list of checks.

```console
$ rapids doctor --verbose
🧑‍⚕️ Performing REQUIRED health check for RAPIDS
Discovering checks
Found check 'quick_cudf_check' provided by 'doctor_check_example.check:my_awesome_check'
...
Discovered 14 checks
Running checks
All checks passed!
```

### Check status

If a check function returns successfully it is assumed the check was successful, regardless of what was returned.

To fail a test an exception should be raised, ideally with a helpful message that gives the user actionable
next steps to resolve the problem.
It may be helpful to catch a common exception and reraise it with a more helpful error message.

```python
# src/doctor_check_example/check.py
def my_awesome_check(**kwargs):
    """A quick check to ensure we can import cudf and create a GPU memory resource."""
    try:
        import cudf
    except ImportError as e:
        raise ImportError(
            "Module cudf not found. Tip: you can install it with `pip install cudf-cu12`"
        ) from e
    s = cudf.Series([1, 2, 3, None, 4])
    assert isinstance(s, cudf.core.series.Series)
```

### Warnings

You can also raise warnings from within your test. These will not cause the test to fail,
but warnings will be caught and presented to the user.

```console
$ rapids doctor
🧑‍⚕️ Performing REQUIRED health check for RAPIDS
Warning: System Memory to total GPU Memory ratio not at least 2:1 ratio.
It is recommended to have double the system memory to GPU memory for optimal performance.
All checks passed!
```

### Check keyword arguments

When calling a check function a number of keyword arguments will be passed.
The list of keywords may change over time so it is recommended to always pack unknown kwargs with `**kwargs`
to ensure forward compatibility.

```python
def my_awesome_check(**kwargs):
    pass
```

Keyword arguments may be added in the future, but never removed, so it's safe to explicitly accept keywords you
know you want to use.

```python
def my_awesome_check(
    verbose=False,  # Has the --verbose flag been set
    **kwargs,
):
    if verbose:
        print("Print additional messages in here")
```

## Benchmark plugins

Any project can add benchmarks to `rapids benchmark` by exposing a function via an entrypoint, similar to how
check plugins work for `rapids doctor`.

Here's a simple example benchmark function:

```python
# src/benchmark_example/benchmark.py
import time
import numpy as np


def my_array_benchmark(verbose=False):
    """Benchmark array operations comparing NumPy vs CuPy."""
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError("numpy is required for this benchmark") from e

    try:
        import cupy as cp
    except ImportError as e:
        raise ImportError("cupy is required for GPU benchmarking") from e

    # Prepare data
    size = 10_000_000
    data = np.random.randn(size).astype(np.float32)

    if verbose:
        print(f"Running array operations on {size:,} elements")

    # CPU timing
    start = time.time()
    cpu_array = np.array(data)
    result_cpu = np.sum(cpu_array**2) + np.mean(cpu_array)
    cpu_time = time.time() - start

    # GPU timing (excluding data transfer)
    gpu_array = cp.array(data)  # Transfer data (not timed)
    _ = len(gpu_array)  # GPU warmup
    start = time.time()
    result_gpu = cp.sum(gpu_array**2) + cp.mean(gpu_array)
    gpu_time = time.time() - start

    if verbose:
        print(f"CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s")

    return cpu_time, gpu_time
```

Register this function by adding an entrypoint to the `rapids_benchmark` group in `pyproject.toml`:

```toml
[project.entry-points.rapids_benchmark]
array_operations = "benchmark_example.benchmark:my_array_benchmark"
```

### Benchmark Options

The benchmark command supports several options to customize execution:

```console
$ rapids benchmark --help

 Usage: rapids benchmark [OPTIONS] [FILTERS]...

 Run performance benchmarks comparing CPU vs GPU implementations.

╭─ Options ───────────────────────────────────────────────────────────────────────────╮
│ --verbose     Enable verbose mode for detailed timing output.                       │
│ --dry-run     Perform a dry run without running benchmarks.                         │
│ --runs        Number of iterations to run each benchmark (default: 5).              │
│ --help        Show this message and exit.                                           │
╰─────────────────────────────────────────────────────────────────────────────────────╯
```

**Examples:**

```console
# Run with custom number of iterations
$ rapids benchmark --runs 10

# Run only specific benchmarks
$ rapids benchmark dataframe

# See detailed timing information
$ rapids benchmark --verbose

# Check what benchmarks are available without running them
$ rapids benchmark --dry-run --verbose
```

### Benchmark function guidelines

Benchmark functions should follow this pattern:

1. **Function signature**: Accept `**kwargs` for forward compatibility
2. **Return value**: Return a tuple of `(cpu_time, gpu_time)` in seconds
3. **Timing**: Handle your own timing within the function
4. **Data transfer**: Exclude GPU data transfer from timing for fair comparison
5. **Error handling**: Fail gracefully with actionable error messages such ImportError messages for missing dependencies
6. **GPU Warmup**: Also consider running a small warmup operation before timing to avoid initialization overhead
