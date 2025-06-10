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
│ doctor    Run health checks to ensure RAPIDS is installed correctly.                │
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

## Developing

```bash
# Install and activate a RAPIDS environment E.g
conda create -y -n rapids-24.12 -c rapidsai -c conda-forge -c nvidia rapids=24.12 python=3.12 'cuda-version>=12.0,<=12.5'
conda activate rapids-24.12

# Install the RAPIDS CLI
git clone git@github.com:rapidsai/rapids-cli.git
cd rapids-cli
pip install -e .

# Run rapids doctor
rapids doctor
# 🧑‍⚕️ Performing REQUIRED health check for RAPIDS
# All checks passed!
```

## Testing

```bash
# Install test dependencies
pip install -e .[test]

# Run pytest
pytest
# ========================= test session starts ==========================
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
