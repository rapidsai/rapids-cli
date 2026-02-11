# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The RAPIDS CLI is a command-line tool for performing common RAPIDS operations, primarily focused on
health checks (`rapids doctor`) and debugging (`rapids debug`). It uses a plugin system that allows
RAPIDS libraries to register their own health checks via Python entry points.

## Common Commands

### Development Setup

```bash
# Install in editable mode
pip install -e .

# Install with test dependencies
pip install -e .[test]
```

### Testing

```bash
# Run all tests (coverage reporting is automatic via pyproject.toml)
pytest

# Run a specific test file
pytest rapids_cli/tests/test_cuda.py

# Run a specific test function
pytest rapids_cli/tests/test_cuda.py::test_function_name

# Generate coverage report without running tests
coverage report

# View detailed HTML coverage report
coverage html && open htmlcov/index.html
```

### Documentation

```bash
# Build HTML documentation
cd docs && make html

# View documentation
open docs/build/html/index.html

# Clean build artifacts
cd docs && make clean
```

### Linting and Pre-commit

```bash
# Install pre-commit hooks
pre-commit install

# Run all pre-commit checks
pre-commit run --all-files

# Individual linters
black .                    # Format code
ruff check --fix .         # Lint with ruff
mypy rapids_cli/          # Type checking
```

### Running the CLI

```bash
# Run doctor checks
rapids doctor
rapids doctor --verbose
rapids doctor --dry-run

# Run debug command
rapids debug
rapids debug --json
```

## Architecture

### CLI Structure

- **Entry point**: `rapids_cli/cli.py` defines the main CLI group and subcommands using rich-click
- **Doctor command**: `rapids_cli/doctor/doctor.py` contains the health check orchestration logic
- **Debug command**: `rapids_cli/debug/debug.py` gathers system/environment information
- **Checks**: Individual checks live in `rapids_cli/doctor/checks/` (gpu.py, cuda_driver.py, memory.py,
  nvlink.py)

### Plugin System

The doctor command discovers and runs checks via Python entry points defined in `pyproject.toml`:

- Entry point group: `rapids_doctor_check`
- Built-in checks are registered in `[project.entry-points.rapids_doctor_check]`
- External packages can register additional checks by adding their own entry points
- Check functions receive `verbose` kwarg and should accept `**kwargs` for forward compatibility
- Checks pass by returning successfully (any return value) and fail by raising exceptions
- Checks can issue warnings using Python's `warnings.warn()` which are caught and displayed

### Check Function Contract

- Accept `verbose=False` and `**kwargs` parameters
- Raise exceptions with helpful error messages for failures
- Return successfully for passing checks (return value is optional string for verbose output)
- Use `warnings.warn()` for non-fatal issues

### Key Dependencies

- `rich` and `rich-click` for terminal output and CLI interface
- `pynvml` (nvidia-ml-py) for GPU information
- `cuda-pathfinder` for locating CUDA installations
- `psutil` for system memory checks

### Configuration

- Package configuration in `pyproject.toml` (build system, dependencies, entry points)
- CLI settings in `rapids_cli/config.yml` (loaded via `config.py`)
- Dependencies managed via `dependencies.yaml` and `rapids-dependency-file-generator`

## Code Style

- Python 3.10+ (minimum version)
- Line length: 120 characters
- Use Google-style docstrings (enforced by ruff with pydocstyle convention)
- Enforce type hints (checked by mypy)
- SPDX license headers required on all files (enforced by pre-commit hook)
- All commits must be signed off with `-s` flag

## Testing Notes

Tests are located in `rapids_cli/tests/`. The test suite runs quickly with 53 tests covering all
modules. GPU-based tests run in CI on actual GPU hardware (L4 instances).

### Coverage Requirements

- Minimum coverage threshold: **95%**
- Coverage is automatically measured when running `pytest`
- Coverage reports are generated in XML format for CI and terminal format for local development
- Test files and `_version.py` are excluded from coverage measurements

## CI/CD

- Pre-commit checks run on all PRs (black, ruff, mypy, shellcheck, etc.)
- Builds both conda packages (noarch: python) and wheels (pure Python)
- Tests run on GPU nodes with CUDA available
- Uses RAPIDS shared workflows for build and test automation
