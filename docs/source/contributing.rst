.. SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Contributing Guide
==================

Thank you for your interest in contributing to RAPIDS CLI! This guide will help you get started.

Getting Started
---------------

Prerequisites
^^^^^^^^^^^^^

- Python 3.10 or later
- Git
- NVIDIA GPU (for testing)
- NVIDIA drivers and CUDA toolkit

Development Setup
^^^^^^^^^^^^^^^^^

1. Fork and clone the repository:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/rapids-cli.git
      cd rapids-cli

2. Create a development environment:

   .. code-block:: bash

      # Using conda (recommended)
      conda create -n rapids-cli-dev python=3.10
      conda activate rapids-cli-dev

      # Or using venv
      python -m venv venv
      source venv/bin/activate

3. Install in editable mode with test dependencies:

   .. code-block:: bash

      pip install -e .[test]

4. Install pre-commit hooks:

   .. code-block:: bash

      pre-commit install

Development Workflow
--------------------

Making Changes
^^^^^^^^^^^^^^

1. Create a feature branch:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. Make your changes following the code style guidelines

3. Add tests for your changes

4. Run tests locally:

   .. code-block:: bash

      pytest

5. Run linting checks:

   .. code-block:: bash

      pre-commit run --all-files

Code Style
----------

The project uses several linting tools to maintain code quality:

Formatting
^^^^^^^^^^

- **Black**: Code formatting (120 char line length)
- **isort**: Import sorting

Linting
^^^^^^^

- **Ruff**: Fast Python linter (replaces flake8, pylint, etc.)
- **mypy**: Static type checking

Run formatters and linters:

.. code-block:: bash

   # Format code
   black .

   # Check with ruff
   ruff check --fix .

   # Type check
   mypy rapids_cli/

Docstrings
^^^^^^^^^^

Use Google-style docstrings:

.. code-block:: python

   def my_function(param1: str, param2: int) -> bool:
       """Brief description of the function.

       Longer description if needed.

       Args:
           param1: Description of param1.
           param2: Description of param2.

       Returns:
           Description of return value.

       Raises:
           ValueError: Description of when this is raised.

       Example:
           >>> my_function("test", 42)
           True
       """
       pass

Testing
-------

Writing Tests
^^^^^^^^^^^^^

- Place tests in ``rapids_cli/tests/``
- Use pytest for testing
- Mock external dependencies (pynvml, subprocess calls, etc.)
- Aim for high coverage (95%+ required)

Test Structure:

.. code-block:: python

   # rapids_cli/tests/test_my_feature.py
   from unittest.mock import patch, MagicMock
   import pytest

   from rapids_cli.my_module import my_function


   def test_my_function_success():
       """Test that my_function works in normal case."""
       result = my_function("input")
       assert result == "expected"


   def test_my_function_failure():
       """Test that my_function handles errors correctly."""
       with pytest.raises(ValueError, match="error message"):
           my_function("invalid")


   def test_my_function_with_mock():
       """Test my_function with mocked dependencies."""
       with patch("pynvml.nvmlInit") as mock_init:
           result = my_function()
           mock_init.assert_called_once()

Running Tests
^^^^^^^^^^^^^

.. code-block:: bash

   # Run all tests
   pytest

   # Run specific test file
   pytest rapids_cli/tests/test_doctor.py

   # Run with coverage
   pytest --cov=rapids_cli

   # Run specific test
   pytest rapids_cli/tests/test_doctor.py::test_doctor_check_all_pass

Pull Request Process
--------------------

1. Ensure all tests pass and coverage is maintained

2. Update documentation if needed

3. Sign your commits:

   .. code-block:: bash

      git commit -s -m "Your commit message"

4. Push to your fork:

   .. code-block:: bash

      git push origin feature/your-feature-name

5. Create a pull request on GitHub

6. Address review feedback

Commit Messages
^^^^^^^^^^^^^^^

Follow conventional commit format:

.. code-block:: text

   <type>: <short summary>

   <detailed description>

   Signed-off-by: Your Name <your.email@example.com>

Types:

- ``feat``: New feature
- ``fix``: Bug fix
- ``docs``: Documentation changes
- ``test``: Adding or updating tests
- ``refactor``: Code refactoring
- ``ci``: CI/CD changes
- ``chore``: Maintenance tasks

Example:

.. code-block:: text

   feat: add support for filtering checks by package name

   This allows users to run only specific checks by providing
   filter arguments to the doctor command.

   Signed-off-by: Jane Doe <jane@example.com>

Documentation
-------------

Building Documentation
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd docs
   make html

   # View in browser
   open build/html/index.html

Documentation lives in ``docs/source/`` and uses Sphinx with reStructuredText.

Adding New Documentation
^^^^^^^^^^^^^^^^^^^^^^^^

1. Create ``.rst`` file in ``docs/source/``

2. Add to table of contents in ``index.rst``

3. Build and verify:

   .. code-block:: bash

      cd docs
      make html

Reporting Issues
----------------

When reporting bugs:

1. Check if issue already exists

2. Provide minimal reproduction example

3. Include debug output:

   .. code-block:: bash

      rapids debug --json > debug_info.json

4. Include:

   - RAPIDS CLI version
   - Python version
   - OS and driver versions
   - Expected vs actual behavior

Feature Requests
^^^^^^^^^^^^^^^^

For feature requests:

1. Describe the use case

2. Explain why existing features don't work

3. Provide example usage

4. Consider contributing the feature!

Code Review Guidelines
----------------------

For Reviewers
^^^^^^^^^^^^^

- Check that tests cover new functionality
- Verify documentation is updated
- Ensure code style is consistent
- Look for potential edge cases
- Validate error messages are helpful

For Contributors
^^^^^^^^^^^^^^^^

- Respond to feedback promptly
- Ask questions if feedback is unclear
- Keep PRs focused on single concern
- Update based on reviews

Release Process
---------------

Releases are managed by maintainers:

1. Version is managed via git tags
2. CI automatically builds packages
3. Packages published to PyPI and conda-forge

Community
---------

- GitHub Discussions: Q&A and ideas
- Slack: Real-time chat at rapids.ai/community
- Issues: Bug reports and features

License
-------

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

Thank You!
----------

Every contribution helps make RAPIDS CLI better. Thank you for your time and effort!
