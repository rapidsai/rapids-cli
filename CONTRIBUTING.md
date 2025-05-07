# RAPIDS CLI Contribution Guidance

Contributions to the RAPIDS CLI are very welcome!

## Developing

This project is written in Python and uses `pytest` for testing.

### Developer environment

We recommend you start with a Python environment containing the [latest RAPIDS release](https://docs.rapids.ai/install/).

Then you can install the RAPIDS CLI from source.

```bash
pip install -e .

# Install test dependencies
pip install -e .[test]
```

### Testing

Tests are run with `pytest`.

```shell
$ pytest
==================== test session starts =====================
platform linux -- Python 3.12.10, pytest-8.3.5, pluggy-1.5.0
rootdir: /home/jtomlinson/Projects/rapids/rapids-cli
configfile: pyproject.toml
plugins: anyio-4.9.0
collected 2 items                                            

rapids_cli/tests/test_cuda.py ..                       [100%]

===================== 2 passed in 0.07s ======================
```

## Linting

This project enforces linting with `black` and `ruff`. You can use [pre-commit](https://pre-commit.com/)
to check this automatically on commit, which will save time as you can catch linting errors before the CI does.

```console
$ pre-commit install
pre-commit installed at .git/hooks/pre-commit

$ pre-commit run --all-files
```

## Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original
work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

  ```bash
  git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the
    open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate
    open source license and I have the right under that license to submit that work with modifications, whether created in
    whole or in part by me, under the same open source license (unless I am permitted to submit under a different license),
    as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and 
    I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution
    (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be
    redistributed consistent with this project or the open source license(s) involved.
  ```
