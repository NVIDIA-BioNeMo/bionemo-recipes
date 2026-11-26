# Contributing to BioNeMo Recipes

## Quick start

1. Check the [open issues](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/issues) and
   [pull requests](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/pulls) for related work.
2. Fork the repository and create a branch from `main`.
3. Make a focused change and run the appropriate [local checks](#local-checks).
4. Open a pull request against `main`. Explain the problem, your change, and the checks you ran.
   Link the relevant issue when there is one.

## Local checks

From the repository root, run the pre-commit hooks for formatting and repository conventions:

```bash
pre-commit run --all-files
```

If you changed code that is intentionally copied across directories, verify that the copies agree:

```bash
python ci/scripts/check_copied_files.py
```
