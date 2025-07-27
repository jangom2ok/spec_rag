# Cursor Settings Alignment

This document ensures Cursor IDE settings are aligned with the project's code quality tools.

## Tool Versions

Ensure the following tool versions match what's specified in `requirements-ci.txt`:
- Black: 23.11.0
- Ruff: 0.1.6
- mypy: 1.7.1

## Configuration Files

The project uses the following configuration files that Cursor should respect:

### pyproject.toml
- Ruff configuration (line length: 88, Python 3.11 target)
- Black configuration (line length: 88)
- Bandit security checks exclusions

### .pre-commit-config.yaml
- Pre-commit hooks for Black, Ruff, mypy, and bandit
- Automatic fixes enabled for Ruff

### mypy.ini
- Type checking configuration for the app/ directory
- Ignore missing imports enabled

## Recommended Cursor Settings

Add these to your Cursor settings to align with the project:

```json
{
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=88"],
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.linting.ruffArgs": ["--line-length=88"],
  "python.linting.mypyEnabled": true,
  "python.linting.mypyArgs": ["--ignore-missing-imports"],
  "editor.formatOnSave": true,
  "editor.rulers": [88]
}
```

## Running Code Quality Checks

Use the provided script to run all checks:
```bash
./scripts/check_code_quality.sh
```

Or use the Python version:
```bash
python scripts/check_code_quality.py
```

## Pre-commit Integration

Ensure pre-commit is installed:
```bash
pre-commit install
```

This will automatically run all checks before each commit.
