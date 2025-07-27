# Fix Code Issues

Run the code quality checker and automatically fix all detected issues.

## Steps:

1. First, run the code issue collector to identify all problems:
   ```bash
   python scripts/collect_code_issues.py
   ```

2. If there are Black formatting issues:
   ```bash
   black app/ tests/
   ```

3. If there are Ruff linting issues:
   ```bash
   ruff check --fix app/ tests/
   ```

4. If there are mypy type checking issues:
   - Analyze each issue and fix type annotations
   - Add `type: ignore` comments where necessary
   - Use `TYPE_CHECKING` imports for conditional type imports

5. If there are pytest failures:
   - Run failing tests individually to understand the issues
   - Fix test expectations or implementation code as needed
   - Skip tests with known limitations (e.g., TestClient with Exception handlers)

6. After fixing all issues, run the collector again to verify:
   ```bash
   python scripts/collect_code_issues.py
   ```

7. If all checks pass, the code is ready for commit!

## Additional Commands:

- Run all checks continuously until passing: `./scripts/check_code_quality.sh`
- Run specific checks:
  - Black: `black --check app/ tests/`
  - Ruff: `ruff check app/ tests/`
  - Mypy: `mypy app/`
  - Pytest: `pytest`
