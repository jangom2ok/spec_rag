#!/usr/bin/env python3
"""
Fix remaining Cursor errors - comprehensive solution
"""

import re
from pathlib import Path


def fix_test_services_missing_coverage_comprehensive(file_path: Path):
    """Comprehensive fix for test_services_missing_coverage.py"""
    content = file_path.read_text(encoding="utf-8")

    # Fix all undefined variables by checking usage context
    lines = content.split("\n")
    fixed_lines = []

    for i, line in enumerate(lines):
        # Fix undefined variables used with patch.object
        if "patch.object(" in line and ', "' in line:
            # Extract the variable name being patched
            match = re.search(r"patch\.object\((\w+),", line)
            if match:
                var_name = match.group(1)
                # Check if this variable was defined with _ prefix
                if f"_{var_name} =" in "\n".join(lines[max(0, i - 10) : i]):
                    line = line.replace(
                        f"patch.object({var_name},", f"patch.object(_{var_name},"
                    )

        # Fix suggestions/collector/etc references
        if 'assert "' in line and " in " in line:
            # Check if variable is undefined
            match = re.search(r" in (\w+)(?:\s|$)", line)
            if match:
                var_name = match.group(1)
                if f"_{var_name} =" in "\n".join(lines[max(0, i - 10) : i]):
                    line = line.replace(f" in {var_name}", f" in _{var_name}")

        # Fix .assert_called references
        if ".assert_called" in line:
            match = re.search(r"(\w+)\.assert_called", line)
            if match:
                var_name = match.group(1)
                if f"_{var_name} =" in "\n".join(lines[max(0, i - 10) : i]):
                    line = line.replace(
                        f"{var_name}.assert_called", f"_{var_name}.assert_called"
                    )

        fixed_lines.append(line)

    content = "\n".join(fixed_lines)

    # Fix specific undefined variables
    # Fix collector references after _collector assignment
    content = re.sub(
        r"(\s+_collector = DocumentCollector[^\n]+\n[^\n]*\n\s+)result = await collector\.",
        r"\1result = await _collector.",
        content,
    )

    # Fix dashboard references
    content = re.sub(
        r"with patch\.object\(dashboard,", r"with patch.object(_dashboard,", content
    )

    # Fix service references
    content = re.sub(
        r"with patch\.object\(service,", r"with patch.object(_service,", content
    )

    # Fix engine references
    content = re.sub(
        r"with patch\.object\(engine,", r"with patch.object(_engine,", content
    )

    # Fix analyzer references
    content = re.sub(r"await analyzer\.", r"await _analyzer.", content)

    # Fix expander references
    content = re.sub(
        r"with patch\.object\(expander,", r"with patch.object(_expander,", content
    )

    # Fix suggestions in assertions
    content = re.sub(
        r'assert "([^"]+)" in suggestions', r'assert "\1" in _suggestions', content
    )
    content = re.sub(
        r"assert any\([^)]+\) for s in suggestions\)",
        r"assert any(\1) for s in _suggestions)",
        content,
    )

    # Fix collector.get_metrics() calls
    content = re.sub(
        r"metrics = collector\.get_metrics\(\)",
        r"metrics = _collector.get_metrics()",
        content,
    )

    file_path.write_text(content, encoding="utf-8")
    print(f"Fixed {file_path.name}")


def fix_test_aperturedb_mock(file_path: Path):
    """Fix test_aperturedb_mock.py"""
    content = file_path.read_text(encoding="utf-8")

    # Fix result.chunks references - result is supposed to be the return value from query
    # Replace (result.chunks if hasattr(result, "chunks") else []) with just result
    content = re.sub(
        r'\(result\.chunks if hasattr\(result, "chunks"\) else \[\]\)',
        "result",
        content,
    )

    # Fix specific assertions that check result length
    content = re.sub(
        r"assert len\(result\) == (\d+)", r"assert len(result) == \1", content
    )

    file_path.write_text(content, encoding="utf-8")
    print(f"Fixed {file_path.name}")


def fix_main_py_handlers(file_path: Path):
    """Fix unused exception handlers in main.py"""
    content = file_path.read_text(encoding="utf-8")

    # The handlers are defined but not registered with the app
    # We need to either remove them or register them

    # Option 1: Register the handlers by adding them to create_app
    if "def create_app" in content:
        # Find the create_app function and add exception handler registrations
        lines = content.split("\n")
        fixed_lines = []
        in_create_app = False
        app_registered = False

        for i, line in enumerate(lines):
            if "def create_app" in line:
                in_create_app = True
            elif in_create_app and "return app" in line and not app_registered:
                # Add exception handler registrations before return
                indent = "    "
                registrations = f"""
{indent}# Register exception handlers
{indent}app.add_exception_handler(DatabaseException, database_exception_handler)
{indent}app.add_exception_handler(VectorDatabaseException, vector_database_exception_handler)
{indent}app.add_exception_handler(AuthenticationException, authentication_exception_handler)
{indent}app.add_exception_handler(RAGSystemException, rag_system_exception_handler)
{indent}app.add_exception_handler(Exception, general_exception_handler)
"""
                fixed_lines.append(registrations)
                app_registered = True

            fixed_lines.append(line)

        content = "\n".join(fixed_lines)

    file_path.write_text(content, encoding="utf-8")
    print(f"Fixed {file_path.name}")


def fix_health_py_import(file_path: Path):
    """Fix DBException import in health.py"""
    content = file_path.read_text(encoding="utf-8")

    # The DBException import is missing - add proper try/except
    if "from aperturedb import DBException" not in content:
        # Add after other imports
        import_section_end = 0
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("from ") or line.startswith("import "):
                import_section_end = i + 1

        # Insert the import with try/except
        import_code = '''
try:
    from aperturedb import DBException
except ImportError:
    # Mock DBException if aperturedb is not available
    class DBException(Exception):
        """Mock ApertureDB exception."""
        pass
'''
        lines.insert(import_section_end, import_code)
        content = "\n".join(lines)

    file_path.write_text(content, encoding="utf-8")
    print(f"Fixed {file_path.name}")


def main():
    """Main function"""
    project_root = Path(__file__).parent.parent

    print("Fixing remaining Cursor errors...")
    print("=" * 60)

    # Fix specific files with targeted solutions
    files_to_fix = [
        (
            project_root / "tests" / "test_services_missing_coverage.py",
            fix_test_services_missing_coverage_comprehensive,
        ),
        (project_root / "tests" / "test_aperturedb_mock.py", fix_test_aperturedb_mock),
        (project_root / "app" / "main.py", fix_main_py_handlers),
        (project_root / "app" / "api" / "health.py", fix_health_py_import),
    ]

    for file_path, fix_func in files_to_fix:
        if file_path.exists():
            try:
                fix_func(file_path)
            except Exception as e:
                print(f"Error fixing {file_path.name}: {e}")
        else:
            print(f"File not found: {file_path}")

    print("\n" + "=" * 60)
    print("Fixes completed!")


if __name__ == "__main__":
    main()
