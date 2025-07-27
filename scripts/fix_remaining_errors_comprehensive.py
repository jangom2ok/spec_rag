#!/usr/bin/env python3
"""
Comprehensive fix for all remaining Cursor errors
"""

import re
from pathlib import Path


def fix_main_py_comprehensive(file_path: Path):
    """Fix all issues in main.py"""
    content = file_path.read_text(encoding="utf-8")

    # First, ensure DBException is properly imported/defined
    if "from aperturedb import DBException" in content:
        # Replace the import section with proper try/except
        content = re.sub(
            r"try:\s*\n\s*from aperturedb import DBException.*?pass",
            '''try:
    from aperturedb import DBException
except ImportError:
    class DBException(Exception):
        """Mock ApertureDB exception."""
        pass''',
            content,
            flags=re.DOTALL,
        )

    # Add custom exception definitions and handlers
    exception_code = '''
# Custom exception classes
class DatabaseException(Exception):
    """Database-related exception."""
    pass


class VectorDatabaseException(Exception):
    """Vector database exception."""
    pass


class AuthenticationException(Exception):
    """Authentication exception."""
    pass


class RAGSystemException(Exception):
    """RAG system exception."""
    pass


# Exception handlers
async def database_exception_handler(request: Request, exc: DatabaseException) -> JSONResponse:
    """Handle database exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "database_error",
                "message": str(exc),
                "code": "DB_ERROR",
            },
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4()),
        }
    )


async def vector_database_exception_handler(request: Request, exc: VectorDatabaseException) -> JSONResponse:
    """Handle vector database exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "vector_db_error",
                "message": str(exc),
                "code": "VECTOR_DB_ERROR",
            },
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4()),
        }
    )


async def authentication_exception_handler(request: Request, exc: AuthenticationException) -> JSONResponse:
    """Handle authentication exceptions."""
    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "type": "authentication_error",
                "message": str(exc),
                "code": "AUTH_ERROR",
            },
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4()),
        }
    )


async def rag_system_exception_handler(request: Request, exc: RAGSystemException) -> JSONResponse:
    """Handle RAG system exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "rag_system_error",
                "message": str(exc),
                "code": "RAG_ERROR",
            },
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4()),
        }
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "internal_error",
                "message": "An unexpected error occurred",
                "code": "INTERNAL_ERROR",
            },
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4()),
        }
    )


'''

    # Find where to insert the exception code (after imports, before create_app)
    create_app_pos = content.find("def create_app")
    if create_app_pos != -1:
        # Find the last import or class definition before create_app
        lines_before = content[:create_app_pos].split("\n")
        insert_pos = 0
        for i in range(len(lines_before) - 1, -1, -1):
            line = lines_before[i].strip()
            if line and not line.startswith("#"):
                insert_pos = len("\n".join(lines_before[: i + 1])) + 1
                break

        # Insert the exception code
        content = content[:insert_pos] + "\n" + exception_code + content[insert_pos:]

    # Remove any duplicate exception handler registrations
    content = re.sub(
        r"\n\s*# Register exception handlers\n\s*app\.add_exception_handler.*?\n",
        "\n",
        content,
        flags=re.DOTALL,
    )

    file_path.write_text(content, encoding="utf-8")


def fix_test_services_comprehensive(file_path: Path):
    """Fix all remaining issues in test_services_missing_coverage.py"""
    content = file_path.read_text(encoding="utf-8")

    # Fix _dashboard usage issue
    content = re.sub(
        r'if hasattr\(dashboard, "_get_usage_data"\)',
        r'if hasattr(_dashboard, "_get_usage_data")',
        content,
    )

    # Fix undefined collector in various test methods
    content = re.sub(
        r"(\s+_collector = DocumentCollector[^\n]+\n)([^\n]+\n)*?(\s+# Test collecting[^\n]+\n\s+result = await )collector\.",
        r"\1\2\3_collector.",
        content,
        flags=re.MULTILINE,
    )

    file_path.write_text(content, encoding="utf-8")


def fix_health_py_comprehensive(file_path: Path):
    """Fix health.py completely"""
    content = file_path.read_text(encoding="utf-8")

    # Find the imports section
    lines = content.split("\n")
    import_end = 0
    for i, line in enumerate(lines):
        if (
            line.strip()
            and not line.startswith("from")
            and not line.startswith("import")
            and not line.startswith("#")
        ):
            import_end = i
            break

    # Insert proper DBException handling after imports
    dbexception_import = '''
# Handle DBException import
try:
    from aperturedb import DBException
except ImportError:
    class DBException(Exception):
        """Mock ApertureDB exception."""
        pass
'''

    lines.insert(import_end, dbexception_import)
    content = "\n".join(lines)

    # Remove any duplicate or problematic DBException definitions
    content = re.sub(r"\nDBException = DBException\n", "\n", content)

    file_path.write_text(content, encoding="utf-8")


def fix_production_config_comprehensive(file_path: Path):
    """Fix production_config.py Client import"""
    content = file_path.read_text(encoding="utf-8")

    # Fix asyncpg import
    if "import asyncpg" in content:
        content = re.sub(
            r"import asyncpg",
            """try:
    import asyncpg
except ImportError:
    asyncpg = None  # asyncpg not available""",
            content,
        )

    # Fix Client import for OpenTelemetry
    content = re.sub(
        r"from opentelemetry[^\n]+Client",
        """try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as Client
except ImportError:
    Client = None  # OpenTelemetry not available""",
        content,
    )

    file_path.write_text(content, encoding="utf-8")


def main():
    """Main function"""
    project_root = Path(__file__).parent.parent

    print("Comprehensive fix for remaining Cursor errors...")
    print("=" * 60)

    fixes = [
        ("app/main.py", fix_main_py_comprehensive),
        ("tests/test_services_missing_coverage.py", fix_test_services_comprehensive),
        ("app/api/health.py", fix_health_py_comprehensive),
        ("app/database/production_config.py", fix_production_config_comprehensive),
    ]

    for file_path_str, fix_func in fixes:
        file_path = project_root / file_path_str
        if file_path.exists():
            try:
                fix_func(file_path)
                print(f"✓ Fixed {file_path.name}")
            except Exception as e:
                print(f"✗ Error fixing {file_path.name}: {e}")
        else:
            print(f"- File not found: {file_path}")

    print("\n" + "=" * 60)
    print("Comprehensive fixes completed!")


if __name__ == "__main__":
    main()
