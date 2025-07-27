#!/usr/bin/env python3
"""Fix ChunkResult import to use correct module"""

from pathlib import Path


def fix_chunkresult_imports():
    """Fix ChunkResult import errors"""
    files_to_fix = [
        "tests/test_api_search_missing_coverage.py",
        "tests/test_services_missing_coverage.py",
    ]

    for file_path in files_to_fix:
        filepath = Path(file_path)
        if not filepath.exists():
            print(f"Skipping {file_path} - file not found")
            continue

        content = filepath.read_text()

        # Replace incorrect import
        wrong_import = "from app.models.documents import ChunkResult"
        correct_import = "from app.services.document_chunker import ChunkResult"

        if wrong_import in content:
            content = content.replace(wrong_import, correct_import)
            filepath.write_text(content)
            print(f"✓ Fixed ChunkResult import in {file_path}")
        else:
            print(f"  No incorrect import found in {file_path}")


if __name__ == "__main__":
    fix_chunkresult_imports()
