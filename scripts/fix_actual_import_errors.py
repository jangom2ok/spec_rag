#!/usr/bin/env python3
"""Fix actual import errors found in test runs"""

import re
from pathlib import Path


def fix_imports():
    """Fix real import errors in test files"""

    fixes = {
        "test_services_missing_coverage.py": [
            # Fix AlertConfig -> AlertingConfig
            {
                "pattern": r"from app\.services\.alerting_service import AlertConfig",
                "replacement": "from app.services.alerting_service import AlertingConfig as AlertConfig",
            },
            # Fix HybridSearchConfig import
            {
                "pattern": r"from app\.services\.hybrid_search_engine import HybridSearchEngine\n",
                "replacement": "from app.services.hybrid_search_engine import HybridSearchEngine, HybridSearchConfig\n",
            },
            # Check if we need to add other missing imports
        ],
        "test_remaining_coverage.py": [
            # Fix any imports in this file if needed
        ],
        "test_api_search_missing_coverage.py": [
            # Fix imports if needed
        ],
    }

    tests_dir = Path("tests")

    for filename, file_fixes in fixes.items():
        filepath = tests_dir / filename
        if not filepath.exists():
            print(f"Skipping {filename} - file not found")
            continue

        print(f"Fixing imports in {filename}")
        content = filepath.read_text()

        for fix in file_fixes:
            pattern = fix["pattern"]
            replacement = fix["replacement"]

            # Count matches before replacement
            matches = len(re.findall(pattern, content))
            if matches > 0:
                content = re.sub(pattern, replacement, content)
                print(f"  - Fixed {matches} occurrence(s) of: {pattern[:50]}...")

        filepath.write_text(content)

    print("\nImport errors fixed!")


if __name__ == "__main__":
    fix_imports()
