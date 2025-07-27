#!/usr/bin/env python3
"""Fix service class names in test files"""

import re
from pathlib import Path


def fix_service_names():
    """Fix incorrect service class names"""

    fixes = {
        "test_services_missing_coverage.py": [
            # Fix DiversityOptimizer -> SearchDiversityService
            {
                "pattern": r"from app\.services\.search_diversity import DiversityOptimizer",
                "replacement": "from app.services.search_diversity import SearchDiversityService as DiversityOptimizer",
            },
            # AdminDashboard is correct, no change needed
            # Fix SearchSuggestionsService import
            {
                "pattern": r"from app\.services\.search_suggestions import SearchSuggestionsService\n",
                "replacement": "from app.services.search_suggestions import SearchSuggestionsService\n",
            },
        ]
    }

    tests_dir = Path("tests")

    for filename, file_fixes in fixes.items():
        filepath = tests_dir / filename
        if not filepath.exists():
            print(f"Skipping {filename} - file not found")
            continue

        print(f"Fixing service class names in {filename}")
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

    print("\nService class names fixed!")


if __name__ == "__main__":
    fix_service_names()
