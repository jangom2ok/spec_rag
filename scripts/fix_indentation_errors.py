#!/usr/bin/env python3
"""Fix indentation errors in test files"""

import re
from pathlib import Path


def fix_indentation():
    """Fix indentation errors caused by try/except blocks"""

    fixes = [
        # Fix the try/except blocks that are not properly indented
        {
            "pattern": r'from app\.services\.metrics_collection import MetricsCollectionService\ntry:\n    from app\.services\.metrics_collection import MetricsConfig\nexcept ImportError:\n    # MetricsConfig might not exist\n    MetricsConfig = type\("MetricsConfig", \(\), {}\)\n\n        config = MetricsConfig\(\)',
            "replacement": """from app.services.metrics_collection import MetricsCollectionService
        try:
            from app.services.metrics_collection import MetricsConfig
        except ImportError:
            # MetricsConfig might not exist
            MetricsConfig = type("MetricsConfig", (), {})

        config = MetricsConfig()""",
        },
        {
            "pattern": r'from app\.services\.logging_analysis import LoggingAnalysisService\ntry:\n    from app\.services\.logging_analysis import LogAnalysisConfig\nexcept ImportError:\n    # LogAnalysisConfig might not exist\n    LogAnalysisConfig = type\("LogAnalysisConfig", \(\), {}\)\n\n        analyzer = LoggingAnalysisService',
            "replacement": """from app.services.logging_analysis import LoggingAnalysisService
        try:
            from app.services.logging_analysis import LogAnalysisConfig
        except ImportError:
            # LogAnalysisConfig might not exist
            LogAnalysisConfig = type("LogAnalysisConfig", (), {})

        analyzer = LoggingAnalysisService""",
        },
    ]

    filepath = Path("tests/test_services_missing_coverage.py")
    content = filepath.read_text()

    print("Fixing indentation errors...")

    for fix in fixes:
        pattern = fix["pattern"]
        replacement = fix["replacement"]

        # Count matches before replacement
        matches = len(re.findall(pattern, content, re.MULTILINE | re.DOTALL))
        if matches > 0:
            content = re.sub(
                pattern, replacement, content, flags=re.MULTILINE | re.DOTALL
            )
            print(f"  - Fixed {matches} occurrence(s)")

    filepath.write_text(content)
    print("Indentation errors fixed!")


if __name__ == "__main__":
    fix_indentation()
