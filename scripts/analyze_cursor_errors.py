#!/usr/bin/env python3
"""Analyze errors from Cursor IDE and create fix scripts"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_cursor_errors(error_text):
    """Parse Cursor error text and extract structured information"""
    errors = []

    # Common error patterns
    patterns = {
        "import_error": r'Cannot import name ["\']([^"\']+)["\'] from ["\']([^"\']+)["\']',
        "attribute_error": r'["\']([^"\']+)["\'] has no attribute ["\']([^"\']+)["\']',
        "parameter_missing": r"Missing required parameter[s]?: ([^\n]+)",
        "type_error": r'Type ["\']([^"\']+)["\'] cannot be assigned to type ["\']([^"\']+)["\']',
        "undefined_var": r'["\']([^"\']+)["\'] is not defined',
        "no_member": r'Module ["\']([^"\']+)["\'] has no member ["\']([^"\']+)["\']',
    }

    lines = error_text.split("\n")
    current_file = None

    for line in lines:
        # Extract file path
        file_match = re.search(r"([^\s]+\.py):(\d+):(\d+)", line)
        if file_match:
            current_file = file_match.group(1)
            line_num = int(file_match.group(2))
            col_num = int(file_match.group(3))

            # Extract error message
            error_msg = line[file_match.end() :].strip()

            # Classify error
            error_type = "unknown"
            details = {}

            for err_type, pattern in patterns.items():
                match = re.search(pattern, error_msg)
                if match:
                    error_type = err_type
                    details = {"groups": match.groups()}
                    break

            errors.append(
                {
                    "file": current_file,
                    "line": line_num,
                    "column": col_num,
                    "type": error_type,
                    "message": error_msg,
                    "details": details,
                }
            )

    return errors


def group_errors_by_type(errors):
    """Group errors by type for batch fixing"""
    grouped = defaultdict(list)
    for error in errors:
        grouped[error["type"]].append(error)
    return grouped


def generate_fix_suggestions(grouped_errors):
    """Generate fix suggestions for each error type"""
    suggestions = []

    for error_type, errors in grouped_errors.items():
        if error_type == "import_error":
            suggestions.append(
                {
                    "type": "import_fixes",
                    "description": f"Fix {len(errors)} import errors",
                    "errors": errors,
                    "fix_approach": "Check if imported names exist, suggest alternatives",
                }
            )
        elif error_type == "attribute_error":
            suggestions.append(
                {
                    "type": "attribute_fixes",
                    "description": f"Fix {len(errors)} attribute errors",
                    "errors": errors,
                    "fix_approach": "Find correct attribute names or add missing attributes",
                }
            )
        elif error_type == "parameter_missing":
            suggestions.append(
                {
                    "type": "parameter_fixes",
                    "description": f"Fix {len(errors)} missing parameter errors",
                    "errors": errors,
                    "fix_approach": "Add required parameters to function calls",
                }
            )

    return suggestions


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_cursor_errors.py <error_file>")
        sys.exit(1)

    error_file = Path(sys.argv[1])
    if not error_file.exists():
        print(f"Error file not found: {error_file}")
        sys.exit(1)

    # Read error text
    error_text = error_file.read_text()

    # Parse errors
    errors = parse_cursor_errors(error_text)
    print(f"Found {len(errors)} errors")

    # Group by type
    grouped = group_errors_by_type(errors)

    # Generate suggestions
    suggestions = generate_fix_suggestions(grouped)

    # Save analysis
    output_file = Path("docs/cursor_error_analysis.json")
    output_file.parent.mkdir(exist_ok=True)

    analysis = {
        "total_errors": len(errors),
        "errors_by_type": {k: len(v) for k, v in grouped.items()},
        "errors": errors,
        "suggestions": suggestions,
    }

    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=2)

    print("\nError Analysis Summary:")
    for error_type, count in analysis["errors_by_type"].items():
        print(f"  {error_type}: {count}")

    print(f"\nAnalysis saved to: {output_file}")
    print("\nNext steps:")
    print("1. Review the analysis in docs/cursor_error_analysis.json")
    print("2. Run: python scripts/fix_cursor_errors.py")


if __name__ == "__main__":
    main()
