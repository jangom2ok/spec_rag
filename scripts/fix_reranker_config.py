#!/usr/bin/env python3
"""Fix RerankerConfig instantiation in tests"""

import re
from pathlib import Path


def fix_reranker_config():
    """Fix RerankerConfig instantiation to include reranker_type"""

    filepath = Path("tests/test_services_missing_coverage.py")
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return

    content = filepath.read_text()

    # Fix RerankerConfig instantiations
    patterns = [
        (
            r'RerankerConfig\(model_name="ms-marco-MiniLM-L-6-v2"\)',
            'RerankerConfig(reranker_type=RerankerType.CROSS_ENCODER, model_name="ms-marco-MiniLM-L-6-v2")',
        ),
        (
            r"RerankerConfig\(\)",
            "RerankerConfig(reranker_type=RerankerType.CROSS_ENCODER)",
        ),
    ]

    # Add RerankerType import if needed
    if "RerankerType" not in content:
        # Find the RerankerConfig import line
        reranker_import_pattern = r"from app\.services\.reranker import (.+)"
        match = re.search(reranker_import_pattern, content)
        if match:
            imports = match.group(1)
            if "RerankerType" not in imports:
                new_imports = imports.rstrip() + ", RerankerType"
                content = re.sub(
                    reranker_import_pattern,
                    f"from app.services.reranker import {new_imports}",
                    content,
                )

    # Apply the patterns
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    filepath.write_text(content)
    print("✓ Fixed RerankerConfig instantiations")


if __name__ == "__main__":
    fix_reranker_config()
