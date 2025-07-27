#!/usr/bin/env python3
"""Fix ChunkResult usage in test files"""

import re
from pathlib import Path


def fix_chunkresult_usage():
    """Fix ChunkResult indexing and len() errors"""

    fixes = {
        "test_services_missing_coverage.py": [
            # Fix len(chunks) to len(chunks.chunks)
            {
                "pattern": r"assert len\(chunks\) >= 2",
                "replacement": "assert len(chunks.chunks) >= 2",
            },
            # Fix chunks[i] to chunks.chunks[i]
            {
                "pattern": r"for i in range\(len\(chunks\) - 1\):",
                "replacement": "for i in range(len(chunks.chunks) - 1):",
            },
            {
                "pattern": r'chunk1_end = chunks\[i\]\["content"\]\[-20:\]',
                "replacement": 'chunk1_end = chunks.chunks[i]["content"][-20:]',
            },
            {
                "pattern": r'chunks\[i \+ 1\]\["content"\]\[:20\]',
                "replacement": 'chunks.chunks[i + 1]["content"][:20]',
            },
            {
                "pattern": r'assert chunk1_end in chunks\[i \+ 1\]\["content"\]',
                "replacement": 'assert chunk1_end in chunks.chunks[i + 1]["content"]',
            },
            # Fix for chunk in chunks:
            {
                "pattern": r"for chunk in chunks:",
                "replacement": "for chunk in chunks.chunks:",
            },
            # Fix any(... for chunk in chunks)
            {
                "pattern": r'assert any\("hierarchy_level" in chunk\.get\("metadata", {}\) for chunk in chunks\)',
                "replacement": 'assert any("hierarchy_level" in chunk.get("metadata", {}) for chunk in chunks.chunks)',
            },
        ],
        "test_remaining_coverage.py": [
            # Similar fixes if needed
            {
                "pattern": r"assert len\(chunks\) >= 1",
                "replacement": "assert len(chunks.chunks) >= 1",
            },
        ],
    }

    tests_dir = Path("tests")

    for filename, file_fixes in fixes.items():
        filepath = tests_dir / filename
        if not filepath.exists():
            print(f"Skipping {filename} - file not found")
            continue

        print(f"Fixing ChunkResult usage in {filename}")
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

    print("\nChunkResult usage fixed!")


if __name__ == "__main__":
    fix_chunkresult_usage()
