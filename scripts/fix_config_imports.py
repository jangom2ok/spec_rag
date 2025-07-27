#!/usr/bin/env python3
"""Fix incorrect config import names"""

import re
from pathlib import Path


def fix_config_imports():
    """Fix incorrect config import names"""

    # Mapping of incorrect to correct names
    fixes = {
        "ExpansionConfig": "QueryExpansionConfig",
        "RetrievalConfig": None,  # This doesn't exist - need to find alternative
        "BackupConfig": None,  # This doesn't exist
        "BatchProcessorConfig": "ProcessingConfig",
    }

    filepath = Path("tests/test_services_missing_coverage.py")
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return

    content = filepath.read_text()

    # Fix ExpansionConfig -> QueryExpansionConfig
    content = content.replace(
        "from app.services.query_expansion import ExpansionConfig",
        "from app.services.query_expansion import QueryExpansionConfig",
    )
    content = content.replace("ExpansionConfig()", "QueryExpansionConfig()")

    # Fix BatchProcessorConfig -> ProcessingConfig
    content = content.replace(
        "from app.services.batch_processor import BatchProcessorConfig",
        "from app.services.document_processing_service import ProcessingConfig",
    )
    content = content.replace("BatchProcessorConfig()", "ProcessingConfig()")

    # Remove non-existent imports
    lines = content.splitlines()
    new_lines = []
    for line in lines:
        if "from app.services.retrieval_enhancer import RetrievalConfig" in line:
            continue  # Skip this import
        if "from app.services.backup_restore import BackupConfig" in line:
            continue  # Skip this import
        new_lines.append(line)

    content = "\n".join(new_lines)

    # Fix service instantiations that use removed configs
    # For RetrievalEnhancer - check if it exists
    content = re.sub(
        r"service = RetrievalEnhancer\(RetrievalConfig\(\)\)",
        "service = RetrievalEnhancer()",  # Use no config
        content,
    )

    # For BackupRestoreService - check if it exists
    content = re.sub(
        r"service = BackupRestoreService\(BackupConfig\(\)\)",
        "service = BackupRestoreService()",  # Use no config
        content,
    )

    filepath.write_text(content)
    print("✓ Fixed config import names")


if __name__ == "__main__":
    fix_config_imports()
