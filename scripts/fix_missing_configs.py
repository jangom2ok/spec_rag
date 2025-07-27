#!/usr/bin/env python3
"""Script to fix missing config parameters in test files"""

import re
from pathlib import Path

# Define the mapping of classes to their required config imports
CLASS_CONFIG_MAPPING = {
    "AdminDashboard": (
        "DashboardConfig",
        "from app.services.admin_dashboard import DashboardConfig",
    ),
    "DocumentChunker": (
        "ChunkingConfig",
        "from app.services.document_chunker import ChunkingConfig",
    ),
    "AlertingService": (
        "AlertConfig",
        "from app.services.alerting_service import AlertConfig",
    ),
    "HybridSearchEngine": (
        "HybridSearchConfig",
        "from app.services.hybrid_search_engine import HybridSearchConfig",
    ),
    "LoggingAnalysisService": (
        "LogAnalysisConfig",
        "from app.services.logging_analysis import LogAnalysisConfig",
    ),
}


def fix_missing_configs(file_path: Path):
    """Fix missing config parameters in a file"""
    content = file_path.read_text()
    modified = False

    for class_name, (config_class, import_stmt) in CLASS_CONFIG_MAPPING.items():
        # Find instantiations without config
        pattern = rf"{class_name}\(\s*\)"
        matches = list(re.finditer(pattern, content))

        if matches:
            modified = True
            # Add import if not present
            if import_stmt not in content and config_class not in content:
                # Find the right place to add import
                import_section_end = content.find("\n\n")
                if import_section_end > 0:
                    content = (
                        content[:import_section_end]
                        + f"\n{import_stmt}"
                        + content[import_section_end:]
                    )

            # Replace instantiations
            for match in reversed(matches):  # Reverse to maintain positions
                start, end = match.span()
                # Create default config
                replacement = f"{class_name}({config_class}())"
                content = content[:start] + replacement + content[end:]

    if modified:
        file_path.write_text(content)
        print(f"Fixed {file_path}")


# Fix test files
test_dir = Path("/Users/jangom2ok/work/git/spec_rag/tests")
for test_file in test_dir.glob("test_*.py"):
    fix_missing_configs(test_file)
