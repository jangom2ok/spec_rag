#!/usr/bin/env python3
"""Fix RerankerService interface usage in tests"""

import re
from pathlib import Path


def fix_reranker_interface():
    """Fix RerankerService.rerank() calls to use RerankRequest"""

    filepath = Path("tests/test_services_missing_coverage.py")
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return

    content = filepath.read_text()

    # Add RerankRequest import to RerankerService imports
    content = re.sub(
        r"from app\.services\.reranker import RerankerService, RerankerConfig, RerankerType",
        "from app.services.reranker import RerankerService, RerankerConfig, RerankerType, RerankRequest",
        content,
    )

    # Fix the rerank calls
    # Pattern 1: rerank(query, documents)
    content = re.sub(
        r'result = await reranker\.rerank\("query", \[\]\)',
        'result = await reranker.rerank(RerankRequest(query="query", documents=[]))',
        content,
    )

    # Pattern 2: rerank(query, single_doc)
    content = re.sub(
        r'result = await reranker\.rerank\("query", single_doc\)',
        'result = await reranker.rerank(RerankRequest(query="query", documents=single_doc))',
        content,
    )

    # Pattern 3: rerank(query, documents) in test
    content = re.sub(
        r"result = await reranker\.rerank\(query, documents\)",
        "result = await reranker.rerank(RerankRequest(query=query, documents=documents))",
        content,
    )

    # Fix result access patterns
    # Change result.chunks to result.documents
    content = re.sub(
        r'result\.chunks if hasattr\(result, "chunks"\) else \[\]',
        'result.documents if hasattr(result, "documents") else []',
        content,
    )

    filepath.write_text(content)
    print("✓ Fixed RerankerService interface usage")


if __name__ == "__main__":
    fix_reranker_interface()
