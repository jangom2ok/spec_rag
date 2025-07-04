"""ドキュメント収集器サービス

TDD実装：テストケースに基づいたドキュメント収集機能
"""

import asyncio
import glob
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import aiofiles

logger = logging.getLogger(__name__)


class SourceType(str, Enum):
    """ドキュメントソースタイプ"""

    TEST = "test"
    FILE = "file"
    CONFLUENCE = "confluence"
    JIRA = "jira"


@dataclass
class CollectionConfig:
    """ドキュメント収集設定"""

    source_type: SourceType
    batch_size: int = 10
    max_concurrent: int = 3
    timeout: int = 30
    filters: dict[str, Any] = field(default_factory=dict)
    metadata_extraction: bool = True
    source_path: str | None = None
    file_patterns: list[str] = field(default_factory=lambda: ["*.txt", "*.md"])

    def __post_init__(self):
        """設定値のバリデーション"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if self.max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than 0")


@dataclass
class CollectionResult:
    """ドキュメント収集結果"""

    documents: list[dict[str, Any]]
    success_count: int
    error_count: int
    errors: list[str]
    collection_time: float

    def get_summary(self) -> dict[str, Any]:
        """収集結果のサマリーを取得"""
        return {
            "total_documents": len(self.documents),
            "success_count": self.success_count,
            "error_count": self.error_count,
            "collection_time": self.collection_time,
            "success_rate": (
                self.success_count / (self.success_count + self.error_count)
                if (self.success_count + self.error_count) > 0
                else 0
            ),
        }


class DocumentSource:
    """ドキュメントソース基底クラス"""

    def __init__(self, config: CollectionConfig):
        self.config = config

    async def fetch_documents(self) -> list[dict[str, Any]]:
        """ドキュメントを取得（サブクラスで実装）"""
        raise NotImplementedError


class TestDocumentSource(DocumentSource):
    """テスト用ドキュメントソース"""

    async def fetch_documents(self) -> list[dict[str, Any]]:
        """テスト用ドキュメントを生成"""
        test_documents = [
            {
                "id": "test-1",
                "title": "Test Document 1",
                "content": "This is test content 1",
                "source_id": "test-1",
                "metadata": {"author": "test_user", "tags": ["test"]},
            },
            {
                "id": "test-2",
                "title": "Test Document 2",
                "content": "This is test content 2",
                "source_id": "test-2",
                "metadata": {"author": "test_user", "tags": ["test", "sample"]},
            },
        ]
        return test_documents


class FileDocumentSource(DocumentSource):
    """ファイルシステムドキュメントソース"""

    async def fetch_documents(self) -> list[dict[str, Any]]:
        """ファイルシステムからドキュメントを取得"""
        if not self.config.source_path:
            raise ValueError("source_path is required for file source")

        documents = []
        source_path = Path(self.config.source_path)

        for pattern in self.config.file_patterns:
            file_paths = glob.glob(str(source_path / pattern))

            for file_path in file_paths:
                try:
                    async with aiofiles.open(file_path, encoding="utf-8") as f:
                        content = await f.read()

                    file_name = Path(file_path).name
                    document = {
                        "id": hashlib.sha256(file_path.encode()).hexdigest(),
                        "title": file_name,
                        "content": content,
                        "source_id": file_path,
                        "file_type": Path(file_path).suffix,
                        "file_path": file_path,
                    }
                    documents.append(document)

                except Exception as e:
                    logger.error(f"Failed to read file {file_path}: {e}")
                    continue

        return documents


class ExternalSourceDocumentSource(DocumentSource):
    """外部ソース統合ドキュメントソース"""

    async def fetch_documents(self) -> list[dict[str, Any]]:
        """外部ソースからドキュメントを取得"""
        from app.services.external_source_integration import (
            ExternalSourceIntegrator,
            SourceConfig as ExtSourceConfig,
            SourceType as ExtSourceType,
            AuthType,
        )

        # CollectionConfigからSourceConfigに変換
        ext_source_type = ExtSourceType.CONFLUENCE
        if self.config.source_type == SourceType.CONFLUENCE:
            ext_source_type = ExtSourceType.CONFLUENCE
        elif self.config.source_type == SourceType.JIRA:
            ext_source_type = ExtSourceType.JIRA
        else:
            raise ValueError(f"Unsupported external source type: {self.config.source_type}")

        # 設定を構築（実際の実装では環境変数から取得）
        ext_config = ExtSourceConfig(
            source_type=ext_source_type,
            base_url=self.config.filters.get("base_url", "https://example.atlassian.net"),
            auth_type=AuthType.API_TOKEN,
            api_token=self.config.filters.get("api_token", "dummy_token"),
            username=self.config.filters.get("username", "user@example.com"),
            max_pages=self.config.filters.get("max_pages", 100),
            timeout=self.config.timeout,
        )

        # 外部ソース統合器を使用してドキュメントを取得
        integrator = ExternalSourceIntegrator(config=ext_config)
        result = await integrator.fetch_documents()

        if result.success:
            return result.documents
        else:
            logger.error(f"External source integration failed: {result.error_message}")
            return []


class DocumentCollector:
    """ドキュメント収集器メインクラス"""

    def __init__(self, config: CollectionConfig):
        self.config = config
        self._source = self._create_source()

    def _create_source(self) -> DocumentSource:
        """ソースタイプに応じたドキュメントソースを作成"""
        if self.config.source_type == SourceType.TEST:
            return TestDocumentSource(self.config)
        elif self.config.source_type == SourceType.FILE:
            return FileDocumentSource(self.config)
        elif self.config.source_type in [SourceType.CONFLUENCE, SourceType.JIRA]:
            return ExternalSourceDocumentSource(self.config)
        else:
            raise ValueError(f"Unsupported source type: {self.config.source_type}")

    async def collect_documents(self) -> CollectionResult:
        """ドキュメントを収集"""
        start_time = datetime.now()
        documents = []
        errors = []
        success_count = 0
        error_count = 0

        try:
            # ソースからドキュメントを取得
            raw_documents = await self._fetch_from_source()

            # バッチ処理で収集
            for i in range(0, len(raw_documents), self.config.batch_size):
                batch = raw_documents[i : i + self.config.batch_size]
                batch_results = await self._process_batch(batch)

                for result in batch_results:
                    if result.get("error"):
                        errors.append(result["error"])
                        error_count += 1
                    else:
                        documents.append(result)
                        success_count += 1

            # 重複排除
            documents = self._deduplicate_documents(documents)

        except Exception as e:
            logger.error(f"Collection failed: {e}")
            errors.append(str(e))
            error_count += 1

        end_time = datetime.now()
        collection_time = (end_time - start_time).total_seconds()

        return CollectionResult(
            documents=documents,
            success_count=success_count,
            error_count=error_count,
            errors=errors,
            collection_time=collection_time,
        )

    async def _fetch_from_source(self) -> list[dict[str, Any]]:
        """ソースからドキュメントを取得"""
        return await self._source.fetch_documents()

    async def _process_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """バッチを処理"""
        results = []

        # セマフォを使用して同時実行数を制限
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def process_document(doc: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await self._process_single_document(doc)

        # 並行処理でドキュメントを処理
        tasks = [process_document(doc) for doc in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 例外を処理
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({"error": str(result)})
            else:
                processed_results.append(result)

        return processed_results

    async def _process_single_document(
        self, document: dict[str, Any]
    ) -> dict[str, Any]:
        """単一ドキュメントを処理"""
        try:
            # バリデーション
            if not self._validate_document(document):
                return {"error": f"Invalid document: {document.get('id', 'unknown')}"}

            # フィルター適用
            if not self._apply_filters(document):
                return {
                    "error": f"Document filtered out: {document.get('id', 'unknown')}"
                }

            # メタデータ抽出
            if self.config.metadata_extraction:
                metadata = self._extract_metadata(document)
                document["extracted_metadata"] = metadata

            return document

        except Exception as e:
            return {
                "error": f"Failed to process document {document.get('id', 'unknown')}: {e}"
            }

    def _validate_document(self, document: dict[str, Any]) -> bool:
        """ドキュメントのバリデーション"""
        required_fields = ["id", "title", "content"]
        return all(field in document for field in required_fields)

    def _apply_filters(self, document: dict[str, Any]) -> bool:
        """フィルターを適用"""
        if not self.config.filters:
            return True

        for key, expected_value in self.config.filters.items():
            if key not in document:
                return False

            if document[key] != expected_value:
                return False

        return True

    def _deduplicate_documents(
        self, documents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """ドキュメントの重複を排除"""
        seen_ids: set[str] = set()
        deduplicated = []

        for doc in documents:
            doc_id = doc.get("id")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                deduplicated.append(doc)

        return deduplicated

    def _extract_metadata(self, document: dict[str, Any]) -> dict[str, Any]:
        """ドキュメントからメタデータを抽出"""
        content = document.get("content", "")

        metadata = {
            "word_count": len(content.split()),
            "char_count": len(content),
            "extracted_at": datetime.now().isoformat(),
        }

        # 既存のメタデータがあれば結合
        existing_metadata = document.get("metadata", {})
        if existing_metadata:
            metadata.update(existing_metadata)

        return metadata
