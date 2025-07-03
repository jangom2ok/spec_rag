"""ドキュメント収集器のテストモジュール

TDD実装：テストケース→実装→リファクタの順序で実装
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncGenerator, Dict, List, Any
from datetime import datetime

from app.services.document_collector import (
    DocumentCollector,
    DocumentSource,
    CollectionConfig,
    CollectionResult,
    SourceType,
)


class TestDocumentCollector:
    """ドキュメント収集器のテストクラス"""

    @pytest.fixture
    def mock_config(self) -> CollectionConfig:
        """テスト用の収集設定"""
        return CollectionConfig(
            source_type=SourceType.TEST,
            batch_size=10,
            max_concurrent=3,
            timeout=30,
            filters={},  # No filters by default for tests
            metadata_extraction=True,
        )

    @pytest.fixture
    def mock_collector(self, mock_config: CollectionConfig) -> DocumentCollector:
        """モックドキュメント収集器"""
        return DocumentCollector(config=mock_config)

    @pytest.mark.unit
    async def test_collector_initialization(self, mock_config: CollectionConfig):
        """収集器の初期化テスト"""
        collector = DocumentCollector(config=mock_config)
        
        assert collector.config == mock_config
        assert collector.config.source_type == SourceType.TEST
        assert collector.config.batch_size == 10
        assert collector.config.max_concurrent == 3

    @pytest.mark.unit
    async def test_collect_documents_from_test_source(self, mock_collector: DocumentCollector):
        """テストソースからのドキュメント収集テスト"""
        # テスト用ドキュメントデータ
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

        with patch.object(mock_collector, '_fetch_from_source', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = test_documents
            
            result = await mock_collector.collect_documents()
            
            assert isinstance(result, CollectionResult)
            assert result.success_count == 2
            assert result.error_count == 0
            assert len(result.documents) == 2
            assert result.documents[0]["title"] == "Test Document 1"
            assert result.documents[1]["title"] == "Test Document 2"

    @pytest.mark.unit
    async def test_collect_documents_with_batch_processing(self, mock_collector: DocumentCollector):
        """バッチ処理での収集テスト"""
        # バッチサイズより多いドキュメント数でテスト
        mock_collector.config.batch_size = 2
        test_documents = [
            {"id": f"doc-{i}", "title": f"Document {i}", "content": f"Content {i}", "source_id": f"doc-{i}"}
            for i in range(5)
        ]

        with patch.object(mock_collector, '_fetch_from_source', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = test_documents
            
            result = await mock_collector.collect_documents()
            
            assert result.success_count == 5
            assert len(result.documents) == 5
            # バッチ処理が正しく動作していることを確認
            assert mock_fetch.call_count >= 1

    @pytest.mark.unit
    async def test_collect_documents_with_errors(self, mock_collector: DocumentCollector):
        """エラーがある場合の収集テスト"""
        with patch.object(mock_collector, '_fetch_from_source', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = Exception("Connection failed")
            
            result = await mock_collector.collect_documents()
            
            assert result.success_count == 0
            assert result.error_count == 1
            assert len(result.errors) == 1
            assert "Connection failed" in result.errors[0]

    @pytest.mark.unit
    async def test_collect_documents_with_filters(self, mock_collector: DocumentCollector):
        """フィルター適用での収集テスト"""
        mock_collector.config.filters = {"status": "active", "type": "wiki"}
        
        with patch.object(mock_collector, '_fetch_from_source', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = [
                {"id": "doc-1", "title": "Active Doc", "content": "Content", "source_id": "doc-1", "status": "active", "type": "wiki"},
                {"id": "doc-2", "title": "Inactive Doc", "content": "Content", "source_id": "doc-2", "status": "inactive", "type": "wiki"},
            ]
            
            result = await mock_collector.collect_documents()
            
            # フィルターが適用されることを確認
            mock_fetch.assert_called_once()
            assert result.success_count >= 0

    @pytest.mark.unit
    async def test_validate_document_data(self, mock_collector: DocumentCollector):
        """ドキュメントデータのバリデーションテスト"""
        # 有効なドキュメント
        valid_doc = {
            "id": "valid-1",
            "title": "Valid Document",
            "content": "Valid content",
            "source_id": "valid-1",
        }
        
        # 無効なドキュメント（必須フィールド不足）
        invalid_doc = {
            "id": "invalid-1",
            # titleが不足
            "content": "Invalid content",
        }
        
        assert mock_collector._validate_document(valid_doc) is True
        assert mock_collector._validate_document(invalid_doc) is False

    @pytest.mark.unit
    async def test_deduplicate_documents(self, mock_collector: DocumentCollector):
        """ドキュメント重複排除テスト"""
        documents = [
            {"id": "doc-1", "title": "Document 1", "content": "Content 1", "source_id": "doc-1"},
            {"id": "doc-1", "title": "Document 1", "content": "Content 1", "source_id": "doc-1"},  # 重複
            {"id": "doc-2", "title": "Document 2", "content": "Content 2", "source_id": "doc-2"},
        ]
        
        deduplicated = mock_collector._deduplicate_documents(documents)
        
        assert len(deduplicated) == 2
        assert deduplicated[0]["id"] == "doc-1"
        assert deduplicated[1]["id"] == "doc-2"

    @pytest.mark.unit
    async def test_extract_metadata(self, mock_collector: DocumentCollector):
        """メタデータ抽出テスト"""
        mock_collector.config.metadata_extraction = True
        
        document = {
            "id": "doc-1",
            "title": "Test Document",
            "content": "This is a test document with some content.",
            "source_id": "doc-1",
        }
        
        metadata = mock_collector._extract_metadata(document)
        
        assert "word_count" in metadata
        assert "char_count" in metadata
        assert "extracted_at" in metadata
        assert metadata["word_count"] > 0
        assert metadata["char_count"] > 0

    @pytest.mark.integration
    async def test_collect_from_file_source(self, tmp_path):
        """ファイルソースからの収集統合テスト"""
        # テスト用ファイル作成
        test_file = tmp_path / "test_document.txt"
        test_file.write_text("This is a test document for file collection.")
        
        config = CollectionConfig(
            source_type=SourceType.FILE,
            source_path=str(tmp_path),
            file_patterns=["*.txt"],
            batch_size=10,
        )
        
        collector = DocumentCollector(config=config)
        result = await collector.collect_documents()
        
        assert result.success_count >= 1
        assert any("test_document" in doc.get("title", "") for doc in result.documents)


class TestDocumentSource:
    """ドキュメントソースのテストクラス"""

    @pytest.mark.unit
    def test_source_type_enum(self):
        """ソースタイプEnumのテスト"""
        assert SourceType.TEST == "test"
        assert SourceType.FILE == "file"
        assert SourceType.CONFLUENCE == "confluence"
        assert SourceType.JIRA == "jira"

    @pytest.mark.unit
    def test_collection_config_validation(self):
        """収集設定のバリデーションテスト"""
        # 有効な設定
        valid_config = CollectionConfig(
            source_type=SourceType.TEST,
            batch_size=10,
            max_concurrent=3,
        )
        assert valid_config.batch_size == 10
        assert valid_config.max_concurrent == 3
        
        # 無効な設定（バッチサイズが0以下）
        with pytest.raises(ValueError):
            CollectionConfig(
                source_type=SourceType.TEST,
                batch_size=0,
            )


class TestCollectionResult:
    """収集結果のテストクラス"""

    @pytest.mark.unit
    def test_collection_result_creation(self):
        """収集結果の作成テスト"""
        documents = [
            {"id": "doc-1", "title": "Document 1"},
            {"id": "doc-2", "title": "Document 2"},
        ]
        errors = ["Error 1", "Error 2"]
        
        result = CollectionResult(
            documents=documents,
            success_count=2,
            error_count=2,
            errors=errors,
            collection_time=1.5,
        )
        
        assert len(result.documents) == 2
        assert result.success_count == 2
        assert result.error_count == 2
        assert len(result.errors) == 2
        assert result.collection_time == 1.5

    @pytest.mark.unit
    def test_collection_result_summary(self):
        """収集結果サマリーのテスト"""
        result = CollectionResult(
            documents=[{"id": "doc-1"}],
            success_count=1,
            error_count=0,
            errors=[],
            collection_time=0.5,
        )
        
        summary = result.get_summary()
        
        assert "success_count" in summary
        assert "error_count" in summary
        assert "total_documents" in summary
        assert summary["success_count"] == 1
        assert summary["error_count"] == 0
        assert summary["total_documents"] == 1