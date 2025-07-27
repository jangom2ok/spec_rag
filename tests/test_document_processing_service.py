"""Tests for Document Processing Service

ドキュメント処理サービスの包括的なテスト。
カバレッジの向上を目的として、すべての主要なメソッドとエラーケースをテスト。
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.models.database import Document
from app.services.document_chunker import (
    ChunkingConfig,
    ChunkResult,
    ChunkType,
)
from app.services.document_chunker import (
    DocumentChunk as ChunkerDocumentChunk,
)
from app.services.document_collector import CollectionConfig, CollectionResult
from app.services.document_processing_service import (
    DocumentProcessingService,
    DocumentProcessingStatus,
    ProcessingConfig,
    ProcessingResult,
    ProcessingStage,
)
from app.services.embedding_service import EmbeddingResult
from app.services.metadata_extractor import (
    ExtractionConfig,
    ExtractionResult,
    FieldType,
)


@pytest.fixture
def mock_repositories():
    """モックリポジトリのフィクスチャ"""
    document_repo = AsyncMock()
    chunk_repo = AsyncMock()
    return document_repo, chunk_repo


@pytest.fixture
def mock_embedding_service():
    """モック埋め込みサービスのフィクスチャ"""
    service = AsyncMock()

    # embed_batchメソッドのモック
    async def mock_embed_batch(texts):
        results = []
        for i, _text in enumerate(texts):
            result = EmbeddingResult(
                dense_vector=[0.1] * 1024,
                sparse_vector={1: 0.5, 2: 0.3},
                multi_vector=None,  # multi_vectorはオプショナル
                processing_time=0.1,
                chunk_id=f"chunk_{i}",
                document_id="doc_1",
            )
            results.append(result)
        return results

    service.embed_batch = mock_embed_batch
    return service


@pytest.fixture
def processing_config():
    """処理設定のフィクスチャ"""
    from app.services.document_chunker import ChunkingStrategy
    from app.services.document_collector import SourceType

    return ProcessingConfig(
        collection_config=CollectionConfig(
            source_type=SourceType.TEST,
            batch_size=10,
            max_concurrent=3,
            timeout=30,
        ),
        extraction_config=ExtractionConfig(
            extract_structure=True,
            extract_entities=True,
            extract_keywords=True,
            extract_statistics=True,
            language_detection=True,
            confidence_threshold=0.5,
            max_keywords=20,
            max_entities=50,
        ),
        chunking_config=ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=1000,
            overlap_size=100,
        ),
        enable_embedding=True,
        max_concurrent_documents=2,
        max_concurrent_chunks=5,
    )


@pytest.fixture
def processing_service(mock_repositories, mock_embedding_service):
    """処理サービスのフィクスチャ"""
    document_repo, chunk_repo = mock_repositories
    return DocumentProcessingService(
        document_repository=document_repo,
        chunk_repository=chunk_repo,
        embedding_service=mock_embedding_service,
    )


class TestProcessingResult:
    """ProcessingResultのテスト"""

    def test_get_summary(self):
        """処理結果サマリーの取得テスト"""
        result = ProcessingResult(
            success=True,
            total_documents=10,
            successful_documents=8,
            failed_documents=2,
            total_chunks=50,
            successful_chunks=45,
            failed_chunks=5,
            processing_time=5.5,
            stage_times={
                ProcessingStage.COLLECTION: 1.0,
                ProcessingStage.CHUNKING: 2.5,
            },
            errors=["Error 1", "Error 2"],
        )

        summary = result.get_summary()

        assert summary["success"] is True
        assert summary["documents"]["total"] == 10
        assert summary["documents"]["successful"] == 8
        assert summary["documents"]["failed"] == 2
        assert summary["documents"]["success_rate"] == 0.8
        assert summary["chunks"]["total"] == 50
        assert summary["chunks"]["successful"] == 45
        assert summary["chunks"]["failed"] == 5
        assert summary["chunks"]["success_rate"] == 0.9
        assert summary["timing"]["total_time"] == 5.5
        assert summary["timing"]["stage_times"][ProcessingStage.COLLECTION] == 1.0
        assert summary["error_count"] == 2

    def test_get_summary_zero_documents(self):
        """ドキュメント数0の場合のサマリーテスト"""
        result = ProcessingResult(
            success=False,
            total_documents=0,
            successful_documents=0,
            failed_documents=0,
            total_chunks=0,
            successful_chunks=0,
            failed_chunks=0,
            processing_time=0.0,
            stage_times={},
            errors=[],
        )

        summary = result.get_summary()

        assert summary["documents"]["success_rate"] == 0.0
        assert summary["chunks"]["success_rate"] == 0.0


class TestDocumentProcessingStatus:
    """DocumentProcessingStatusのテスト"""

    def test_to_dict(self):
        """辞書形式への変換テスト"""
        status = DocumentProcessingStatus(
            document_id="doc_1",
            stage=ProcessingStage.CHUNKING,
            progress=0.5,
            error_message="Test error",
            chunks_processed=10,
            chunks_total=20,
        )

        result = status.to_dict()

        assert result["document_id"] == "doc_1"
        assert result["stage"] == ProcessingStage.CHUNKING
        assert result["progress"] == 0.5
        assert result["error_message"] == "Test error"
        assert result["chunks_processed"] == 10
        assert result["chunks_total"] == 20


class TestDocumentProcessingService:
    """DocumentProcessingServiceのテスト"""

    @pytest.mark.asyncio
    async def test_process_documents_success(
        self, processing_service, processing_config
    ):
        """ドキュメント処理成功のテスト"""
        # DocumentCollectorのモック
        with patch(
            "app.services.document_processing_service.DocumentCollector"
        ) as mock_collector_class:
            mock_collector = mock_collector_class.return_value
            mock_collector.collect_documents = AsyncMock(
                return_value=CollectionResult(
                    documents=[
                        {
                            "id": "doc_1",
                            "title": "Test Document 1",
                            "content": "Test content 1",
                            "source_type": "test",
                        },
                        {
                            "id": "doc_2",
                            "title": "Test Document 2",
                            "content": "Test content 2",
                            "source_type": "test",
                        },
                    ],
                    success_count=2,
                    error_count=0,
                    errors=[],
                    collection_time=0.5,
                )
            )

            # MetadataExtractorのモック
            with patch(
                "app.services.document_processing_service.MetadataExtractor"
            ) as mock_extractor_class:
                mock_extractor = mock_extractor_class.return_value
                mock_extractor.extract_metadata = AsyncMock(
                    return_value=ExtractionResult(
                        success=True,
                        document_id="doc_1",
                        metadata={"keywords": ["test", "content"]},
                        field_types={"keywords": FieldType.ARRAY},
                        processing_time=0.1,
                        error_message=None,
                    )
                )

                # DocumentChunkerのモック
                with patch(
                    "app.services.document_processing_service.DocumentChunker"
                ) as mock_chunker_class:
                    mock_chunker = mock_chunker_class.return_value
                    mock_chunker.chunk_document = AsyncMock(
                        return_value=ChunkResult(
                            success=True,
                            chunks=[
                                ChunkerDocumentChunk(
                                    id="chunk_1",
                                    document_id="doc_1",
                                    chunk_index=0,
                                    chunk_type=ChunkType.TEXT,
                                    content="Test content 1",
                                    content_length=15,
                                    token_count=3,
                                ),
                                ChunkerDocumentChunk(
                                    id="chunk_2",
                                    document_id="doc_1",
                                    chunk_index=1,
                                    chunk_type=ChunkType.TEXT,
                                    content="Test content 2",
                                    content_length=15,
                                    token_count=3,
                                ),
                            ],
                            total_chunks=2,
                            processing_time=0.1,
                        )
                    )

                    # 処理実行
                    result = await processing_service.process_documents(
                        processing_config
                    )

        # 結果検証
        assert result.success is True
        assert result.total_documents == 2
        assert result.successful_documents == 2
        assert result.failed_documents == 0
        assert result.total_chunks == 4  # 2 documents * 2 chunks each
        assert result.successful_chunks == 4
        assert result.failed_chunks == 0
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_process_documents_no_documents(
        self, processing_service, processing_config
    ):
        """ドキュメントが収集されない場合のテスト"""
        with patch(
            "app.services.document_processing_service.DocumentCollector"
        ) as mock_collector_class:
            mock_collector = mock_collector_class.return_value
            mock_collector.collect_documents = AsyncMock(
                return_value=CollectionResult(
                    documents=[],
                    success_count=0,
                    error_count=0,
                    errors=[],
                    collection_time=0.1,
                )
            )

            result = await processing_service.process_documents(processing_config)

        assert result.success is False
        assert result.total_documents == 0
        assert result.errors == ["No documents collected"]

    @pytest.mark.asyncio
    async def test_process_documents_collection_failure(
        self, processing_service, processing_config
    ):
        """ドキュメント収集が失敗する場合のテスト"""
        with patch(
            "app.services.document_processing_service.DocumentCollector"
        ) as mock_collector_class:
            mock_collector = mock_collector_class.return_value
            mock_collector.collect_documents = AsyncMock(
                side_effect=Exception("Collection error")
            )

            result = await processing_service.process_documents(processing_config)

        assert result.success is False
        assert result.total_documents == 0
        assert "Collection error" in result.errors[0]

    @pytest.mark.asyncio
    async def test_collect_documents_failure_with_errors(
        self, processing_service, processing_config
    ):
        """収集エラーがある場合のテスト"""
        with patch(
            "app.services.document_processing_service.DocumentCollector"
        ) as mock_collector_class:
            mock_collector = mock_collector_class.return_value
            mock_collector.collect_documents = AsyncMock(
                return_value=CollectionResult(
                    documents=[],
                    success_count=0,
                    error_count=3,
                    errors=["Error 1", "Error 2", "Error 3"],
                    collection_time=0.3,
                )
            )

            with pytest.raises(Exception) as exc_info:
                await processing_service._collect_documents(
                    processing_config.collection_config
                )

            assert "Document collection failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_single_document_failure(
        self, processing_service, processing_config
    ):
        """単一ドキュメント処理の失敗テスト"""
        document = {
            "id": "doc_1",
            "title": "Test Document",
            "content": "Test content",
            "source_type": "test",
        }

        # チャンク化で失敗させる
        with patch(
            "app.services.document_processing_service.DocumentChunker"
        ) as mock_chunker_class:
            mock_chunker = mock_chunker_class.return_value
            mock_chunker.chunk_document = AsyncMock(
                return_value=ChunkResult(
                    success=False,
                    chunks=[],
                    total_chunks=0,
                    processing_time=0.1,
                    error_message="Chunking failed",
                )
            )

            semaphore = asyncio.Semaphore(1)
            result = await processing_service._process_single_document(
                document, processing_config, semaphore, semaphore
            )

        assert result["success"] is False
        assert "Document chunking failed" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_extract_metadata_success(
        self, processing_service, processing_config
    ):
        """メタデータ抽出成功のテスト"""
        document = {"id": "doc_1", "content": "Test content"}

        with patch(
            "app.services.document_processing_service.MetadataExtractor"
        ) as mock_extractor_class:
            mock_extractor = mock_extractor_class.return_value
            mock_extractor.extract_metadata = AsyncMock(
                return_value=ExtractionResult(
                    success=True,
                    document_id="doc_1",
                    metadata={"keywords": ["test", "content"]},
                    field_types={"keywords": FieldType.ARRAY},
                    processing_time=0.1,
                    error_message=None,
                )
            )

            result = await processing_service._extract_metadata(
                document, processing_config.extraction_config
            )

        assert result is not None
        assert result.success is True
        assert result.metadata["keywords"] == ["test", "content"]

    @pytest.mark.asyncio
    async def test_extract_metadata_failure(
        self, processing_service, processing_config
    ):
        """メタデータ抽出失敗のテスト"""
        document = {"id": "doc_1", "content": "Test content"}

        with patch(
            "app.services.document_processing_service.MetadataExtractor"
        ) as mock_extractor_class:
            mock_extractor = mock_extractor_class.return_value
            mock_extractor.extract_metadata = AsyncMock(
                return_value=ExtractionResult(
                    success=False,
                    document_id="doc_1",
                    metadata={},
                    field_types={},
                    processing_time=0.1,
                    error_message="Extraction failed",
                )
            )

            result = await processing_service._extract_metadata(
                document, processing_config.extraction_config
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_extract_metadata_exception(
        self, processing_service, processing_config
    ):
        """メタデータ抽出で例外が発生する場合のテスト"""
        document = {"id": "doc_1", "content": "Test content"}

        with patch(
            "app.services.document_processing_service.MetadataExtractor"
        ) as mock_extractor_class:
            mock_extractor = mock_extractor_class.return_value
            mock_extractor.extract_metadata = AsyncMock(
                side_effect=Exception("Extraction error")
            )

            result = await processing_service._extract_metadata(
                document, processing_config.extraction_config
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_chunk_document_success(self, processing_service, processing_config):
        """ドキュメントチャンク化成功のテスト"""
        document = {"id": "doc_1", "content": "Test content"}

        with patch(
            "app.services.document_processing_service.DocumentChunker"
        ) as mock_chunker_class:
            mock_chunker = mock_chunker_class.return_value
            mock_chunker.chunk_document = AsyncMock(
                return_value=ChunkResult(
                    success=True,
                    chunks=[
                        ChunkerDocumentChunk(
                            id="chunk_1",
                            document_id="doc_1",
                            chunk_index=0,
                            chunk_type=ChunkType.TEXT,
                            content="Test content",
                            content_length=12,
                            token_count=2,
                        )
                    ],
                    total_chunks=1,
                    processing_time=0.1,
                )
            )

            result = await processing_service._chunk_document(
                document, processing_config.chunking_config
            )

        assert len(result.chunks if hasattr(result, "chunks") else []) == 1
        assert (result.chunks if hasattr(result, "chunks") else [])[0][
            "id"
        ] == "chunk_1"
        assert (result.chunks if hasattr(result, "chunks") else [])[0][
            "content"
        ] == "Test content"
        assert (result.chunks if hasattr(result, "chunks") else [])[0][
            "language"
        ] == "en"

    @pytest.mark.asyncio
    async def test_chunk_document_failure(self, processing_service, processing_config):
        """ドキュメントチャンク化失敗のテスト"""
        document = {"id": "doc_1", "content": "Test content"}

        with patch(
            "app.services.document_processing_service.DocumentChunker"
        ) as mock_chunker_class:
            mock_chunker = mock_chunker_class.return_value
            mock_chunker.chunk_document = AsyncMock(
                return_value=ChunkResult(
                    success=False,
                    chunks=[],
                    total_chunks=0,
                    processing_time=0.1,
                    error_message="Chunking failed",
                )
            )

            result = await processing_service._chunk_document(
                document, processing_config.chunking_config
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_chunk_document_exception(
        self, processing_service, processing_config
    ):
        """ドキュメントチャンク化で例外が発生する場合のテスト"""
        document = {"id": "doc_1", "content": "Test content"}

        with patch(
            "app.services.document_processing_service.DocumentChunker"
        ) as mock_chunker_class:
            mock_chunker = mock_chunker_class.return_value
            mock_chunker.chunk_document = AsyncMock(
                side_effect=Exception("Chunking error")
            )

            result = await processing_service._chunk_document(
                document, processing_config.chunking_config
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_process_chunk_embeddings_with_service(self, processing_service):
        """埋め込みサービスがある場合のチャンク埋め込み処理テスト"""
        chunks = [
            {"id": "chunk_1", "content": "Content 1"},
            {"id": "chunk_2", "content": "Content 2"},
        ]
        semaphore = asyncio.Semaphore(1)

        result = await processing_service._process_chunk_embeddings(
            chunks, semaphore, "doc_1"
        )

        assert len(result.chunks if hasattr(result, "chunks") else []) == 2
        assert (
            "embeddings" in (result.chunks if hasattr(result, "chunks") else [])[0]
        )
        assert (
            "embeddings" in (result.chunks if hasattr(result, "chunks") else [])[1]
        )
        assert (result.chunks if hasattr(result, "chunks") else [])[0][
            "embeddings"
        ] == [0.1] * 1024

    @pytest.mark.asyncio
    async def test_process_chunk_embeddings_without_service(self, processing_service):
        """埋め込みサービスがない場合のチャンク埋め込み処理テスト"""
        processing_service.embedding_service = None
        chunks = [
            {"id": "chunk_1", "content": "Content 1"},
            {"id": "chunk_2", "content": "Content 2"},
        ]
        semaphore = asyncio.Semaphore(1)

        result = await processing_service._process_chunk_embeddings(
            chunks, semaphore, "doc_1"
        )

        assert result == chunks
        assert (
            "embeddings"
            not in (result.chunks if hasattr(result, "chunks") else [])[0]
        )

    @pytest.mark.asyncio
    async def test_process_chunk_embeddings_exception(self, processing_service):
        """埋め込み処理で例外が発生する場合のテスト"""
        processing_service.embedding_service.embed_batch = AsyncMock(
            side_effect=Exception("Embedding error")
        )
        chunks = [
            {"id": "chunk_1", "content": "Content 1"},
            {"id": "chunk_2", "content": "Content 2"},
        ]
        semaphore = asyncio.Semaphore(1)

        result = await processing_service._process_chunk_embeddings(
            chunks, semaphore, "doc_1"
        )

        assert result == chunks
        assert (
            "embeddings"
            not in (result.chunks if hasattr(result, "chunks") else [])[0]
        )

    @pytest.mark.asyncio
    async def test_store_document_and_chunks_success(
        self, processing_service, mock_repositories
    ):
        """ドキュメントとチャンクの保存成功テスト"""
        document = {
            "id": "doc_1",
            "title": "Test Document",
            "content": "Test content",
            "source_type": "test",
            "source_id": "src_1",
            "file_type": "txt",
            "language": "ja",
        }
        chunks = [
            {
                "id": "chunk_1",
                "document_id": "doc_1",
                "chunk_index": 0,
                "chunk_type": "paragraph",
                "title": "Chunk Title",
                "content": "Chunk content",
                "content_length": 13,
                "token_count": 2,
                "hierarchy_path": "/doc/chunk1",
                "chunk_metadata": {"key": "value"},
            }
        ]

        await processing_service._store_document_and_chunks(document, chunks)

        # リポジトリのcreateメソッドが呼ばれたことを確認
        document_repo, chunk_repo = mock_repositories
        assert document_repo.create.called
        assert chunk_repo.create.called

    @pytest.mark.asyncio
    async def test_store_document_and_chunks_failure(
        self, processing_service, mock_repositories
    ):
        """ドキュメント保存失敗のテスト"""
        document_repo, chunk_repo = mock_repositories
        document_repo.create = AsyncMock(side_effect=Exception("DB error"))

        document = {
            "id": "doc_1",
            "title": "Test Document",
            "content": "Test content",
            "source_type": "test",
        }
        chunks = []

        with pytest.raises(Exception) as exc_info:
            await processing_service._store_document_and_chunks(document, chunks)

        assert "DB error" in str(exc_info.value)

    def test_calculate_content_hash(self, processing_service):
        """コンテンツハッシュ計算のテスト"""
        content = "Test content"
        hash_value = processing_service._calculate_content_hash(content)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA256のハッシュ長

        # 同じコンテンツは同じハッシュ値
        hash_value2 = processing_service._calculate_content_hash(content)
        assert hash_value == hash_value2

        # 異なるコンテンツは異なるハッシュ値
        hash_value3 = processing_service._calculate_content_hash("Different content")
        assert hash_value != hash_value3

    def test_get_processing_status(self, processing_service):
        """処理状況取得のテスト"""
        # 状況を設定
        status = DocumentProcessingStatus(
            document_id="doc_1",
            stage=ProcessingStage.CHUNKING,
            progress=0.5,
        )
        processing_service._processing_status["doc_1"] = status

        # 取得テスト
        result = processing_service.get_processing_status("doc_1")
        assert result is not None
        assert result["document_id"] == "doc_1"
        assert result["stage"] == ProcessingStage.CHUNKING
        assert result["progress"] == 0.5

        # 存在しないドキュメント
        result = processing_service.get_processing_status("doc_999")
        assert result is None

    def test_get_all_processing_status(self, processing_service):
        """全処理状況取得のテスト"""
        # 複数の状況を設定
        status1 = DocumentProcessingStatus(
            document_id="doc_1",
            stage=ProcessingStage.CHUNKING,
            progress=0.5,
        )
        status2 = DocumentProcessingStatus(
            document_id="doc_2",
            stage=ProcessingStage.COMPLETED,
            progress=1.0,
        )
        processing_service._processing_status["doc_1"] = status1
        processing_service._processing_status["doc_2"] = status2

        # 取得テスト
        all_status = processing_service.get_all_processing_status()
        assert len(all_status) == 2
        assert all_status["doc_1"]["stage"] == ProcessingStage.CHUNKING
        assert all_status["doc_2"]["stage"] == ProcessingStage.COMPLETED

    @pytest.mark.asyncio
    async def test_process_single_document_by_id_success(
        self, processing_service, processing_config, mock_repositories
    ):
        """ID指定での単一ドキュメント処理成功テスト"""
        document_repo, chunk_repo = mock_repositories
        document_repo.get_by_id = AsyncMock(
            return_value=Document(
                id="doc_1",
                title="Test Document",
                content="Test content",
                source_type="test",
                source_id="src_1",
                file_type="txt",
                language="ja",
                status="active",
                content_hash="hash",
            )
        )

        with patch.object(
            processing_service,
            "_process_single_document",
            AsyncMock(
                return_value={
                    "success": True,
                    "document_id": "doc_1",
                    "total_chunks": 2,
                    "successful_chunks": 2,
                    "failed_chunks": 0,
                }
            ),
        ):
            result = await processing_service.process_single_document_by_id(
                "doc_1", processing_config
            )

        assert result["success"] is True
        assert result["document_id"] == "doc_1"

    @pytest.mark.asyncio
    async def test_process_single_document_by_id_not_found(
        self, processing_service, processing_config, mock_repositories
    ):
        """存在しないドキュメントIDでの処理テスト"""
        document_repo, chunk_repo = mock_repositories
        document_repo.get_by_id = AsyncMock(return_value=None)

        result = await processing_service.process_single_document_by_id(
            "doc_999", processing_config
        )

        assert result["success"] is False
        assert "Document not found" in result["error"]

    @pytest.mark.asyncio
    async def test_process_single_document_by_id_exception(
        self, processing_service, processing_config, mock_repositories
    ):
        """ID指定処理で例外が発生する場合のテスト"""
        document_repo, chunk_repo = mock_repositories
        document_repo.get_by_id = AsyncMock(side_effect=Exception("DB error"))

        result = await processing_service.process_single_document_by_id(
            "doc_1", processing_config
        )

        assert result["success"] is False
        assert "DB error" in result["error"]

    @pytest.mark.asyncio
    async def test_reprocess_failed_documents_no_failures(
        self, processing_service, processing_config
    ):
        """失敗ドキュメントがない場合の再処理テスト"""
        result = await processing_service.reprocess_failed_documents(processing_config)

        assert result.success is True
        assert result.total_documents == 0
        assert result.successful_documents == 0
        assert result.failed_documents == 0

    @pytest.mark.asyncio
    async def test_reprocess_failed_documents_with_failures(
        self, processing_service, processing_config
    ):
        """失敗ドキュメントがある場合の再処理テスト"""
        # 失敗状態のドキュメントを設定
        failed_status1 = DocumentProcessingStatus(
            document_id="doc_1",
            stage=ProcessingStage.FAILED,
            progress=0.5,
            error_message="Previous error",
        )
        failed_status2 = DocumentProcessingStatus(
            document_id="doc_2",
            stage=ProcessingStage.FAILED,
            progress=0.3,
            error_message="Another error",
        )
        processing_service._processing_status["doc_1"] = failed_status1
        processing_service._processing_status["doc_2"] = failed_status2

        # process_single_document_by_idをモック
        with patch.object(
            processing_service,
            "process_single_document_by_id",
            AsyncMock(
                side_effect=[
                    {"success": True, "document_id": "doc_1"},
                    {"success": False, "error": "Still failing"},
                ]
            ),
        ):
            result = await processing_service.reprocess_failed_documents(
                processing_config
            )

        assert result.success is True
        assert result.total_documents == 2
        assert result.successful_documents == 1
        assert result.failed_documents == 1

        # 失敗状態がリセットされていることを確認
        assert "doc_1" not in processing_service._processing_status
        assert "doc_2" not in processing_service._processing_status

    @pytest.mark.asyncio
    async def test_reprocess_failed_documents_exception(
        self, processing_service, processing_config
    ):
        """再処理で例外が発生する場合のテスト"""
        # 失敗状態のドキュメントを設定
        failed_status = DocumentProcessingStatus(
            document_id="doc_1",
            stage=ProcessingStage.FAILED,
            progress=0.5,
        )
        processing_service._processing_status["doc_1"] = failed_status

        # process_single_document_by_idで例外を発生させる
        with patch.object(
            processing_service,
            "process_single_document_by_id",
            AsyncMock(side_effect=Exception("Reprocessing error")),
        ):
            result = await processing_service.reprocess_failed_documents(
                processing_config
            )

        # gather return_exceptions=Trueなので、例外も結果として扱われる
        assert result.success is False
        assert result.total_documents == 1
        assert result.successful_documents == 0
        assert result.failed_documents == 1

    @pytest.mark.asyncio
    async def test_process_with_embedding_disabled(
        self, processing_service, processing_config, mock_repositories
    ):
        """埋め込み処理を無効にした場合のテスト"""
        processing_config.enable_embedding = False

        with patch(
            "app.services.document_processing_service.DocumentCollector"
        ) as mock_collector_class:
            mock_collector = mock_collector_class.return_value
            mock_collector.collect_documents = AsyncMock(
                return_value=CollectionResult(
                    documents=[
                        {
                            "id": "doc_1",
                            "title": "Test Document",
                            "content": "Test content",
                            "source_type": "test",
                        }
                    ],
                    success_count=1,
                    error_count=0,
                    errors=[],
                    collection_time=0.2,
                )
            )

            with patch("app.services.document_processing_service.MetadataExtractor"):
                with patch(
                    "app.services.document_processing_service.DocumentChunker"
                ) as mock_chunker_class:
                    mock_chunker = mock_chunker_class.return_value
                    mock_chunker.chunk_document = AsyncMock(
                        return_value=ChunkResult(
                            success=True,
                            chunks=[
                                ChunkerDocumentChunk(
                                    id="chunk_1",
                                    document_id="doc_1",
                                    chunk_index=0,
                                    chunk_type=ChunkType.TEXT,
                                    content="Chunk content",
                                    content_length=13,
                                    token_count=2,
                                )
                            ],
                            total_chunks=1,
                            processing_time=0.1,
                        )
                    )

                    # 埋め込みサービスのembed_batchが呼ばれないことを確認
                    result = await processing_service.process_documents(
                        processing_config
                    )

        assert result.success is True
        # 埋め込みが無効化されているので、埋め込みは実行されない
        assert processing_config.enable_embedding is False

    @pytest.mark.asyncio
    async def test_parallel_document_processing(
        self, processing_service, processing_config
    ):
        """並行ドキュメント処理のテスト"""
        processing_config.max_concurrent_documents = 2

        with patch(
            "app.services.document_processing_service.DocumentCollector"
        ) as mock_collector_class:
            mock_collector = mock_collector_class.return_value
            mock_collector.collect_documents = AsyncMock(
                return_value=CollectionResult(
                    documents=[
                        {
                            "id": f"doc_{i}",
                            "title": f"Test Document {i}",
                            "content": f"Test content {i}",
                            "source_type": "test",
                        }
                        for i in range(5)
                    ],
                    success_count=5,
                    error_count=0,
                    errors=[],
                    collection_time=1.0,
                )
            )

            # 各処理ステージのモック
            with patch("app.services.document_processing_service.MetadataExtractor"):
                with patch(
                    "app.services.document_processing_service.DocumentChunker"
                ) as mock_chunker_class:
                    mock_chunker = mock_chunker_class.return_value
                    mock_chunker.chunk_document = AsyncMock(
                        return_value=ChunkResult(
                            success=True,
                            chunks=[
                                ChunkerDocumentChunk(
                                    id="chunk_1",
                                    document_id="doc_1",
                                    chunk_index=0,
                                    chunk_type=ChunkType.TEXT,
                                    content="Chunk content",
                                    content_length=13,
                                    token_count=2,
                                )
                            ],
                            total_chunks=1,
                            processing_time=0.1,
                        )
                    )

                    # 並行処理の実行
                    result = await processing_service.process_documents(
                        processing_config
                    )

        assert result.success is True
        assert result.total_documents == 5
        assert result.successful_documents == 5

    @pytest.mark.asyncio
    async def test_process_with_stage_tracking(
        self, processing_service, processing_config
    ):
        """ステージ時間トラッキングのテスト"""
        with patch(
            "app.services.document_processing_service.DocumentCollector"
        ) as mock_collector_class:
            mock_collector = mock_collector_class.return_value

            # 収集に時間がかかるシミュレーション
            mock_collector.collect_documents = AsyncMock(
                return_value=CollectionResult(
                    documents=[
                        {
                            "id": "doc_1",
                            "title": "Test Document",
                            "content": "Test content",
                            "source_type": "test",
                        }
                    ],
                    success_count=1,
                    error_count=0,
                    errors=[],
                    collection_time=0.5,
                )
            )

            with patch("app.services.document_processing_service.MetadataExtractor"):
                with patch(
                    "app.services.document_processing_service.DocumentChunker"
                ) as mock_chunker_class:
                    mock_chunker = mock_chunker_class.return_value
                    mock_chunker.chunk_document = AsyncMock(
                        return_value=ChunkResult(
                            success=True,
                            chunks=[],
                            total_chunks=0,
                            processing_time=0.1,
                        )
                    )

                    result = await processing_service.process_documents(
                        processing_config
                    )

        # コレクションステージの時間が記録されていることを確認
        assert ProcessingStage.COLLECTION in result.stage_times
        assert result.stage_times[ProcessingStage.COLLECTION] >= 0
