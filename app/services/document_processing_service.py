"""ドキュメント処理サービス

全体的なドキュメント処理のオーケストレーション
- ドキュメント収集
- メタデータ抽出
- チャンク化
- 埋め込み処理
- データベース保存
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from app.models.database import Document, DocumentChunk
from app.repositories.chunk_repository import DocumentChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.services.document_chunker import (
    ChunkingConfig,
    DocumentChunker,
)
from app.services.document_collector import (
    CollectionConfig,
    DocumentCollector,
)
from app.services.embedding_service import EmbeddingService
from app.services.metadata_extractor import ExtractionConfig, MetadataExtractor

logger = logging.getLogger(__name__)


class ProcessingStage(str, Enum):
    """処理ステージ"""

    COLLECTION = "collection"
    METADATA_EXTRACTION = "metadata_extraction"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORAGE = "storage"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingConfig:
    """ドキュメント処理設定"""

    # 収集設定
    collection_config: CollectionConfig

    # メタデータ抽出設定
    extraction_config: ExtractionConfig

    # チャンク化設定
    chunking_config: ChunkingConfig

    # 埋め込み設定
    enable_embedding: bool = True

    # 並行処理設定
    max_concurrent_documents: int = 5
    max_concurrent_chunks: int = 10

    # エラー処理設定
    continue_on_error: bool = True
    max_retries: int = 3


@dataclass
class ProcessingResult:
    """処理結果"""

    success: bool
    total_documents: int
    successful_documents: int
    failed_documents: int
    total_chunks: int
    successful_chunks: int
    failed_chunks: int
    processing_time: float
    stage_times: dict[str, float]
    errors: list[str]

    def get_summary(self) -> dict[str, Any]:
        """処理結果のサマリーを取得"""
        return {
            "success": self.success,
            "documents": {
                "total": self.total_documents,
                "successful": self.successful_documents,
                "failed": self.failed_documents,
                "success_rate": self.successful_documents
                / max(self.total_documents, 1),
            },
            "chunks": {
                "total": self.total_chunks,
                "successful": self.successful_chunks,
                "failed": self.failed_chunks,
                "success_rate": self.successful_chunks / max(self.total_chunks, 1),
            },
            "timing": {
                "total_time": self.processing_time,
                "stage_times": self.stage_times,
            },
            "error_count": len(self.errors),
        }


@dataclass
class DocumentProcessingStatus:
    """ドキュメント処理状況"""

    document_id: str
    stage: ProcessingStage
    progress: float  # 0.0 - 1.0
    error_message: str | None = None
    chunks_processed: int = 0
    chunks_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "document_id": self.document_id,
            "stage": self.stage,
            "progress": self.progress,
            "error_message": self.error_message,
            "chunks_processed": self.chunks_processed,
            "chunks_total": self.chunks_total,
        }


class DocumentProcessingService:
    """ドキュメント処理サービス"""

    def __init__(
        self,
        document_repository: DocumentRepository,
        chunk_repository: DocumentChunkRepository,
        embedding_service: EmbeddingService | None = None,
    ):
        self.document_repository = document_repository
        self.chunk_repository = chunk_repository
        self.embedding_service = embedding_service

        # 処理状況トラッキング
        self._processing_status: dict[str, DocumentProcessingStatus] = {}
        self._stage_times: dict[str, float] = {}

    async def process_documents(self, config: ProcessingConfig) -> ProcessingResult:
        """ドキュメントを一括処理"""
        start_time = datetime.now()
        errors = []

        try:
            logger.info("Starting document processing pipeline")

            # Stage 1: ドキュメント収集
            stage_start = datetime.now()
            documents = await self._collect_documents(config.collection_config)
            self._stage_times[ProcessingStage.COLLECTION] = (
                datetime.now() - stage_start
            ).total_seconds()

            if not documents:
                return ProcessingResult(
                    success=False,
                    total_documents=0,
                    successful_documents=0,
                    failed_documents=0,
                    total_chunks=0,
                    successful_chunks=0,
                    failed_chunks=0,
                    processing_time=0.0,
                    stage_times=self._stage_times,
                    errors=["No documents collected"],
                )

            logger.info(f"Collected {len(documents)} documents")

            # 並行処理のためのセマフォ
            doc_semaphore = asyncio.Semaphore(config.max_concurrent_documents)
            chunk_semaphore = asyncio.Semaphore(config.max_concurrent_chunks)

            # ドキュメント処理タスクを作成
            processing_tasks = []
            for doc in documents:
                task = self._process_single_document(
                    doc, config, doc_semaphore, chunk_semaphore
                )
                processing_tasks.append(task)

            # 並行処理実行
            results = await asyncio.gather(*processing_tasks, return_exceptions=True)

            # 結果の集計
            successful_docs = 0
            failed_docs = 0
            total_chunks = 0
            successful_chunks = 0
            failed_chunks = 0

            for result in results:
                if isinstance(result, Exception):
                    errors.append(str(result))
                    failed_docs += 1
                elif isinstance(result, dict):
                    if result.get("success", False):
                        successful_docs += 1
                        successful_chunks += result.get("successful_chunks", 0)
                        failed_chunks += result.get("failed_chunks", 0)
                        total_chunks += result.get("total_chunks", 0)
                    else:
                        failed_docs += 1
                        errors.extend(result.get("errors", []))

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            return ProcessingResult(
                success=successful_docs > 0,
                total_documents=len(documents),
                successful_documents=successful_docs,
                failed_documents=failed_docs,
                total_chunks=total_chunks,
                successful_chunks=successful_chunks,
                failed_chunks=failed_chunks,
                processing_time=processing_time,
                stage_times=self._stage_times,
                errors=errors,
            )

        except Exception as e:
            logger.error(f"Document processing pipeline failed: {e}")
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            return ProcessingResult(
                success=False,
                total_documents=0,
                successful_documents=0,
                failed_documents=0,
                total_chunks=0,
                successful_chunks=0,
                failed_chunks=0,
                processing_time=processing_time,
                stage_times=self._stage_times,
                errors=[str(e)],
            )

    async def _collect_documents(
        self, config: CollectionConfig
    ) -> list[dict[str, Any]]:
        """ドキュメントを収集"""
        collector = DocumentCollector(config=config)
        result = await collector.collect_documents()

        if not result.success:
            raise Exception(f"Document collection failed: {result.errors}")

        return result.documents

    async def _process_single_document(
        self,
        document: dict[str, Any],
        config: ProcessingConfig,
        doc_semaphore: asyncio.Semaphore,
        chunk_semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        """単一ドキュメントを処理"""
        async with doc_semaphore:
            document_id = document.get("id", "unknown")

            try:
                # 処理状況を初期化
                self._processing_status[document_id] = DocumentProcessingStatus(
                    document_id=document_id,
                    stage=ProcessingStage.METADATA_EXTRACTION,
                    progress=0.0,
                )

                # Stage 2: メタデータ抽出
                metadata_result = await self._extract_metadata(
                    document, config.extraction_config
                )
                if metadata_result:
                    document.update({"extracted_metadata": metadata_result.metadata})

                self._processing_status[document_id].progress = 0.3
                self._processing_status[document_id].stage = ProcessingStage.CHUNKING

                # Stage 3: チャンク化
                chunks = await self._chunk_document(document, config.chunking_config)
                if not chunks:
                    raise Exception("Document chunking failed")

                self._processing_status[document_id].progress = 0.6
                self._processing_status[document_id].chunks_total = len(chunks)
                self._processing_status[document_id].stage = ProcessingStage.EMBEDDING

                # Stage 4: 埋め込み処理（並行実行）
                if config.enable_embedding and self.embedding_service:
                    chunks = await self._process_chunk_embeddings(
                        chunks, chunk_semaphore, document_id
                    )

                self._processing_status[document_id].progress = 0.8
                self._processing_status[document_id].stage = ProcessingStage.STORAGE

                # Stage 5: データベース保存
                await self._store_document_and_chunks(document, chunks)

                # 完了
                self._processing_status[document_id].progress = 1.0
                self._processing_status[document_id].stage = ProcessingStage.COMPLETED

                return {
                    "success": True,
                    "document_id": document_id,
                    "total_chunks": len(chunks),
                    "successful_chunks": len(chunks),
                    "failed_chunks": 0,
                }

            except Exception as e:
                logger.error(f"Failed to process document {document_id}: {e}")
                self._processing_status[document_id].stage = ProcessingStage.FAILED
                self._processing_status[document_id].error_message = str(e)

                return {
                    "success": False,
                    "document_id": document_id,
                    "errors": [str(e)],
                }

    async def _extract_metadata(
        self, document: dict[str, Any], config: ExtractionConfig
    ) -> Any | None:
        """メタデータを抽出"""
        try:
            extractor = MetadataExtractor(config=config)
            result = await extractor.extract_metadata(document)

            if result.success:
                return result
            else:
                logger.warning(
                    f"Metadata extraction failed for {document.get('id')}: {result.error_message}"
                )
                return None

        except Exception as e:
            logger.error(f"Metadata extraction error: {e}")
            return None

    async def _chunk_document(
        self, document: dict[str, Any], config: ChunkingConfig
    ) -> list[dict[str, Any]]:
        """ドキュメントをチャンク化"""
        try:
            chunker = DocumentChunker(config=config)
            result = await chunker.chunk_document(document)

            if result.success:
                # DocumentChunkオブジェクトを辞書に変換
                chunks_dict = []
                for chunk in result.chunks:
                    chunk_dict = {
                        "id": chunk.id,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "chunk_type": chunk.chunk_type,
                        "title": chunk.title,
                        "content": chunk.content,
                        "content_length": chunk.content_length,
                        "token_count": chunk.token_count,
                        "hierarchy_path": chunk.hierarchy_path,
                        "chunk_metadata": chunk.chunk_metadata,
                        "language": chunk.language,
                    }
                    chunks_dict.append(chunk_dict)

                return chunks_dict
            else:
                logger.error(f"Document chunking failed: {result.error_message}")
                return []

        except Exception as e:
            logger.error(f"Document chunking error: {e}")
            return []

    async def _process_chunk_embeddings(
        self,
        chunks: list[dict[str, Any]],
        semaphore: asyncio.Semaphore,
        document_id: str,
    ) -> list[dict[str, Any]]:
        """チャンクの埋め込み処理"""
        if not self.embedding_service:
            return chunks

        try:
            # チャンクのテキストを抽出
            texts = [chunk["content"] for chunk in chunks]

            # バッチで埋め込み生成
            embeddings = await self.embedding_service.generate_embeddings(texts)

            # 埋め込みをチャンクに追加
            for i, chunk in enumerate(chunks):
                if i < len(embeddings):
                    chunk["embeddings"] = embeddings[i]

                # 処理状況を更新
                if document_id in self._processing_status:
                    self._processing_status[document_id].chunks_processed = i + 1

            return chunks

        except Exception as e:
            logger.error(f"Embedding processing failed: {e}")
            return chunks

    async def _store_document_and_chunks(
        self, document: dict[str, Any], chunks: list[dict[str, Any]]
    ) -> None:
        """ドキュメントとチャンクをデータベースに保存"""
        try:
            # ドキュメントの保存
            doc_model = Document(
                id=document["id"],
                source_type=document["source_type"],
                source_id=document.get("source_id", document["id"]),
                title=document["title"],
                content=document["content"],
                content_hash=self._calculate_content_hash(document["content"]),
                file_type=document.get("file_type"),
                language=document.get("language", "en"),
                status="active",
                processed_at=datetime.now(),
            )

            await self.document_repository.create(doc_model)

            # チャンクの保存
            for chunk_data in chunks:
                chunk_model = DocumentChunk(
                    id=chunk_data["id"],
                    document_id=chunk_data["document_id"],
                    chunk_index=chunk_data["chunk_index"],
                    chunk_type=chunk_data["chunk_type"],
                    title=chunk_data.get("title"),
                    content=chunk_data["content"],
                    content_length=chunk_data["content_length"],
                    token_count=chunk_data.get("token_count"),
                    hierarchy_path=chunk_data.get("hierarchy_path"),
                    chunk_metadata=chunk_data.get("chunk_metadata"),
                )

                await self.chunk_repository.create(chunk_model)

        except Exception as e:
            logger.error(f"Database storage failed: {e}")
            raise

    def _calculate_content_hash(self, content: str) -> str:
        """コンテンツのハッシュ値を計算"""
        import hashlib

        return hashlib.sha256(content.encode()).hexdigest()

    def get_processing_status(self, document_id: str) -> dict[str, Any] | None:
        """処理状況を取得"""
        status = self._processing_status.get(document_id)
        return status.to_dict() if status else None

    def get_all_processing_status(self) -> dict[str, dict[str, Any]]:
        """全ての処理状況を取得"""
        return {
            doc_id: status.to_dict()
            for doc_id, status in self._processing_status.items()
        }

    async def process_single_document_by_id(
        self, document_id: str, config: ProcessingConfig
    ) -> dict[str, Any]:
        """IDで指定された単一ドキュメントを処理"""
        try:
            # ドキュメントを取得
            document = await self.document_repository.get_by_id(document_id)
            if not document:
                raise Exception(f"Document not found: {document_id}")

            # ドキュメントを辞書形式に変換
            doc_dict = {
                "id": document.id,
                "title": document.title,
                "content": document.content,
                "source_type": document.source_type,
                "source_id": document.source_id,
                "file_type": document.file_type,
                "language": document.language,
            }

            # 単一ドキュメント処理
            semaphore = asyncio.Semaphore(1)
            result = await self._process_single_document(
                doc_dict, config, semaphore, semaphore
            )

            return result

        except Exception as e:
            logger.error(f"Single document processing failed: {e}")
            return {"success": False, "error": str(e)}

    async def reprocess_failed_documents(
        self, config: ProcessingConfig
    ) -> ProcessingResult:
        """失敗したドキュメントを再処理"""
        try:
            # 失敗状態のドキュメントIDを取得
            failed_ids = [
                doc_id
                for doc_id, status in self._processing_status.items()
                if status.stage == ProcessingStage.FAILED
            ]

            if not failed_ids:
                return ProcessingResult(
                    success=True,
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

            # 失敗したドキュメントの処理状況をリセット
            for doc_id in failed_ids:
                if doc_id in self._processing_status:
                    del self._processing_status[doc_id]

            # 再処理実行
            start_time = datetime.now()

            processing_tasks = []
            for doc_id in failed_ids:
                task = self.process_single_document_by_id(doc_id, config)
                processing_tasks.append(task)

            results = await asyncio.gather(*processing_tasks, return_exceptions=True)

            # 結果の集計
            successful_docs = sum(
                1 for r in results if isinstance(r, dict) and r.get("success", False)
            )
            failed_docs = len(failed_ids) - successful_docs

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            return ProcessingResult(
                success=successful_docs > 0,
                total_documents=len(failed_ids),
                successful_documents=successful_docs,
                failed_documents=failed_docs,
                total_chunks=0,  # 再処理では詳細なチャンク統計は計算しない
                successful_chunks=0,
                failed_chunks=0,
                processing_time=processing_time,
                stage_times={},
                errors=[],
            )

        except Exception as e:
            logger.error(f"Failed document reprocessing failed: {e}")
            return ProcessingResult(
                success=False,
                total_documents=0,
                successful_documents=0,
                failed_documents=0,
                total_chunks=0,
                successful_chunks=0,
                failed_chunks=0,
                processing_time=0.0,
                stage_times={},
                errors=[str(e)],
            )
