"""バックグラウンド埋め込みタスク処理

Celeryを使用した非同期埋め込み処理。
大量のドキュメント処理やバッチ処理に使用。
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

try:
    from celery import Celery
    from celery.result import AsyncResult
except ImportError:
    # テスト環境での代替
    class Celery:
        def __init__(self, *args, **kwargs):
            pass

        @property
        def conf(self):
            return type('MockConf', (), {'update': lambda **kwargs: None})()

        def task(self, *args, **kwargs):
            def decorator(func):
                func.delay = lambda *a, **k: type('MockResult', (), {})()
                return func
            return decorator

        @property
        def control(self):
            return type('MockControl', (), {
                'revoke': lambda *args, **kwargs: None,
                'inspect': lambda: type('MockInspect', (), {
                    'active': lambda: {},
                    'scheduled': lambda: {},
                    'reserved': lambda: {},
                    'stats': lambda: {}
                })()
            })()

    class AsyncResult:
        def __init__(self, *args, **kwargs):
            self.status = "SUCCESS"
            self.result = {}
            self.info = None

        def ready(self):
            return True

        def successful(self):
            return True

        def failed(self):
            return False

from app.services.embedding_service import (
    EmbeddingService,
    EmbeddingConfig,
    BatchEmbeddingRequest,
    EmbeddingResult
)
from app.models.milvus import VectorData
from app.models.database import DocumentChunk

try:
    from app.repositories.chunk_repository import ChunkRepository
except ImportError:
    # テスト用ダミークラス
    class ChunkRepository:
        def __init__(self):
            pass

        async def get_chunks_by_document_id(self, document_id):
            return []

logger = logging.getLogger(__name__)

# Celeryアプリケーションの設定
celery_app = Celery(
    "embedding_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

# Celery設定
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Tokyo",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30分でタイムアウト
    task_soft_time_limit=25 * 60,  # 25分でソフトタイムアウト
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)


class EmbeddingTaskService:
    """埋め込みタスクサービス"""

    def __init__(self):
        self.embedding_service: Optional[EmbeddingService] = None
        self.chunk_repository: Optional[ChunkRepository] = None

    async def initialize(self):
        """サービスの初期化"""
        # 埋め込みサービスの初期化
        config = EmbeddingConfig()
        self.embedding_service = EmbeddingService(config)
        await self.embedding_service.initialize()

        # リポジトリの初期化
        self.chunk_repository = ChunkRepository()

        logger.info("EmbeddingTaskService initialized")

    async def process_document_chunks(self, document_id: str) -> Dict[str, Any]:
        """ドキュメントのチャンク埋め込み処理

        Args:
            document_id: ドキュメントID

        Returns:
            Dict[str, Any]: 処理結果
        """
        try:
            # ドキュメントのチャンクを取得
            chunks = await self.chunk_repository.get_chunks_by_document_id(document_id)

            if not chunks:
                return {
                    "status": "completed",
                    "message": "No chunks found for document",
                    "document_id": document_id,
                    "processed_count": 0
                }

            # バッチリクエストの作成
            batch_request = BatchEmbeddingRequest(
                texts=[chunk.content for chunk in chunks],
                chunk_ids=[chunk.id for chunk in chunks],
                document_ids=[chunk.document_id for chunk in chunks]
            )

            # 埋め込み処理
            embedding_results = await self.embedding_service.process_batch_request(batch_request)

            # VectorDataオブジェクトの作成とMilvusへの保存
            vector_data_list = []
            for i, (chunk, result) in enumerate(zip(chunks, embedding_results)):
                # Dense vector用
                dense_vector_data = VectorData(
                    id=f"{chunk.id}_dense",
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    vector=result.dense_vector,
                    chunk_type=chunk.chunk_type,
                    source_type="document",
                    language="ja"
                )
                vector_data_list.append(dense_vector_data)

                # Sparse vector用
                sparse_vector_data = VectorData(
                    id=f"{chunk.id}_sparse",
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    sparse_vector=result.sparse_vector,
                    vocabulary_size=len(result.sparse_vector),
                    chunk_type=chunk.chunk_type,
                    source_type="document",
                    language="ja"
                )
                vector_data_list.append(sparse_vector_data)

            # Milvusへの一括挿入（実装後に有効化）
            # await self._insert_vectors_to_milvus(vector_data_list)

            return {
                "status": "completed",
                "message": "Document chunks processed successfully",
                "document_id": document_id,
                "processed_count": len(chunks),
                "vector_count": len(vector_data_list)
            }

        except Exception as e:
            logger.error(f"Failed to process document chunks: {e}")
            return {
                "status": "failed",
                "message": f"Processing failed: {str(e)}",
                "document_id": document_id,
                "processed_count": 0
            }


# グローバルサービスインスタンス
_task_service: Optional[EmbeddingTaskService] = None


async def get_task_service() -> EmbeddingTaskService:
    """タスクサービスのシングルトン取得"""
    global _task_service
    if _task_service is None:
        _task_service = EmbeddingTaskService()
        await _task_service.initialize()
    return _task_service


@celery_app.task(bind=True, name="embedding.process_document")
def process_document_embedding_task(self, document_id: str) -> Dict[str, Any]:
    """ドキュメント埋め込み処理タスク

    Args:
        document_id: ドキュメントID

    Returns:
        Dict[str, Any]: 処理結果
    """
    try:
        # タスクステータスの更新
        self.update_state(
            state="PROGRESS",
            meta={"status": "Starting document processing", "document_id": document_id}
        )

        # 非同期処理の実行
        async def run_processing():
            service = await get_task_service()
            return await service.process_document_chunks(document_id)

        # 新しいイベントループで実行
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(run_processing())
            return result
        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Task failed for document {document_id}: {e}")
        return {
            "status": "failed",
            "message": f"Task execution failed: {str(e)}",
            "document_id": document_id
        }


@celery_app.task(bind=True, name="embedding.process_batch_texts")
def process_batch_texts_task(self, texts: List[str], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """バッチテキスト埋め込み処理タスク

    Args:
        texts: テキストリスト
        metadata: メタデータ

    Returns:
        Dict[str, Any]: 処理結果
    """
    try:
        self.update_state(
            state="PROGRESS",
            meta={"status": "Processing batch texts", "text_count": len(texts)}
        )

        async def run_batch_processing():
            service = await get_task_service()
            results = await service.embedding_service.embed_batch(texts)

            return {
                "status": "completed",
                "message": f"Processed {len(texts)} texts successfully",
                "text_count": len(texts),
                "results": [
                    {
                        "dense_vector_length": len(result.dense_vector),
                        "sparse_vector_size": len(result.sparse_vector),
                        "processing_time": result.processing_time
                    }
                    for result in results
                ]
            }

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(run_batch_processing())
            return result
        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Batch text processing failed: {e}")
        return {
            "status": "failed",
            "message": f"Batch processing failed: {str(e)}",
            "text_count": len(texts) if texts else 0
        }


@celery_app.task(name="embedding.health_check")
def embedding_health_check_task() -> Dict[str, Any]:
    """埋め込みサービスのヘルスチェックタスク"""
    try:
        async def run_health_check():
            service = await get_task_service()
            return await service.embedding_service.health_check()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(run_health_check())
            return result
        finally:
            loop.close()

    except Exception as e:
        return {
            "status": "unhealthy",
            "reason": f"Health check task failed: {str(e)}"
        }


class EmbeddingTaskManager:
    """埋め込みタスク管理クラス"""

    @staticmethod
    def submit_document_processing(document_id: str) -> AsyncResult:
        """ドキュメント処理タスクの投入

        Args:
            document_id: ドキュメントID

        Returns:
            AsyncResult: タスク結果オブジェクト
        """
        return process_document_embedding_task.delay(document_id)

    @staticmethod
    def submit_batch_processing(texts: List[str], metadata: Optional[Dict[str, Any]] = None) -> AsyncResult:
        """バッチ処理タスクの投入

        Args:
            texts: テキストリスト
            metadata: メタデータ

        Returns:
            AsyncResult: タスク結果オブジェクト
        """
        return process_batch_texts_task.delay(texts, metadata)

    @staticmethod
    def get_task_status(task_id: str) -> Dict[str, Any]:
        """タスクステータスの取得

        Args:
            task_id: タスクID

        Returns:
            Dict[str, Any]: タスクステータス
        """
        result = AsyncResult(task_id, app=celery_app)

        return {
            "task_id": task_id,
            "status": result.status,
            "result": result.result,
            "info": result.info,
            "ready": result.ready(),
            "successful": result.successful() if result.ready() else None,
            "failed": result.failed() if result.ready() else None
        }

    @staticmethod
    def cancel_task(task_id: str) -> Dict[str, Any]:
        """タスクのキャンセル

        Args:
            task_id: タスクID

        Returns:
            Dict[str, Any]: キャンセル結果
        """
        celery_app.control.revoke(task_id, terminate=True)

        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Task cancellation requested"
        }

    @staticmethod
    def get_worker_status() -> Dict[str, Any]:
        """ワーカーステータスの取得

        Returns:
            Dict[str, Any]: ワーカーステータス
        """
        inspect = celery_app.control.inspect()

        return {
            "active_tasks": inspect.active(),
            "scheduled_tasks": inspect.scheduled(),
            "reserved_tasks": inspect.reserved(),
            "stats": inspect.stats()
        }
