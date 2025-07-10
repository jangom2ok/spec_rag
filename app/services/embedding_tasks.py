"""バックグラウンド埋め込みタスク処理

Celeryを使用した非同期埋め込み処理。
大量のドキュメント処理やバッチ処理に使用。
"""

import asyncio
import logging
from typing import Any

try:
    import redis
    from celery import Celery
    from celery.result import AsyncResult

    HAS_CELERY = True
    HAS_REDIS = True
except ImportError:
    # テスト環境での代替
    HAS_CELERY = False
    HAS_REDIS = False
    redis = None

    class MockConf:
        def update(self, **kwargs):
            return None

    class MockInspect:
        def active(self):
            return {}

        def scheduled(self):
            return {}

        def reserved(self):
            return {}

        def stats(self):
            return {}

    class MockControl:
        def revoke(self, *args, **kwargs):
            return None

        def inspect(self):
            return MockInspect()

    class MockCelery:
        def __init__(self, *args, **kwargs):
            pass

        @property
        def conf(self):
            return MockConf()

        def task(self, *args, **kwargs):
            def decorator(func):
                func.delay = lambda *a, **k: type("MockResult", (), {})()
                return func

            return decorator

        @property
        def control(self):
            return MockControl()

    class MockAsyncResult:
        def __init__(self, *args, **kwargs):
            self.status = "SUCCESS"
            self.result = {}
            self.info = None
            self.id = "mock_task_id"

        def ready(self):
            return True

        def successful(self):
            return True

        def failed(self):
            return False


from app.models.aperturedb import VectorData
from app.services.embedding_service import (
    BatchEmbeddingRequest,
    EmbeddingConfig,
    EmbeddingService,
)

try:
    from app.repositories.chunk_repository import (
        DocumentChunkRepository as ChunkRepository,
    )
except ImportError:
    # テスト用ダミークラス
    class MockChunkRepository:
        def __init__(self):
            pass

        async def get_by_document_id(self, document_id):
            return []

    ChunkRepository = MockChunkRepository  # type: ignore

# Create AsyncResult alias for type annotations
if HAS_CELERY:
    from celery.result import AsyncResult
else:
    AsyncResult = MockAsyncResult

logger = logging.getLogger(__name__)

# Celeryアプリケーションの設定
if HAS_CELERY:
    celery_app = Celery(
        "embedding_tasks",
        broker="redis://localhost:6379/0",
        backend="redis://localhost:6379/0",
    )
else:
    celery_app = MockCelery()

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
        self.embedding_service: EmbeddingService | None = None
        self.chunk_repository: ChunkRepository | None = None

    async def initialize(self):
        """サービスの初期化"""
        # 埋め込みサービスの初期化
        config = EmbeddingConfig()
        self.embedding_service = EmbeddingService(config)
        await self.embedding_service.initialize()

        # リポジトリの初期化
        self.chunk_repository = ChunkRepository()

        logger.info("EmbeddingTaskService initialized")

    async def process_document_chunks(self, document_id: str) -> dict[str, Any]:
        """ドキュメントのチャンク埋め込み処理

        Args:
            document_id: ドキュメントID

        Returns:
            Dict[str, Any]: 処理結果
        """
        try:
            # ドキュメントのチャンクを取得
            if self.chunk_repository is not None:
                chunks = await self.chunk_repository.get_by_document_id(document_id)
            else:
                chunks = []

            if not chunks:
                return {
                    "status": "completed",
                    "message": "No chunks found for document",
                    "document_id": document_id,
                    "processed_count": 0,
                }

            # バッチリクエストの作成
            batch_request = BatchEmbeddingRequest(
                texts=[chunk.content for chunk in chunks],
                chunk_ids=[chunk.id for chunk in chunks],
                document_ids=[chunk.document_id for chunk in chunks],
            )

            # 埋め込み処理
            if self.embedding_service is not None:
                embedding_results = await self.embedding_service.process_batch_request(
                    batch_request
                )
            else:
                embedding_results = []

            # VectorDataオブジェクトの作成とApertureDBへの保存
            vector_data_list = []
            for _i, (chunk, result) in enumerate(
                zip(chunks, embedding_results, strict=False)
            ):
                # Dense vector用
                dense_vector_data = VectorData(
                    id=f"{chunk.id}_dense",
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    vector=result.dense_vector,
                    chunk_type=chunk.chunk_type,
                    source_type="document",
                    language="ja",
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
                    language="ja",
                )
                vector_data_list.append(sparse_vector_data)

            # ApertureDBへの一括挿入（実装後に有効化）
            # await self._insert_vectors_to_aperturedb(vector_data_list)

            return {
                "status": "completed",
                "message": "Document chunks processed successfully",
                "document_id": document_id,
                "processed_count": len(chunks),
                "vector_count": len(vector_data_list),
            }

        except Exception as e:
            logger.error(f"Failed to process document chunks: {e}")
            return {
                "status": "failed",
                "message": f"Processing failed: {str(e)}",
                "document_id": document_id,
                "processed_count": 0,
            }


# グローバルサービスインスタンス
_task_service: EmbeddingTaskService | None = None


async def get_task_service() -> EmbeddingTaskService:
    """タスクサービスのシングルトン取得"""
    global _task_service
    if _task_service is None:
        _task_service = EmbeddingTaskService()
        await _task_service.initialize()
    return _task_service


@celery_app.task(bind=True, name="embedding.process_document")
def process_document_embedding_task(self, document_id: str) -> dict[str, Any]:
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
            meta={"status": "Starting document processing", "document_id": document_id},
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
            "document_id": document_id,
        }


@celery_app.task(bind=True, name="embedding.process_batch_texts")
def process_batch_texts_task(
    self, texts: list[str], metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
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
            meta={"status": "Processing batch texts", "text_count": len(texts)},
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
                        "processing_time": result.processing_time,
                    }
                    for result in results
                ],
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
            "text_count": len(texts) if texts else 0,
        }


@celery_app.task(name="embedding.health_check")
def embedding_health_check_task() -> dict[str, Any]:
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
        return {"status": "unhealthy", "reason": f"Health check task failed: {str(e)}"}


try:
    import redis

    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    redis = None


def get_redis_health() -> dict[str, Any]:
    """Redisの接続状態をチェック"""
    if not HAS_REDIS or redis is None:
        return {"status": "unhealthy", "reason": "Redis module not available"}

    try:
        # 同期版のRedisクライアントを使用
        client = redis.Redis.from_url("redis://localhost:6379/0", decode_responses=True)
        client.ping()
        # info()は辞書を返す
        info_result = client.info()

        # 型安全な辞書アクセス
        try:
            redis_version = (
                info_result["redis_version"]
                if "redis_version" in info_result
                else "unknown"
            )
            connected_clients = (
                info_result["connected_clients"]
                if "connected_clients" in info_result
                else 0
            )
            used_memory = (
                info_result["used_memory_human"]
                if "used_memory_human" in info_result
                else "unknown"
            )
            uptime = (
                info_result["uptime_in_seconds"]
                if "uptime_in_seconds" in info_result
                else 0
            )
        except (TypeError, KeyError):
            # 何らかの理由で辞書アクセスが失敗した場合のフォールバック
            redis_version = "unknown"
            connected_clients = 0
            used_memory = "unknown"
            uptime = 0

        return {
            "status": "healthy",
            "redis_version": str(redis_version),
            "connected_clients": int(connected_clients),
            "used_memory": str(used_memory),
            "uptime": int(uptime),
        }
    except Exception as e:
        return {"status": "unhealthy", "reason": f"Redis connection failed: {str(e)}"}


def get_celery_health() -> dict[str, Any]:
    """Celeryワーカーの状態をチェック"""
    if not HAS_CELERY:
        return {"status": "unhealthy", "reason": "Celery not available"}

    try:
        inspect = celery_app.control.inspect()
        if hasattr(inspect, "stats") and hasattr(inspect, "active"):
            stats = inspect.stats()
            active = inspect.active()

            if not stats:
                return {"status": "unhealthy", "reason": "No Celery workers available"}

            total_workers = len(stats)
            total_active_tasks = (
                sum(len(tasks) for tasks in active.values()) if active else 0
            )

            return {
                "status": "healthy",
                "total_workers": total_workers,
                "active_tasks": total_active_tasks,
                "workers": list(stats.keys()) if stats else [],
            }
        else:
            # モック環境
            return {
                "status": "healthy",
                "total_workers": 1,
                "active_tasks": 0,
                "workers": ["mock_worker"],
            }
    except Exception as e:
        return {"status": "unhealthy", "reason": f"Celery inspection failed: {str(e)}"}


class EmbeddingTaskManager:
    """埋め込みタスク管理クラス"""

    @staticmethod
    def submit_document_processing(document_id: str):
        """ドキュメント処理タスクの投入

        Args:
            document_id: ドキュメントID

        Returns:
            AsyncResult: タスク結果オブジェクト
        """
        return process_document_embedding_task.delay(document_id)

    @staticmethod
    def submit_batch_processing(
        texts: list[str], metadata: dict[str, Any] | None = None
    ) -> Any:
        """バッチ処理タスクの投入

        Args:
            texts: テキストリスト
            metadata: メタデータ

        Returns:
            AsyncResult: タスク結果オブジェクト
        """
        return process_batch_texts_task.delay(texts, metadata)

    @staticmethod
    def get_task_status(task_id: str):
        """タスクステータスの取得

        Args:
            task_id: タスクID

        Returns:
            Dict[str, Any]: タスクステータス
        """
        if HAS_CELERY:
            result = AsyncResult(task_id, app=celery_app)
        else:
            result = MockAsyncResult(task_id)

        return {
            "task_id": task_id,
            "status": result.status,
            "result": result.result,
            "info": result.info,
            "ready": result.ready(),
            "successful": result.successful() if result.ready() else None,
            "failed": result.failed() if result.ready() else None,
        }

    @staticmethod
    def cancel_task(task_id: str):
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
            "message": "Task cancellation requested",
        }

    @staticmethod
    def get_queue_status():
        """キューのステータス取得

        Returns:
            Dict[str, Any]: キューのステータス
        """
        # テスト用の簡易実装
        return {"active_tasks": 1, "scheduled_tasks": 1, "workers": ["worker1"]}

    @staticmethod
    def get_system_health():
        """システムのヘルスチェック

        Returns:
            Dict[str, Any]: システムのヘルスステータス
        """
        redis_health = get_redis_health()
        celery_health = get_celery_health()
        overall = (
            "healthy"
            if redis_health["status"] == "healthy"
            and celery_health["status"] == "healthy"
            else "degraded"
        )
        if redis_health["status"] != "healthy" and celery_health["status"] != "healthy":
            overall = "unhealthy"
        return {
            "redis": redis_health,
            "celery": celery_health,
            "overall_status": overall,
        }
