"""Redis統合テスト

EmbeddingTasksとRedisの統合テスト
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from app.services.embedding_tasks import (
    get_redis_health,
    get_celery_health,
    EmbeddingTaskManager
)


class TestRedisIntegration:
    """Redis統合テスト"""

    @pytest.mark.asyncio
    async def test_redis_health_check_success(self):
        """Redis接続成功時のヘルスチェック"""
        with patch("app.services.embedding_tasks.redis") as mock_redis_module:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_client.info.return_value = {
                "redis_version": "7.0.0",
                "connected_clients": 5,
                "used_memory_human": "1.2M",
                "uptime_in_seconds": 3600
            }
            mock_redis_module.Redis.from_url.return_value = mock_client

            with patch("app.services.embedding_tasks.HAS_REDIS", True):
                result = get_redis_health()

            assert result["status"] == "healthy"
            assert result["redis_version"] == "7.0.0"
            assert result["connected_clients"] == 5
            assert result["used_memory"] == "1.2M"
            assert result["uptime"] == 3600

    @pytest.mark.asyncio
    async def test_redis_health_check_connection_failure(self):
        """Redis接続失敗時のヘルスチェック"""
        with patch("app.services.embedding_tasks.redis") as mock_redis_module:
            mock_client = Mock()
            mock_client.ping.side_effect = Exception("Connection refused")
            mock_redis_module.Redis.from_url.return_value = mock_client

            with patch("app.services.embedding_tasks.HAS_REDIS", True):
                result = get_redis_health()

            assert result["status"] == "unhealthy"
            assert "Connection refused" in result["reason"]

    @pytest.mark.asyncio
    async def test_redis_health_check_module_unavailable(self):
        """Redisモジュール未インストール時のヘルスチェック"""
        with patch("app.services.embedding_tasks.HAS_REDIS", False):
            result = get_redis_health()

        assert result["status"] == "unhealthy"
        assert "Redis module not available" in result["reason"]


class TestCeleryIntegration:
    """Celery統合テスト"""

    @pytest.mark.asyncio
    async def test_celery_health_check_success(self):
        """Celery接続成功時のヘルスチェック"""
        with patch("app.services.embedding_tasks.celery_app") as mock_celery_app:
            mock_inspect = Mock()
            mock_inspect.stats.return_value = {
                "worker1": {"pool": {"max-concurrency": 4}},
                "worker2": {"pool": {"max-concurrency": 4}}
            }
            mock_inspect.active.return_value = {
                "worker1": [{"id": "task1"}, {"id": "task2"}],
                "worker2": []
            }
            mock_celery_app.control.inspect.return_value = mock_inspect

            with patch("app.services.embedding_tasks.HAS_CELERY", True):
                result = get_celery_health()

            assert result["status"] == "healthy"
            assert result["total_workers"] == 2
            assert result["active_tasks"] == 2
            assert len(result["workers"]) == 2

    @pytest.mark.asyncio
    async def test_celery_health_check_no_workers(self):
        """Celeryワーカー未起動時のヘルスチェック"""
        with patch("app.services.embedding_tasks.celery_app") as mock_celery_app:
            mock_inspect = Mock()
            mock_inspect.stats.return_value = {}
            mock_inspect.active.return_value = {}
            mock_celery_app.control.inspect.return_value = mock_inspect

            with patch("app.services.embedding_tasks.HAS_CELERY", True):
                result = get_celery_health()

            assert result["status"] == "unhealthy"
            assert "No Celery workers available" in result["reason"]

    @pytest.mark.asyncio
    async def test_celery_health_check_module_unavailable(self):
        """Celeryモジュール未インストール時のヘルスチェック"""
        with patch("app.services.embedding_tasks.HAS_CELERY", False):
            result = get_celery_health()

        assert result["status"] == "unhealthy"
        assert "Celery not available" in result["reason"]


class TestEmbeddingTaskManager:
    """EmbeddingTaskManagerのテスト"""

    @pytest.mark.asyncio
    async def test_submit_document_processing_task(self):
        """ドキュメント処理タスクの投入テスト"""
        with patch("app.services.embedding_tasks.process_document_embedding") as mock_task:
            mock_result = Mock()
            mock_result.id = "task_12345"
            mock_task.delay.return_value = mock_result

            result = EmbeddingTaskManager.submit_document_processing("doc_001")

            assert result.id == "task_12345"
            mock_task.delay.assert_called_once_with("doc_001")

    @pytest.mark.asyncio
    async def test_submit_batch_processing_task(self):
        """バッチ処理タスクの投入テスト"""
        with patch("app.services.embedding_tasks.process_batch_embedding") as mock_task:
            mock_result = Mock()
            mock_result.id = "batch_task_67890"
            mock_task.delay.return_value = mock_result

            batch_request = {
                "texts": ["text1", "text2", "text3"],
                "chunk_ids": ["chunk1", "chunk2", "chunk3"],
                "document_ids": ["doc1", "doc1", "doc2"]
            }

            result = EmbeddingTaskManager.submit_batch_processing(batch_request)

            assert result.id == "batch_task_67890"
            mock_task.delay.assert_called_once_with(batch_request)

    @pytest.mark.asyncio
    async def test_get_task_status(self):
        """タスク状態取得テスト"""
        with patch("app.services.embedding_tasks.AsyncResult") as mock_async_result:
            mock_result = Mock()
            mock_result.status = "SUCCESS"
            mock_result.result = {"processed_count": 5}
            mock_result.ready.return_value = True
            mock_result.successful.return_value = True
            mock_async_result.return_value = mock_result

            status = EmbeddingTaskManager.get_task_status("task_12345")

            assert status["status"] == "SUCCESS"
            assert status["result"]["processed_count"] == 5
            assert status["ready"] is True
            assert status["successful"] is True

    @pytest.mark.asyncio
    async def test_cancel_task(self):
        """タスクキャンセルテスト"""
        with patch("app.services.embedding_tasks.celery_app") as mock_celery_app:
            mock_celery_app.control.revoke.return_value = None

            result = EmbeddingTaskManager.cancel_task("task_12345")

            assert result["status"] == "cancelled"
            assert result["task_id"] == "task_12345"
            mock_celery_app.control.revoke.assert_called_once_with("task_12345", terminate=True)


class TestQueueProcessing:
    """キュー処理テスト"""

    @pytest.mark.asyncio
    async def test_queue_monitoring(self):
        """キュー監視機能テスト"""
        with patch("app.services.embedding_tasks.celery_app") as mock_celery_app:
            mock_inspect = Mock()
            mock_inspect.active.return_value = {
                "worker1": [{"id": "task1", "name": "process_document_embedding"}]
            }
            mock_inspect.scheduled.return_value = {
                "worker1": [{"id": "task2", "name": "process_batch_embedding"}]
            }
            mock_celery_app.control.inspect.return_value = mock_inspect

            status = EmbeddingTaskManager.get_queue_status()

            assert status["active_tasks"] == 1
            assert status["scheduled_tasks"] == 1
            assert len(status["workers"]) >= 1

    @pytest.mark.asyncio
    async def test_health_check_integration(self):
        """統合ヘルスチェックテスト"""
        with patch("app.services.embedding_tasks.get_redis_health") as mock_redis_health:
            with patch("app.services.embedding_tasks.get_celery_health") as mock_celery_health:
                mock_redis_health.return_value = {"status": "healthy"}
                mock_celery_health.return_value = {"status": "healthy"}

                health = EmbeddingTaskManager.get_system_health()

                assert health["redis"]["status"] == "healthy"
                assert health["celery"]["status"] == "healthy"
                assert health["overall_status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_with_failures(self):
        """障害時の統合ヘルスチェックテスト"""
        with patch("app.services.embedding_tasks.get_redis_health") as mock_redis_health:
            with patch("app.services.embedding_tasks.get_celery_health") as mock_celery_health:
                mock_redis_health.return_value = {"status": "unhealthy", "reason": "Connection failed"}
                mock_celery_health.return_value = {"status": "healthy"}

                health = EmbeddingTaskManager.get_system_health()

                assert health["redis"]["status"] == "unhealthy"
                assert health["celery"]["status"] == "healthy"
                assert health["overall_status"] == "degraded"
