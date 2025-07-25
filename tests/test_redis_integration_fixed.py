"""Redis統合テスト（修正版）

EmbeddingTasksとRedisの統合テスト - 外部依存をモックで解決
"""

from unittest.mock import Mock, patch

import pytest

from app.services.embedding_tasks import (
    EmbeddingTaskManager,
    get_celery_health,
    get_redis_health,
)


class TestRedisIntegration:
    """Redis統合テスト"""

    @pytest.mark.asyncio
    async def test_redis_health_check_success(self, mock_redis_client):
        """Redis接続成功時のヘルスチェック"""
        with patch("app.services.embedding_tasks.redis") as mock_redis_module:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_client.info.return_value = {
                "redis_version": "7.0.0",
                "connected_clients": 5,
                "used_memory_human": "1.2M",
                "uptime_in_seconds": 3600,
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
                "worker2": {"pool": {"max-concurrency": 4}},
            }
            mock_inspect.active.return_value = {
                "worker1": [{"id": "task1"}, {"id": "task2"}],
                "worker2": [],
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
    async def test_submit_document_processing_task(self, mock_celery_app):
        """ドキュメント処理タスクの投入テスト"""
        # Mock the task and its result
        mock_task_result = Mock()
        mock_task_result.id = "task_12345"

        with patch("app.services.embedding_tasks.HAS_CELERY", True):
            with patch(
                "app.services.embedding_tasks.process_document_embedding_task"
            ) as mock_task:
                mock_task.delay.return_value = mock_task_result

                result = EmbeddingTaskManager.submit_document_processing("doc_001")

                assert result.id == "task_12345"
                mock_task.delay.assert_called_once_with("doc_001")

    @pytest.mark.asyncio
    async def test_submit_batch_processing_task(self, mock_celery_app):
        """バッチ処理タスクの投入テスト"""
        # Mock the task and its result
        mock_task_result = Mock()
        mock_task_result.id = "batch_task_67890"

        with patch("app.services.embedding_tasks.HAS_CELERY", True):
            with patch(
                "app.services.embedding_tasks.process_batch_texts_task"
            ) as mock_task:
                mock_task.delay.return_value = mock_task_result

                texts = ["text1", "text2", "text3"]
                metadata = {
                    "chunk_ids": ["chunk1", "chunk2", "chunk3"],
                    "document_ids": ["doc1", "doc1", "doc2"],
                }

                result = EmbeddingTaskManager.submit_batch_processing(texts, metadata)

                assert result.id == "batch_task_67890"
                mock_task.delay.assert_called_once_with(texts, metadata)

    @pytest.mark.asyncio
    async def test_get_task_status(self, mock_celery_app):
        """タスク状態取得テスト"""
        import os
        with patch.dict(os.environ, {"TESTING": "false"}):
            with patch("app.services.embedding_tasks.HAS_CELERY", True):
                with patch("app.services.embedding_tasks.AsyncResult") as mock_async_result:
                    mock_result = Mock()
                    mock_result.status = "SUCCESS"
                    mock_result.result = {"processed_count": 5}
                    mock_result.ready.return_value = True
                    mock_result.successful.return_value = True
                    mock_async_result.return_value = mock_result

                    status = EmbeddingTaskManager.get_task_status("task_12345")

                    assert status["status"] == "SUCCESS"
                    assert status["result"] == {"processed_count": 5}
                    assert status["ready"] is True
                    assert status["successful"] is True

    @pytest.mark.asyncio
    async def test_cancel_task(self, mock_celery_app):
        """タスクキャンセルテスト"""
        with patch("app.services.embedding_tasks.HAS_CELERY", True):
            with patch(
                "app.services.embedding_tasks.celery_app"
            ) as mock_celery_app_inner:
                mock_celery_app_inner.control.revoke.return_value = None

                result = EmbeddingTaskManager.cancel_task("task_12345")

                assert result["status"] == "cancelled"
                assert result["task_id"] == "task_12345"
                mock_celery_app_inner.control.revoke.assert_called_once_with(
                    "task_12345", terminate=True
                )

    @pytest.mark.asyncio
    async def test_submit_document_processing_task_no_celery(self):
        """Celery未使用時のドキュメント処理タスク"""
        with patch("app.services.embedding_tasks.HAS_CELERY", False):
            # Should use mock implementation
            result = EmbeddingTaskManager.submit_document_processing("doc_001")

            assert hasattr(result, "id")
            assert result.id.startswith("mock-task-")

    @pytest.mark.asyncio
    async def test_get_task_status_no_celery(self):
        """Celery未使用時のタスク状態取得"""
        with patch("app.services.embedding_tasks.HAS_CELERY", False):
            # Mock task should always be successful
            status = EmbeddingTaskManager.get_task_status("mock-task-12345")

            assert status["status"] == "SUCCESS"
            assert status["ready"] is True
            assert status["successful"] is True


class TestQueueProcessing:
    """キュー処理テスト"""

    @pytest.mark.asyncio
    async def test_queue_monitoring(self, mock_celery_app):
        """キュー監視機能テスト"""
        with patch("app.services.embedding_tasks.HAS_CELERY", True):
            with patch(
                "app.services.embedding_tasks.celery_app"
            ) as mock_celery_app_inner:
                mock_inspect = Mock()
                mock_inspect.active.return_value = {
                    "worker1": [{"id": "task1", "name": "process_document_embedding"}]
                }
                mock_inspect.scheduled.return_value = {
                    "worker1": [{"id": "task2", "name": "process_batch_embedding"}]
                }
                mock_inspect.reserved.return_value = {"worker1": []}
                mock_celery_app_inner.control.inspect.return_value = mock_inspect

                # Get queue status
                queue_status = EmbeddingTaskManager.get_queue_status()

                assert queue_status is not None
                assert "active_tasks" in queue_status
                assert "scheduled_tasks" in queue_status
                assert "workers" in queue_status
                assert queue_status["active_tasks"] == 1
                assert queue_status["scheduled_tasks"] == 1
                assert "worker1" in queue_status["workers"]

    @pytest.mark.asyncio
    async def test_get_worker_status(self, mock_celery_app):
        """ワーカーステータス取得テスト"""
        import os
        with patch.dict(os.environ, {"TESTING": "false"}):
            with patch("app.services.embedding_tasks.HAS_CELERY", True):
                with patch(
                    "app.services.embedding_tasks.celery_app"
                ) as mock_celery_app_inner:
                    mock_inspect = Mock()
                    mock_inspect.active.return_value = {}
                    mock_inspect.scheduled.return_value = {}
                    mock_inspect.reserved.return_value = {}
                    mock_inspect.stats.return_value = {
                        "worker1": {
                            "pool": {"max-concurrency": 4, "processes": [1234, 5678]},
                            "total": 100,
                        }
                    }
                    mock_celery_app_inner.control.inspect.return_value = mock_inspect

                    worker_status = EmbeddingTaskManager.get_worker_status()

                    assert worker_status is not None
                    assert "stats" in worker_status
                    assert "worker1" in worker_status["stats"]
                    assert worker_status["stats"]["worker1"]["pool"]["max-concurrency"] == 4
                    assert worker_status["stats"]["worker1"]["total"] == 100


# Ensure fixtures from conftest_extended.py are available
pytest_plugins = ["conftest_extended"]
