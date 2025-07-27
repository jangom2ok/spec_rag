"""
Test coverage for app/services/embedding_tasks.py to achieve 100% coverage.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.services.embedding_tasks import (
    EmbeddingTaskManager,
    EmbeddingTaskService,
    MockAsyncResult,
    MockCelery,
    MockConf,
    MockControl,
    MockInspect,
    celery_app,
    embedding_health_check_task,
    get_celery_health,
    get_redis_health,
    get_task_service,
    process_batch_texts_task,
    process_document_embedding_task,
)


class TestMockClasses:
    """Test mock classes for when Celery is not available."""

    def test_mock_conf(self):
        """Test MockConf class."""
        conf = MockConf()
        result = conf.update(test="value")
        assert result is None

    def test_mock_inspect(self):
        """Test MockInspect class."""
        inspect = MockInspect()

        assert inspect.active() == {}
        assert inspect.scheduled() == {}
        assert inspect.reserved() == {}
        assert inspect.stats() == {}

    def test_mock_control(self):
        """Test MockControl class."""
        control = MockControl()

        # Test revoke
        assert control.revoke("task_id", terminate=True) is None

        # Test inspect
        inspect = control.inspect()
        assert isinstance(inspect, MockInspect)

    def test_mock_celery(self):
        """Test MockCelery class."""
        celery = MockCelery()

        # Test conf property
        assert isinstance(celery.conf, MockConf)

        # Test task decorator
        @celery.task
        def test_task():
            return "test"

        # Test that delay is added to the decorated function
        result = test_task.delay()
        assert hasattr(result, "id")
        assert hasattr(result, "state")
        assert result.state == "PENDING"
        assert result.id.startswith("mock-task-")

        # Test control property
        assert isinstance(celery.control, MockControl)

    def test_mock_async_result(self):
        """Test MockAsyncResult class."""
        result = MockAsyncResult("test_id")

        assert result.status == "SUCCESS"
        assert result.result == {}
        assert result.info is None
        assert result.id == "mock_task_id"
        assert result.ready() is True
        assert result.successful() is True
        assert result.failed() is False


class TestEmbeddingTaskService:
    """Test EmbeddingTaskService class."""

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test service initialization."""
        service = EmbeddingTaskService()

        with patch("app.services.embedding_tasks.EmbeddingService") as mock_embedding:
            with patch("app.services.embedding_tasks.ChunkRepository"):
                mock_embedding_instance = Mock()
                mock_embedding_instance.initialize = AsyncMock()
                mock_embedding.return_value = mock_embedding_instance

                await service.initialize()

                assert service.embedding_service is not None
                assert service.chunk_repository is not None
                mock_embedding_instance.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_document_chunks_no_chunks(self):
        """Test processing with no chunks."""
        service = EmbeddingTaskService()
        service.chunk_repository = Mock()
        service.chunk_repository.get_by_document_id = AsyncMock(return_value=[])

        result = await service.process_document_chunks("test_doc_id")

        assert result["status"] == "completed"
        assert result["processed_count"] == 0
        assert result["message"] == "No chunks found for document"

    @pytest.mark.asyncio
    async def test_process_document_chunks_with_chunks(self):
        """Test processing with chunks."""
        service = EmbeddingTaskService()

        # Mock chunks
        mock_chunk = Mock()
        mock_chunk.id = "chunk_1"
        mock_chunk.document_id = "doc_1"
        mock_chunk.content = "test content"
        mock_chunk.chunk_type = "text"

        service.chunk_repository = Mock()
        service.chunk_repository.get_by_document_id = AsyncMock(
            return_value=[mock_chunk]
        )

        # Mock embedding service
        mock_result = Mock()
        mock_result.dense_vector = [0.1, 0.2, 0.3]
        mock_result.sparse_vector = {"token1": 0.5}

        service.embedding_service = Mock()
        service.embedding_service.process_batch_request = AsyncMock(
            return_value=[mock_result]
        )

        result = await service.process_document_chunks("test_doc_id")

        assert result["status"] == "completed"
        assert result["processed_count"] == 1
        assert result["vector_count"] == 2  # dense + sparse

    @pytest.mark.asyncio
    async def test_process_document_chunks_exception(self):
        """Test processing with exception."""
        service = EmbeddingTaskService()
        service.chunk_repository = Mock()
        service.chunk_repository.get_by_document_id = AsyncMock(
            side_effect=Exception("Test error")
        )

        result = await service.process_document_chunks("test_doc_id")

        assert result["status"] == "failed"
        assert "Test error" in result["message"]

    @pytest.mark.asyncio
    async def test_get_task_service(self):
        """Test get_task_service singleton."""
        with patch("app.services.embedding_tasks._task_service", None):
            with patch.object(
                EmbeddingTaskService, "initialize", new_callable=AsyncMock
            ):
                service1 = await get_task_service()
                service2 = await get_task_service()

                assert service1 is service2


class TestCeleryTasks:
    """Test Celery tasks."""

    def test_process_document_embedding_task(self):
        """Test document embedding task."""
        with patch("app.services.embedding_tasks.get_task_service") as mock_get_service:
            mock_service = Mock()
            mock_service.process_document_chunks = AsyncMock(
                return_value={"status": "completed", "processed_count": 5}
            )
            mock_get_service.return_value = mock_service

            # Mock self parameter
            mock_self = Mock()
            mock_self.update_state = Mock()

            result = process_document_embedding_task(mock_self, "test_doc_id")

            assert result["status"] == "completed"
            mock_self.update_state.assert_called()

    def test_process_document_embedding_task_exception(self):
        """Test document embedding task with exception."""
        with patch("app.services.embedding_tasks.get_task_service") as mock_get_service:
            mock_get_service.side_effect = Exception("Service error")

            mock_self = Mock()
            mock_self.update_state = Mock()

            result = process_document_embedding_task(mock_self, "test_doc_id")

            assert result["status"] == "failed"
            assert "Service error" in result["message"]

    def test_process_batch_texts_task(self):
        """Test batch texts task."""
        with patch("app.services.embedding_tasks.get_task_service") as mock_get_service:
            mock_service = Mock()
            mock_embedding_service = Mock()
            mock_result = Mock()
            mock_result.dense_vector = [0.1, 0.2]
            mock_result.sparse_vector = {"token": 0.5}
            mock_result.processing_time = 0.1

            mock_embedding_service.embed_batch = AsyncMock(return_value=[mock_result])
            mock_service.embedding_service = mock_embedding_service
            mock_get_service.return_value = mock_service

            mock_self = Mock()
            mock_self.update_state = Mock()

            result = process_batch_texts_task(mock_self, ["test text"])

            assert result["status"] == "completed"
            assert result["text_count"] == 1
            assert len(result["results"]) == 1

    def test_process_batch_texts_task_no_service(self):
        """Test batch texts task with no embedding service."""
        with patch("app.services.embedding_tasks.get_task_service") as mock_get_service:
            mock_service = Mock()
            mock_service.embedding_service = None
            mock_get_service.return_value = mock_service

            mock_self = Mock()
            mock_self.update_state = Mock()

            result = process_batch_texts_task(mock_self, ["test text"])

            assert result["status"] == "failed"
            assert "not initialized" in result["message"]

    def test_embedding_health_check_task(self):
        """Test health check task."""
        with patch("app.services.embedding_tasks.get_task_service") as mock_get_service:
            mock_service = Mock()
            mock_embedding_service = Mock()
            mock_embedding_service.health_check = AsyncMock(
                return_value={"status": "healthy"}
            )
            mock_service.embedding_service = mock_embedding_service
            mock_get_service.return_value = mock_service

            result = embedding_health_check_task()

            assert result["status"] == "healthy"

    def test_embedding_health_check_task_exception(self):
        """Test health check task with exception."""
        with patch("app.services.embedding_tasks.get_task_service") as mock_get_service:
            mock_get_service.side_effect = Exception("Health check error")

            result = embedding_health_check_task()

            assert result["status"] == "unhealthy"
            assert "Health check error" in result["reason"]


class TestHealthChecks:
    """Test health check functions."""

    def test_get_redis_health_no_redis(self):
        """Test Redis health when Redis is not available."""
        with patch("app.services.embedding_tasks.HAS_REDIS", False):
            result = get_redis_health()

            assert result["status"] == "unhealthy"
            assert "not available" in result["reason"]

    def test_get_redis_health_success(self):
        """Test Redis health check success."""
        with patch("app.services.embedding_tasks.HAS_REDIS", True):
            with patch("app.services.embedding_tasks.redis") as mock_redis:
                mock_client = Mock()
                mock_client.ping = Mock()
                mock_client.info = Mock(
                    return_value={
                        "redis_version": "6.2.0",
                        "connected_clients": 5,
                        "used_memory_human": "1.5M",
                        "uptime_in_seconds": 3600,
                    }
                )
                mock_redis.Redis.from_url.return_value = mock_client

                result = get_redis_health()

                assert result["status"] == "healthy"
                assert result["redis_version"] == "6.2.0"
                assert result["connected_clients"] == 5

    def test_get_redis_health_exception(self):
        """Test Redis health check with exception."""
        with patch("app.services.embedding_tasks.HAS_REDIS", True):
            with patch("app.services.embedding_tasks.redis") as mock_redis:
                mock_redis.Redis.from_url.side_effect = Exception("Connection failed")

                result = get_redis_health()

                assert result["status"] == "unhealthy"
                assert "Connection failed" in result["reason"]

    def test_get_celery_health_no_celery(self):
        """Test Celery health when Celery is not available."""
        with patch("app.services.embedding_tasks.HAS_CELERY", False):
            result = get_celery_health()

            assert result["status"] == "unhealthy"
            assert "not available" in result["reason"]

    def test_get_celery_health_success(self):
        """Test Celery health check success."""
        with patch("app.services.embedding_tasks.HAS_CELERY", True):
            mock_inspect = Mock()
            mock_inspect.stats = Mock(return_value={"worker1": {"stats": "data"}})
            mock_inspect.active = Mock(return_value={"worker1": ["task1", "task2"]})

            with patch.object(celery_app.control, "inspect", return_value=mock_inspect):
                result = get_celery_health()

                assert result["status"] == "healthy"
                assert result["total_workers"] == 1
                assert result["active_tasks"] == 2
                assert "worker1" in result["workers"]

    def test_get_celery_health_no_workers(self):
        """Test Celery health check with no workers."""
        with patch("app.services.embedding_tasks.HAS_CELERY", True):
            mock_inspect = Mock()
            mock_inspect.stats = Mock(return_value=None)

            with patch.object(celery_app.control, "inspect", return_value=mock_inspect):
                result = get_celery_health()

                assert result["status"] == "unhealthy"
                assert "No Celery workers" in result["reason"]


class TestEmbeddingTaskManager:
    """Test EmbeddingTaskManager class."""

    def test_submit_document_processing(self):
        """Test submit document processing."""
        with patch.object(process_document_embedding_task, "delay") as mock_delay:
            mock_delay.return_value = Mock(id="task_123")

            EmbeddingTaskManager.submit_document_processing("doc_123")

            mock_delay.assert_called_once_with("doc_123")

    def test_submit_batch_processing(self):
        """Test submit batch processing."""
        with patch.object(process_batch_texts_task, "delay") as mock_delay:
            mock_delay.return_value = Mock(id="task_456")

            EmbeddingTaskManager.submit_batch_processing(
                ["text1", "text2"], {"meta": "data"}
            )

            mock_delay.assert_called_once_with(["text1", "text2"], {"meta": "data"})

    def test_get_task_status(self):
        """Test get task status."""
        with patch("app.services.embedding_tasks.AsyncResult") as mock_async_result:
            mock_result = Mock()
            mock_result.status = "SUCCESS"
            mock_result.result = {"data": "result"}
            mock_result.info = None
            mock_result.ready = Mock(return_value=True)
            mock_result.successful = Mock(return_value=True)
            mock_result.failed = Mock(return_value=False)
            mock_async_result.return_value = mock_result

            status = EmbeddingTaskManager.get_task_status("task_123")

            assert status["task_id"] == "task_123"
            assert status["status"] == "SUCCESS"
            assert status["ready"] is True
            assert status["successful"] is True

    def test_cancel_task(self):
        """Test cancel task."""
        with patch.object(celery_app.control, "revoke") as mock_revoke:
            result = EmbeddingTaskManager.cancel_task("task_123")

            mock_revoke.assert_called_once_with("task_123", terminate=True)
            assert result["status"] == "cancelled"

    def test_get_queue_status(self):
        """Test get queue status."""
        result = EmbeddingTaskManager.get_queue_status()

        assert "active_tasks" in result
        assert "scheduled_tasks" in result
        assert "workers" in result

    def test_get_worker_status(self):
        """Test get worker status."""
        with patch("app.services.embedding_tasks.HAS_CELERY", True):
            mock_inspect = Mock()
            mock_inspect.active = Mock(return_value={"worker1": []})
            mock_inspect.scheduled = Mock(return_value={})
            mock_inspect.reserved = Mock(return_value={})
            mock_inspect.stats = Mock(return_value={"worker1": {}})

            with patch.object(celery_app.control, "inspect", return_value=mock_inspect):
                result = EmbeddingTaskManager.get_worker_status()

                assert "active_tasks" in result
                assert "scheduled_tasks" in result
                assert "reserved_tasks" in result
                assert "stats" in result

    def test_get_system_health(self):
        """Test get system health."""
        with patch("app.services.embedding_tasks.get_redis_health") as mock_redis:
            with patch("app.services.embedding_tasks.get_celery_health") as mock_celery:
                mock_redis.return_value = {"status": "healthy"}
                mock_celery.return_value = {"status": "healthy"}

                result = EmbeddingTaskManager.get_system_health()

                assert result["overall_status"] == "healthy"
                assert result["redis"]["status"] == "healthy"
                assert result["celery"]["status"] == "healthy"

    def test_get_system_health_degraded(self):
        """Test get system health degraded."""
        with patch("app.services.embedding_tasks.get_redis_health") as mock_redis:
            with patch("app.services.embedding_tasks.get_celery_health") as mock_celery:
                mock_redis.return_value = {"status": "unhealthy"}
                mock_celery.return_value = {"status": "healthy"}

                result = EmbeddingTaskManager.get_system_health()

                assert result["overall_status"] == "degraded"

    def test_get_system_health_unhealthy(self):
        """Test get system health unhealthy."""
        with patch("app.services.embedding_tasks.get_redis_health") as mock_redis:
            with patch("app.services.embedding_tasks.get_celery_health") as mock_celery:
                mock_redis.return_value = {"status": "unhealthy"}
                mock_celery.return_value = {"status": "unhealthy"}

                result = EmbeddingTaskManager.get_system_health()

                assert result["overall_status"] == "unhealthy"
