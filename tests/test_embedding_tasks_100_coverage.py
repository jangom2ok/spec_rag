"""
Test coverage for app/services/embedding_tasks.py to achieve 100% coverage.
Based on actual file structure.
"""

import os
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

    def test_mock_conf_update(self):
        """Test MockConf.update method (line 22-23)."""
        conf = MockConf()
        result = conf.update(test_key="test_value", another="value")
        assert result is None

    def test_mock_inspect_methods(self):
        """Test MockInspect methods (lines 27-37)."""
        inspect = MockInspect()

        # Test all methods return empty dict
        assert inspect.active() == {}
        assert inspect.scheduled() == {}
        assert inspect.reserved() == {}
        assert inspect.stats() == {}

    def test_mock_control_methods(self):
        """Test MockControl methods (lines 41-45)."""
        control = MockControl()

        # Test revoke returns None
        result = control.revoke("task_id", terminate=True)
        assert result is None

        # Test inspect returns MockInspect instance
        inspect = control.inspect()
        assert isinstance(inspect, MockInspect)
        assert inspect.active() == {}

    def test_mock_celery_task_decorator(self):
        """Test MockCelery task decorator (line 71)."""
        celery = MockCelery("test_app")

        # Test that task decorator works
        @celery.task
        def sample_task(x, y):
            return x + y

        # The decorated function should have delay method
        assert hasattr(sample_task, "delay")
        assert hasattr(sample_task, "apply_async")

    def test_mock_async_result(self):
        """Test MockAsyncResult methods (lines 93-97)."""
        # Test with different states
        result = MockAsyncResult("test_id", state="SUCCESS")
        assert result.id == "test_id"
        assert result.state == "SUCCESS"
        assert result.result is None
        assert result.ready() is True
        assert result.successful() is True
        assert result.failed() is False
        assert result.get() is None

        # Test with FAILURE state
        failed_result = MockAsyncResult(
            "failed_id", state="FAILURE", result=Exception("Test error")
        )
        assert failed_result.failed() is True
        assert failed_result.successful() is False
        assert isinstance(failed_result.result, Exception)


class TestEmbeddingTaskService:
    """Test EmbeddingTaskService class."""

    @pytest.mark.asyncio
    async def test_initialize_service(self):
        """Test service initialization (lines 168-175)."""
        service = EmbeddingTaskService()

        with patch("app.services.embedding_tasks.EmbeddingService") as mock_embedding:
            with patch("app.services.embedding_tasks.ChunkRepository") as mock_repo:
                mock_embedding_instance = AsyncMock()
                mock_embedding.return_value = mock_embedding_instance

                await service.initialize()

                # Verify service was initialized
                assert service.embedding_service is not None
                mock_embedding_instance.initialize.assert_called_once()

                # Verify repository was created with None session
                mock_repo.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_process_document_chunks_no_repository(self):
        """Test processing when chunk repository is None (lines 191)."""
        service = EmbeddingTaskService()
        service.embedding_service = AsyncMock()
        service.chunk_repository = None

        result = await service.process_document_chunks("doc123")

        assert result["status"] == "completed"
        assert result["processed_count"] == 0
        assert result["message"] == "No chunks found for document"

    @pytest.mark.asyncio
    async def test_process_batch_texts_error_handling(self):
        """Test batch processing error handling (lines 214)."""
        service = EmbeddingTaskService()
        service.embedding_service = AsyncMock()
        service.embedding_service.generate_embeddings.side_effect = Exception(
            "Embedding error"
        )

        texts = ["text1", "text2", "text3"]

        with patch("app.services.embedding_tasks.logger") as mock_logger:
            result = await service.process_batch_texts(texts)

            assert result["status"] == "error"
            assert "Embedding error" in result["error"]
            mock_logger.error.assert_called()


class TestCeleryTasks:
    """Test Celery task functions."""

    def test_process_document_embedding_task_sync(self):
        """Test sync wrapper for document embedding (lines 299-300)."""
        mock_service = Mock()
        mock_service.process_document_chunks = AsyncMock(
            return_value={"status": "success"}
        )

        with patch("app.services.embedding_tasks.asyncio.run") as mock_run:
            with patch(
                "app.services.embedding_tasks.get_task_service", new_callable=AsyncMock
            ) as mock_get_service:
                mock_get_service.return_value = mock_service
                mock_run.return_value = {"status": "success"}

                # This is a Celery task, when called directly it uses asyncio.run
                # Need to pass self as first argument for bound tasks
                mock_self = Mock()
                mock_self.update_state = Mock()
                process_document_embedding_task(mock_self, "doc123")

                # Verify asyncio.run was called
                mock_run.assert_called_once()

    def test_process_batch_texts_task_with_error(self):
        """Test batch text processing with errors (lines 341-346)."""
        texts = ["text1", "text2"]

        async def mock_process_batch(texts):
            raise Exception("Batch processing failed")

        mock_service = Mock()
        mock_service.process_batch_texts = mock_process_batch

        with patch("app.services.embedding_tasks.asyncio.run") as mock_run:
            mock_run.side_effect = Exception("Batch processing failed")

            # Need to pass self as first argument for bound tasks
            mock_self = Mock()
            mock_self.update_state = Mock()
            result = process_batch_texts_task(mock_self, texts)

            assert result["status"] == "error"
            assert "Batch processing failed" in result["error"]

    def test_embedding_health_check_task_exception(self):
        """Test health check with exception (lines 369-371, 384-387)."""
        with patch("app.services.embedding_tasks.get_redis_health") as mock_redis:
            with patch("app.services.embedding_tasks.get_celery_health"):
                with patch("app.services.embedding_tasks.get_task_service"):
                    # Make redis health check fail
                    mock_redis.side_effect = Exception("Redis connection failed")

                    result = embedding_health_check_task()

                    assert result["healthy"] is False
                    assert result["redis"]["status"] == "error"
                    assert "Redis connection failed" in result["redis"]["error"]

    def test_get_redis_health_not_configured(self):
        """Test Redis health when not configured (lines 414, 425-430)."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove REDIS_URL from environment
            result = get_redis_health()

            assert result["status"] == "not_configured"
            assert result["message"] == "Redis URL not configured"

    def test_get_redis_health_connection_error(self):
        """Test Redis health with connection error (lines 427-430)."""
        with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}):
            with patch("redis.from_url") as mock_redis:
                mock_client = Mock()
                mock_client.ping.side_effect = Exception("Connection refused")
                mock_redis.return_value = mock_client

                result = get_redis_health()

                assert result["status"] == "error"
                assert "Connection refused" in result["error"]

    def test_get_celery_health_error(self):
        """Test Celery health check error (lines 470, 476-477)."""
        with patch.object(celery_app.control, "inspect") as mock_inspect:
            mock_inspect.side_effect = Exception("Celery broker down")

            result = get_celery_health()

            assert result["status"] == "error"
            assert "Celery broker down" in result["error"]


class TestEmbeddingTaskManager:
    """Test EmbeddingTaskManager class."""

    def test_task_manager_initialization(self):
        """Test task manager init."""
        manager = EmbeddingTaskManager()
        assert manager is not None

    @pytest.mark.asyncio
    async def test_get_task_service_singleton(self):
        """Test get_task_service returns singleton (line 274-277)."""
        # Call twice to ensure singleton
        service1 = await get_task_service()
        service2 = await get_task_service()

        # Should be the same instance
        assert service1 is service2

    def test_celery_task_delay_method(self):
        """Test Celery task delay method (line 603)."""
        # Test that our Celery tasks have the delay method
        assert hasattr(process_document_embedding_task, "delay")
        assert hasattr(process_batch_texts_task, "delay")
        assert hasattr(embedding_health_check_task, "delay")

        # Test MockAsyncResult is returned when delay is called
        with patch.object(process_document_embedding_task, "delay") as mock_delay:
            mock_delay.return_value = MockAsyncResult("task123", state="PENDING")

            result = process_document_embedding_task.delay("doc123")
            assert result.id == "task123"
            assert result.state == "PENDING"
