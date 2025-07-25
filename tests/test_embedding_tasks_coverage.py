"""
Test coverage for app/services/embedding_tasks.py to achieve 100% coverage.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio
from typing import Any

from app.services.embedding_tasks import (
    MockConf,
    MockInspect,
    MockControl,
    MockCelery,
    MockTask,
    celery_app,
    is_celery_available,
    get_celery_stats,
    process_embeddings_task,
    batch_process_embeddings_task,
    check_task_status,
    revoke_task,
    retry_failed_tasks,
    process_embeddings_async,
    batch_process_embeddings_async,
    update_embeddings_task,
    delete_embeddings_task,
    reindex_embeddings_task,
    check_embedding_health,
    process_embeddings_with_priority,
    monitor_embedding_queue,
    cleanup_old_embeddings,
    export_embeddings_task,
    import_embeddings_task,
    validate_embeddings_task,
    EmbeddingTaskResult,
    TaskPriority,
)
from app.services.embedding_service import (
    BatchEmbeddingRequest,
    EmbeddingConfig,
    EmbeddingService,
)
from app.models.aperturedb import VectorData


class TestMockClasses:
    """Test mock classes for when Celery is not available."""

    def test_mock_conf(self):
        """Test MockConf class (lines 22-23)."""
        conf = MockConf()
        result = conf.update(test="value")
        assert result is None

    def test_mock_inspect(self):
        """Test MockInspect class (lines 27-37)."""
        inspect = MockInspect()
        
        assert inspect.active() == {}
        assert inspect.scheduled() == {}
        assert inspect.reserved() == {}
        assert inspect.stats() == {}

    def test_mock_control(self):
        """Test MockControl class (lines 41-45)."""
        control = MockControl()
        
        # Test revoke
        result = control.revoke("task_id", terminate=True)
        assert result is None
        
        # Test inspect
        inspect = control.inspect()
        assert isinstance(inspect, MockInspect)

    def test_mock_celery(self):
        """Test MockCelery class (lines 71)."""
        # Test task decorator
        celery = MockCelery("test_app")
        
        @celery.task
        def test_task():
            return "test"
        
        assert hasattr(test_task, "delay")


class TestCeleryIntegration:
    """Test Celery integration functions."""

    def test_is_celery_available_false(self):
        """Test when Celery is not available (lines 93-97)."""
        # Mock celery_app.control.inspect to return None
        with patch.object(celery_app.control, 'inspect', return_value=None):
            assert not is_celery_available()

    def test_is_celery_available_exception(self):
        """Test when Celery check raises exception (lines 96-97)."""
        with patch.object(celery_app.control, 'inspect', side_effect=Exception("Connection error")):
            assert not is_celery_available()

    def test_get_celery_stats_not_available(self):
        """Test get stats when Celery not available (lines 111-120)."""
        with patch("app.services.embedding_tasks.is_celery_available", return_value=False):
            stats = get_celery_stats()
            
            assert stats["available"] is False
            assert stats["active_tasks"] == 0
            assert stats["scheduled_tasks"] == 0
            assert stats["reserved_tasks"] == 0
            assert stats["workers"] == 0

    def test_get_celery_stats_with_active_tasks(self):
        """Test get stats with active tasks (lines 113-118)."""
        mock_inspect = Mock()
        mock_inspect.active.return_value = {
            "worker1": [{"id": "task1"}, {"id": "task2"}],
            "worker2": [{"id": "task3"}]
        }
        mock_inspect.scheduled.return_value = {"worker1": [{"id": "task4"}]}
        mock_inspect.reserved.return_value = {}
        mock_inspect.stats.return_value = {"worker1": {}, "worker2": {}}
        
        with patch.object(celery_app.control, 'inspect', return_value=mock_inspect):
            with patch("app.services.embedding_tasks.is_celery_available", return_value=True):
                stats = get_celery_stats()
                
                assert stats["available"] is True
                assert stats["active_tasks"] == 3
                assert stats["scheduled_tasks"] == 1
                assert stats["workers"] == 2


class TestEmbeddingTasks:
    """Test embedding task functions."""

    @pytest.mark.asyncio
    async def test_process_embeddings_task_direct_call(self):
        """Test direct task call when not Celery worker (line 133)."""
        with patch("app.services.embedding_tasks.is_celery_worker", return_value=False):
            with patch("app.services.embedding_tasks.process_embeddings_async") as mock_async:
                mock_async.return_value = {"status": "success"}
                
                result = process_embeddings_task("test_doc", {"content": "test"})
                
                # Should call async version via asyncio.run
                assert result == {"status": "success"}

    def test_batch_process_embeddings_error_handling(self):
        """Test batch processing error handling (lines 168-175)."""
        documents = [
            {"id": "doc1", "content": "test1"},
            {"id": "doc2", "content": "test2"}
        ]
        
        with patch("app.services.embedding_tasks.process_embeddings_async") as mock_process:
            # Make first doc succeed, second fail
            mock_process.side_effect = [
                {"status": "success", "document_id": "doc1"},
                Exception("Processing failed")
            ]
            
            results = batch_process_embeddings_task(documents)
            
            assert len(results) == 2
            assert results[0]["status"] == "success"
            assert results[1]["status"] == "error"
            assert "Processing failed" in results[1]["error"]

    def test_check_task_status_not_found(self):
        """Test checking status of non-existent task (line 191)."""
        mock_result = Mock()
        mock_result.state = None
        
        with patch("app.services.embedding_tasks.AsyncResult", return_value=mock_result):
            status = check_task_status("nonexistent_task_id")
            
            assert status["status"] == "UNKNOWN"
            assert status["task_id"] == "nonexistent_task_id"

    def test_revoke_task_success(self):
        """Test revoking a task (line 214)."""
        with patch.object(celery_app.control, 'revoke') as mock_revoke:
            result = revoke_task("task_id_to_revoke", terminate=True)
            
            assert result["success"] is True
            assert result["task_id"] == "task_id_to_revoke"
            mock_revoke.assert_called_once_with("task_id_to_revoke", terminate=True)

    @pytest.mark.asyncio
    async def test_process_embeddings_async_with_existing_vector(self):
        """Test processing with existing vector data (lines 274-277)."""
        mock_service = AsyncMock()
        mock_embedding = Mock()
        mock_embedding.dense = [0.1, 0.2, 0.3]
        mock_embedding.sparse = {"indices": [1, 2], "values": [0.5, 0.6]}
        mock_service.generate_embeddings.return_value = [mock_embedding]
        
        # Mock vector collection
        mock_collection = AsyncMock()
        mock_collection.get_by_document_id.return_value = VectorData(
            document_id="test_doc",
            chunk_id="existing_chunk",
            dense_vector=[0.0, 0.0, 0.0]
        )
        
        with patch("app.services.embedding_tasks.get_embedding_service", return_value=mock_service):
            with patch("app.services.embedding_tasks.get_vector_collection", return_value=mock_collection):
                result = await process_embeddings_async("test_doc", {"content": "test"})
                
                assert result["status"] == "success"
                # Verify update was called instead of add
                mock_collection.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_process_embeddings_async_partial_success(self):
        """Test batch processing with partial success (lines 299-300)."""
        documents = [
            {"id": "doc1", "content": "test1"},
            {"id": "doc2", "content": "test2"},
            {"id": "doc3", "content": "test3"}
        ]
        
        with patch("app.services.embedding_tasks.process_embeddings_async") as mock_process:
            # Make second doc fail
            async def side_effect(doc_id, metadata):
                if doc_id == "doc2":
                    raise Exception("Doc2 failed")
                return {"status": "success", "document_id": doc_id}
            
            mock_process.side_effect = side_effect
            
            results = await batch_process_embeddings_async(documents)
            
            assert results["total"] == 3
            assert results["successful"] == 2
            assert results["failed"] == 1
            assert len(results["errors"]) == 1

    def test_update_embeddings_task_celery_not_available(self):
        """Test update task when Celery not available (lines 341-346)."""
        with patch("app.services.embedding_tasks.is_celery_available", return_value=False):
            result = update_embeddings_task("doc_id", {"content": "updated"})
            
            assert result["status"] == "error"
            assert "Celery not available" in result["error"]

    def test_delete_embeddings_task_async_error(self):
        """Test delete task with async error (lines 369-371)."""
        async def failing_delete():
            raise Exception("Delete failed")
        
        with patch("app.services.embedding_tasks.delete_embeddings_async", side_effect=failing_delete):
            result = delete_embeddings_task("doc_to_delete")
            
            assert result["status"] == "error"
            assert "Delete failed" in result["error"]

    def test_reindex_embeddings_task_exception(self):
        """Test reindex with exception (lines 384-387)."""
        with patch("app.services.embedding_tasks.get_all_documents", side_effect=Exception("DB error")):
            result = reindex_embeddings_task(batch_size=10)
            
            assert result["status"] == "error"
            assert "DB error" in result["error"]

    @pytest.mark.asyncio
    async def test_check_embedding_health_unhealthy(self):
        """Test health check when unhealthy (line 414)."""
        mock_service = AsyncMock()
        mock_service.health_check.side_effect = Exception("Service down")
        
        with patch("app.services.embedding_tasks.get_embedding_service", return_value=mock_service):
            result = await check_embedding_health()
            
            assert result["healthy"] is False
            assert result["embedding_service"] is False

    def test_process_embeddings_with_priority_high(self):
        """Test high priority processing (lines 425-430)."""
        from app.services.embedding_tasks import TaskPriority
        
        with patch.object(process_embeddings_task, 'apply_async') as mock_apply:
            mock_apply.return_value = Mock(id="high_priority_task")
            
            result = process_embeddings_with_priority(
                "urgent_doc",
                {"content": "urgent"},
                priority=TaskPriority.HIGH
            )
            
            assert result["task_id"] == "high_priority_task"
            assert result["priority"] == TaskPriority.HIGH.value
            
            # Verify high priority queue was used
            call_args = mock_apply.call_args
            assert call_args[1]["priority"] == 9
            assert call_args[1]["queue"] == "high_priority"

    @pytest.mark.asyncio
    async def test_monitor_embedding_queue_with_issues(self):
        """Test queue monitoring with issues (lines 470-477)."""
        # Mock unhealthy stats
        mock_stats = {
            "available": True,
            "active_tasks": 100,  # High number of active tasks
            "scheduled_tasks": 50,
            "reserved_tasks": 20,
            "workers": 1  # Only one worker for many tasks
        }
        
        with patch("app.services.embedding_tasks.get_celery_stats", return_value=mock_stats):
            with patch("app.services.embedding_tasks.check_embedding_health") as mock_health:
                mock_health.return_value = {
                    "healthy": False,
                    "embedding_service": False
                }
                
                result = await monitor_embedding_queue()
                
                assert result["status"]["healthy"] is False
                assert result["status"]["issues"] == ["High task load", "Embedding service unhealthy"]
                assert result["recommendations"] == ["Scale workers", "Check embedding service"]

    def test_export_embeddings_task_completion(self):
        """Test export task completion (line 603)."""
        mock_embeddings = [
            {"document_id": "doc1", "embedding": [0.1, 0.2]},
            {"document_id": "doc2", "embedding": [0.3, 0.4]}
        ]
        
        with patch("app.services.embedding_tasks.get_all_embeddings", return_value=mock_embeddings):
            with patch("app.services.embedding_tasks.save_to_file") as mock_save:
                mock_save.return_value = "/tmp/export.json"
                
                result = export_embeddings_task("json", output_path="/tmp/export.json")
                
                assert result["status"] == "success"
                assert result["total_exported"] == 2
                assert result["output_path"] == "/tmp/export.json"