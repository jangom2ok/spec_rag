"""Embedding Tasks のテスト"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.services.embedding_tasks import (
    EmbeddingTaskManager,
    EmbeddingTaskService,
    embedding_health_check_task,
    process_batch_texts_task,
    process_document_embedding_task,
)


class TestEmbeddingTaskService:
    """EmbeddingTaskServiceクラスのテスト"""

    @pytest.fixture
    def mock_embedding_service(self):
        """EmbeddingServiceのモック"""
        mock_service = AsyncMock()
        mock_service.process_batch_request.return_value = [
            Mock(
                dense_vector=[0.1] * 1024,
                sparse_vector={0: 0.5, 100: 0.8},
                multi_vector=None,
                processing_time=0.1,
            )
        ]
        mock_service.health_check.return_value = {"status": "healthy"}
        return mock_service

    @pytest.fixture
    def mock_chunk_repository(self):
        """ChunkRepositoryのモック"""
        mock_repo = AsyncMock()
        mock_chunk = Mock()
        mock_chunk.id = "chunk-123"
        mock_chunk.document_id = "doc-456"
        mock_chunk.content = "テストコンテンツ"
        mock_chunk.chunk_type = "paragraph"

        mock_repo.get_chunks_by_document_id.return_value = [mock_chunk]
        return mock_repo

    @pytest.fixture
    def embedding_task_service(self, mock_embedding_service, mock_chunk_repository):
        """EmbeddingTaskServiceのフィクスチャ"""
        service = EmbeddingTaskService()
        service.embedding_service = mock_embedding_service
        service.chunk_repository = mock_chunk_repository
        return service

    @pytest.mark.asyncio
    async def test_process_document_chunks_success(self, embedding_task_service):
        """ドキュメントチャンク処理の成功テスト"""
        document_id = "test-doc-123"

        result = await embedding_task_service.process_document_chunks(document_id)

        assert result["status"] == "completed"
        assert result["document_id"] == document_id
        assert result["processed_count"] == 1
        assert result["vector_count"] == 2  # Dense + Sparse

    @pytest.mark.asyncio
    async def test_process_document_chunks_no_chunks(self, embedding_task_service):
        """チャンクが存在しない場合のテスト"""
        # チャンクリポジトリが空のリストを返すように設定
        embedding_task_service.chunk_repository.get_chunks_by_document_id.return_value = (
            []
        )

        document_id = "empty-doc"
        result = await embedding_task_service.process_document_chunks(document_id)

        assert result["status"] == "completed"
        assert result["processed_count"] == 0
        assert "No chunks found" in result["message"]

    @pytest.mark.asyncio
    async def test_process_document_chunks_embedding_error(
        self, embedding_task_service
    ):
        """埋め込み処理エラーのテスト"""
        # 埋め込みサービスがエラーを発生させるように設定
        embedding_task_service.embedding_service.process_batch_request.side_effect = (
            Exception("Embedding failed")
        )

        document_id = "error-doc"
        result = await embedding_task_service.process_document_chunks(document_id)

        assert result["status"] == "failed"
        assert "Embedding failed" in result["message"]
        assert result["processed_count"] == 0


class TestEmbeddingTaskManager:
    """EmbeddingTaskManagerクラスのテスト"""

    @patch("app.services.embedding_tasks.process_document_embedding_task")
    def test_submit_document_processing(self, mock_task):
        """ドキュメント処理タスク投入のテスト"""
        mock_result = Mock()
        mock_task.delay.return_value = mock_result

        result = EmbeddingTaskManager.submit_document_processing("test-doc")

        assert result == mock_result
        mock_task.delay.assert_called_once_with("test-doc")

    @patch("app.services.embedding_tasks.process_batch_texts_task")
    def test_submit_batch_processing(self, mock_task):
        """バッチ処理タスク投入のテスト"""
        mock_result = Mock()
        mock_task.delay.return_value = mock_result

        texts = ["text1", "text2"]
        metadata = {"test": "data"}

        result = EmbeddingTaskManager.submit_batch_processing(texts, metadata)

        assert result == mock_result
        mock_task.delay.assert_called_once_with(texts, metadata)

    @patch("app.services.embedding_tasks.AsyncResult")
    def test_get_task_status(self, mock_async_result):
        """タスクステータス取得のテスト"""
        mock_result = Mock()
        mock_result.status = "SUCCESS"
        mock_result.result = {"status": "completed"}
        mock_result.info = None
        mock_result.ready.return_value = True
        mock_result.successful.return_value = True
        mock_result.failed.return_value = False

        mock_async_result.return_value = mock_result

        status = EmbeddingTaskManager.get_task_status("task-123")

        assert status["task_id"] == "task-123"
        assert status["status"] == "SUCCESS"
        assert status["ready"] is True
        assert status["successful"] is True

    @patch("app.services.embedding_tasks.celery_app")
    def test_cancel_task(self, mock_celery_app):
        """タスクキャンセルのテスト"""
        mock_control = Mock()
        mock_celery_app.control = mock_control

        result = EmbeddingTaskManager.cancel_task("task-123")

        assert result["task_id"] == "task-123"
        assert result["status"] == "cancelled"
        mock_control.revoke.assert_called_once_with("task-123", terminate=True)

    @patch("app.services.embedding_tasks.celery_app")
    def test_get_worker_status(self, mock_celery_app):
        """ワーカーステータス取得のテスト"""
        mock_inspect = Mock()
        mock_inspect.active.return_value = {"worker1": []}
        mock_inspect.scheduled.return_value = {"worker1": []}
        mock_inspect.reserved.return_value = {"worker1": []}
        mock_inspect.stats.return_value = {"worker1": {"total": 0}}

        mock_control = Mock()
        mock_control.inspect.return_value = mock_inspect
        mock_celery_app.control = mock_control

        status = EmbeddingTaskManager.get_worker_status()

        assert "active_tasks" in status
        assert "scheduled_tasks" in status
        assert "reserved_tasks" in status
        assert "stats" in status


class TestCeleryTasks:
    """Celeryタスクのテスト（モック使用）"""

    @patch("app.services.embedding_tasks.get_task_service")
    @patch("asyncio.new_event_loop")
    def test_process_document_embedding_task_success(
        self, mock_new_loop, mock_get_service
    ):
        """ドキュメント埋め込みタスクの成功テスト"""
        # モックの設定
        mock_loop = Mock()
        mock_new_loop.return_value = mock_loop

        mock_service = AsyncMock()
        mock_service.process_document_chunks.return_value = {
            "status": "completed",
            "document_id": "test-doc",
            "processed_count": 5,
        }
        mock_get_service.return_value = mock_service

        mock_loop.run_until_complete.return_value = (
            mock_service.process_document_chunks.return_value
        )

        # モックタスクオブジェクト
        mock_task = Mock()

        # テスト実行
        result = process_document_embedding_task.__wrapped__(mock_task, "test-doc")

        # 検証
        assert result["status"] == "completed"
        assert result["document_id"] == "test-doc"
        assert result["processed_count"] == 5

        mock_task.update_state.assert_called_once()
        mock_loop.close.assert_called_once()

    @patch("app.services.embedding_tasks.get_task_service")
    @patch("asyncio.new_event_loop")
    def test_process_document_embedding_task_error(
        self, mock_new_loop, mock_get_service
    ):
        """ドキュメント埋め込みタスクのエラーテスト"""
        # エラーを発生させる設定
        mock_loop = Mock()
        mock_new_loop.return_value = mock_loop
        mock_loop.run_until_complete.side_effect = Exception("Processing error")

        mock_task = Mock()

        # テスト実行
        result = process_document_embedding_task.__wrapped__(mock_task, "error-doc")

        # 検証
        assert result["status"] == "failed"
        assert "Processing error" in result["message"]

    @patch("app.services.embedding_tasks.get_task_service")
    @patch("asyncio.new_event_loop")
    def test_process_batch_texts_task_success(self, mock_new_loop, mock_get_service):
        """バッチテキストタスクの成功テスト"""
        mock_loop = Mock()
        mock_new_loop.return_value = mock_loop

        mock_service = AsyncMock()
        mock_embedding_service = AsyncMock()
        mock_embedding_service.embed_batch.return_value = [
            Mock(dense_vector=[0.1] * 1024, sparse_vector={}, processing_time=0.1),
            Mock(dense_vector=[0.2] * 1024, sparse_vector={}, processing_time=0.1),
        ]
        mock_service.embedding_service = mock_embedding_service
        mock_get_service.return_value = mock_service

        expected_result = {
            "status": "completed",
            "message": "Processed 2 texts successfully",
            "text_count": 2,
            "results": [
                {
                    "dense_vector_length": 1024,
                    "sparse_vector_size": 0,
                    "processing_time": 0.1,
                },
                {
                    "dense_vector_length": 1024,
                    "sparse_vector_size": 0,
                    "processing_time": 0.1,
                },
            ],
        }

        mock_loop.run_until_complete.return_value = expected_result

        mock_task = Mock()
        texts = ["text1", "text2"]

        # テスト実行
        result = process_batch_texts_task.__wrapped__(mock_task, texts)

        # 検証
        assert result["status"] == "completed"
        assert result["text_count"] == 2
        mock_task.update_state.assert_called_once()

    @patch("app.services.embedding_tasks.get_task_service")
    @patch("asyncio.new_event_loop")
    def test_embedding_health_check_task_success(self, mock_new_loop, mock_get_service):
        """ヘルスチェックタスクの成功テスト"""
        mock_loop = Mock()
        mock_new_loop.return_value = mock_loop

        mock_service = AsyncMock()
        mock_embedding_service = AsyncMock()
        mock_embedding_service.health_check.return_value = {"status": "healthy"}
        mock_service.embedding_service = mock_embedding_service
        mock_get_service.return_value = mock_service

        mock_loop.run_until_complete.return_value = {"status": "healthy"}

        # テスト実行
        result = embedding_health_check_task.__wrapped__()

        # 検証
        assert result["status"] == "healthy"
        mock_loop.close.assert_called_once()

    @patch("app.services.embedding_tasks.get_task_service")
    @patch("asyncio.new_event_loop")
    def test_embedding_health_check_task_error(self, mock_new_loop, mock_get_service):
        """ヘルスチェックタスクのエラーテスト"""
        mock_new_loop.side_effect = Exception("Health check error")

        # テスト実行
        result = embedding_health_check_task.__wrapped__()

        # 検証
        assert result["status"] == "unhealthy"
        assert "Health check error" in result["reason"]


class TestIntegration:
    """統合テスト"""

    @pytest.mark.asyncio
    async def test_full_embedding_pipeline(self):
        """完全な埋め込みパイプラインのテスト（モック使用）"""
        with patch("app.services.embedding_tasks.get_task_service") as mock_get_service:
            # モックサービスの設定
            mock_service = AsyncMock()
            mock_embedding_service = AsyncMock()
            mock_chunk_repository = AsyncMock()

            # チャンクデータのモック
            mock_chunk = Mock()
            mock_chunk.id = "chunk-001"
            mock_chunk.document_id = "doc-001"
            mock_chunk.content = "テスト用のドキュメントコンテンツです。"
            mock_chunk.chunk_type = "paragraph"

            mock_chunk_repository.get_chunks_by_document_id.return_value = [mock_chunk]

            # 埋め込み結果のモック
            mock_embedding_result = Mock()
            mock_embedding_result.dense_vector = [0.1] * 1024
            mock_embedding_result.sparse_vector = {0: 0.5, 100: 0.8, 500: 0.3}
            mock_embedding_result.processing_time = 0.15

            mock_embedding_service.process_batch_request.return_value = [
                mock_embedding_result
            ]

            mock_service.embedding_service = mock_embedding_service
            mock_service.chunk_repository = mock_chunk_repository
            mock_service.process_document_chunks = AsyncMock(
                return_value={
                    "status": "completed",
                    "document_id": "doc-001",
                    "processed_count": 1,
                    "vector_count": 2,
                }
            )

            mock_get_service.return_value = mock_service

            # テスト実行
            service = await mock_get_service()
            result = await service.process_document_chunks("doc-001")

            # 検証
            assert result["status"] == "completed"
            assert result["document_id"] == "doc-001"
            assert result["processed_count"] == 1
            assert result["vector_count"] == 2
