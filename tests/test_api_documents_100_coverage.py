"""
Final test file to achieve 100% coverage for app/api/documents.py.
Covers the remaining missing lines.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.documents import (
    get_db,
    get_current_user_or_api_key,
    update_document,
    process_documents,
    get_processing_status,
    get_all_processing_status,
    get_processing_status,
    _background_document_processing,
    DocumentUpdate,
    SourceType,
    ProcessingConfigRequest,
)


class TestDatabaseSessionFinally:
    """Test database session finally block."""

    @pytest.mark.asyncio
    async def test_get_db_finally_block(self):
        """Test finally block execution (line 51)."""
        mock_session = AsyncMock()
        mock_session.close = AsyncMock()
        
        # Create a context manager that properly handles the finally block
        class MockContextManager:
            async def __aenter__(self):
                return mock_session
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                # This simulates the automatic cleanup by async with
                return False
        
        with patch("app.api.documents.SessionLocal") as mock_local:
            mock_local.return_value = MockContextManager()
            
            # Use the generator and let it complete normally
            gen = get_db()
            session = await gen.__anext__()
            assert session == mock_session
            
            # Complete the generator (this triggers the finally block)
            try:
                await gen.__anext__()
            except StopAsyncIteration:
                pass
            
            # The close should have been called in the finally block
            mock_session.close.assert_called_once()


class TestAuthenticationEdgeCases:
    """Test authentication edge cases."""

    @pytest.mark.asyncio
    async def test_jwt_blacklisted_token_inside_try(self):
        """Test blacklisted token raises inside try block (line 165)."""
        # This test verifies the specific HTTPException is raised
        # but caught by the outer exception handler
        with patch("app.core.auth.is_token_blacklisted", create=True) as mock_blacklist:
            with patch("app.core.auth.verify_token", create=True) as mock_verify:
                with patch("app.core.auth.users_storage", create=True) as mock_users:
                    # Set up the mock to make is_token_blacklisted accessible
                    mock_blacklist.return_value = True
                    
                    # We need to ensure the import succeeds
                    with patch.dict("sys.modules", {"app.core.auth": Mock(
                        is_token_blacklisted=mock_blacklist,
                        verify_token=mock_verify,
                        users_storage=mock_users
                    )}):
                        with pytest.raises(HTTPException) as exc_info:
                            await get_current_user_or_api_key(
                                authorization="Bearer blacklisted-token",
                                x_api_key=None
                            )
                        
                        # The exception is caught and re-raised as generic auth error
                        assert exc_info.value.status_code == 401
                        assert exc_info.value.detail == "Authentication required"

    @pytest.mark.asyncio
    async def test_jwt_exception_with_logging(self):
        """Test JWT exception with debug logging (lines 176, 178-179)."""
        # Mock the logging.debug call
        with patch("app.api.documents.logging.debug") as mock_debug:
            with patch("app.core.auth.verify_token", create=True) as mock_verify:
                mock_verify.side_effect = Exception("Verify failed")
                
                with patch.dict("sys.modules", {"app.core.auth": Mock(
                    verify_token=mock_verify
                )}):
                    with pytest.raises(HTTPException) as exc_info:
                        await get_current_user_or_api_key(
                            authorization="Bearer bad-token",
                            x_api_key=None
                        )
                    
                    # Verify logging was called
                    mock_debug.assert_called()
                    assert "JWT認証に失敗" in mock_debug.call_args[0][0]
                    
                    assert exc_info.value.status_code == 401


class TestDocumentUpdateEdgeCases:
    """Test document update edge cases."""

    @pytest.mark.asyncio  
    async def test_update_document_existing_doc_dict(self):
        """Test updating document with existing_doc dict creation (line 276)."""
        user = {"permissions": ["write"]}
        
        # Test when all fields are None (no updates)
        doc_update = DocumentUpdate()  # All fields None by default
        
        result = await update_document(
            document_id="test-id",
            document_update=doc_update,
            current_user=user
        )
        
        # Should return the original document unchanged
        assert result.id == "test-id"
        assert result.title == "Test Document"
        assert result.content == "Test content"
        assert result.source_type == SourceType.test

    @pytest.mark.asyncio
    async def test_update_document_content_hash_comment(self):
        """Test content update triggers hash recalculation comment (line 290)."""
        user = {"permissions": ["write"]}
        
        # Update content to trigger the comment block
        doc_update = DocumentUpdate(content="Updated content for hash")
        
        result = await update_document(
            document_id="test-id",
            document_update=doc_update,
            current_user=user
        )
        
        assert result.content == "Updated content for hash"
        # The hash calculation is just a comment/pass, so no actual effect


class TestProcessingEdgeCases:
    """Test processing edge cases."""

    @pytest.mark.asyncio
    async def test_process_documents_background_task_added(self):
        """Test background task is added (lines 329-330)."""
        config = ProcessingConfigRequest(
            source_type=SourceType.confluence,
            source_path="/test/path"
        )
        user = {"permissions": ["write"]}
        mock_service = AsyncMock()
        
        # Create a real BackgroundTasks instance to track calls
        background_tasks = BackgroundTasks()
        
        # Spy on add_task
        with patch.object(background_tasks, 'add_task') as mock_add_task:
            result = await process_documents(
                config=config,
                background_tasks=background_tasks,
                current_user=user,
                processing_service=mock_service
            )
            
            # Verify add_task was called with correct arguments
            mock_add_task.assert_called_once()
            args = mock_add_task.call_args[0]
            assert args[0] == _background_document_processing
            assert args[1] == mock_service
            # args[2] is the converted config
            
            assert result.success is True


class TestProcessingStatusEdgeCases:
    """Test processing status edge cases."""

    @pytest.mark.asyncio
    async def test_get_processing_status_http_exception_reraised(self):
        """Test HTTPException is re-raised (lines 416-417)."""
        user = {"permissions": ["read"]}
        mock_service = Mock()
        
        # Make get_processing_status raise HTTPException
        mock_service.get_processing_status.side_effect = HTTPException(
            status_code=404, detail="Not found"
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await get_processing_status(
                document_id="doc123",
                current_user=user,
                processing_service=mock_service
            )
        
        # Should re-raise the same exception
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Not found"

    @pytest.mark.asyncio
    async def test_get_processing_status_generic_exception_wrapped(self):
        """Test generic exception is wrapped (lines 418-422)."""
        user = {"permissions": ["read"]}
        mock_service = Mock()
        
        # Make get_processing_status raise generic exception
        mock_service.get_processing_status.side_effect = RuntimeError("Internal error")
        
        with patch("app.api.documents.logger.error") as mock_logger:
            with pytest.raises(HTTPException) as exc_info:
                await get_processing_status(
                    document_id="doc123",
                    current_user=user,
                    processing_service=mock_service
                )
            
            # Verify logging
            mock_logger.assert_called_once()
            
            # Should wrap in HTTPException
            assert exc_info.value.status_code == 500
            assert exc_info.value.detail == "Failed to get processing status"


class TestAllProcessingStatusEdgeCases:
    """Test all processing status edge cases."""

    @pytest.mark.asyncio
    async def test_get_all_status_iteration_and_creation(self):
        """Test status iteration and object creation (lines 434-435)."""
        user = {"permissions": ["read"]}
        mock_service = Mock()
        
        # Return empty dict to test the iteration
        mock_service.get_all_processing_status.return_value = {}
        
        result = await get_all_processing_status(
            current_user=user,
            processing_service=mock_service
        )
        
        assert result == {}
        
        # Test with actual data to cover the loop
        mock_service.get_all_processing_status.return_value = {
            "doc1": {
                "document_id": "doc1",
                "stage": "processing",
                "progress": 0.5
            }
        }
        
        result = await get_all_processing_status(
            current_user=user,
            processing_service=mock_service
        )
        
        assert "doc1" in result
        assert result["doc1"].document_id == "doc1"

    @pytest.mark.asyncio
    async def test_get_all_status_exception_wrapped(self):
        """Test exception wrapping (lines 451-453)."""
        user = {"permissions": ["read"]}
        mock_service = Mock()
        
        mock_service.get_all_processing_status.side_effect = Exception("Status error")
        
        with patch("app.api.documents.logger.error") as mock_logger:
            with pytest.raises(HTTPException) as exc_info:
                await get_all_processing_status(
                    current_user=user,
                    processing_service=mock_service
                )
            
            # Verify logging
            mock_logger.assert_called_once()
            assert "Failed to get all processing status" in str(mock_logger.call_args)
            
            assert exc_info.value.status_code == 500
            assert exc_info.value.detail == "Failed to get processing status"


class TestBackgroundProcessingEdgeCases:
    """Test background processing edge cases."""

    @pytest.mark.asyncio
    async def test_background_processing_success_with_logging(self):
        """Test successful background processing with logging (lines 559-560)."""
        mock_service = AsyncMock()
        mock_result = Mock()
        mock_result.get_summary.return_value = "Summary: 10 docs processed"
        mock_service.process_documents.return_value = mock_result
        
        mock_config = Mock()
        
        with patch("app.api.documents.logger.info") as mock_logger:
            await _background_document_processing(mock_service, mock_config)
            
            # Verify logging
            mock_logger.assert_called_once()
            assert "Background processing completed" in str(mock_logger.call_args)
            assert "Summary: 10 docs processed" in str(mock_logger.call_args)

    @pytest.mark.asyncio
    async def test_background_processing_exception_with_logging(self):
        """Test background processing exception with logging (lines 561-562)."""
        mock_service = AsyncMock()
        mock_service.process_documents.side_effect = Exception("Process failed")
        
        mock_config = Mock()
        
        with patch("app.api.documents.logger.error") as mock_logger:
            # Should not raise exception
            await _background_document_processing(mock_service, mock_config)
            
            # Verify error logging
            mock_logger.assert_called_once()
            assert "Background processing failed" in str(mock_logger.call_args)
            assert "Process failed" in str(mock_logger.call_args)