"""
Final comprehensive test coverage for app/api/documents.py to achieve 100% coverage.
This file uses strategic patching to cover all missing lines.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import BackgroundTasks, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.documents import (
    DocumentUpdate,
    ProcessingConfigRequest,
    SourceType,
    _background_document_processing,
    _convert_to_processing_config,
    get_all_processing_status,
    get_current_user_or_api_key,
    get_db,
    get_document_processing_service,
    get_document_repository,
    get_processing_status,
    process_documents,
    process_documents_sync,
    reprocess_document,
    update_document,
)
from app.repositories.document_repository import DocumentRepository
from app.services.document_processing_service import DocumentProcessingService


class TestAuthentication:
    """Test authentication function covering all branches."""

    @pytest.mark.asyncio
    async def test_get_current_user_api_key_valid(self):
        """Test successful API key authentication."""
        with patch("app.api.documents.validate_api_key") as mock_validate:
            mock_validate.return_value = {
                "user_id": "api_user",
                "permissions": ["read", "write"],
            }

            result = await get_current_user_or_api_key(
                authorization=None, x_api_key="valid-api-key"
            )

            assert result["user_id"] == "api_user"
            assert result["auth_type"] == "api_key"
            assert "read" in result["permissions"]

    @pytest.mark.asyncio
    async def test_get_current_user_api_key_invalid(self):
        """Test invalid API key authentication."""
        with patch("app.api.documents.validate_api_key") as mock_validate:
            mock_validate.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                await get_current_user_or_api_key(
                    authorization=None, x_api_key="invalid-api-key"
                )

            assert exc_info.value.status_code == 401
            assert exc_info.value.detail == "Authentication required"

    @pytest.mark.asyncio
    async def test_get_current_user_jwt_blacklisted(self):
        """Test JWT with blacklisted token (line 165)."""
        with patch("app.core.auth.is_token_blacklisted") as mock_blacklist:
            with patch("app.core.auth.verify_token"):
                with patch("app.core.auth.users_storage"):
                    mock_blacklist.return_value = True

                    with pytest.raises(HTTPException) as exc_info:
                        await get_current_user_or_api_key(
                            authorization="Bearer blacklisted-token", x_api_key=None
                        )

                    # Due to exception handling, it returns generic error
                    assert exc_info.value.status_code == 401
                    assert exc_info.value.detail == "Authentication required"

    @pytest.mark.asyncio
    async def test_get_current_user_jwt_no_user(self):
        """Test JWT when user not found (line 176)."""
        with patch("app.core.auth.is_token_blacklisted") as mock_blacklist:
            with patch("app.core.auth.verify_token") as mock_verify:
                with patch("app.core.auth.users_storage") as mock_users:
                    mock_blacklist.return_value = False
                    mock_verify.return_value = {"sub": "unknown@example.com"}
                    mock_users.get.return_value = None

                    with pytest.raises(HTTPException) as exc_info:
                        await get_current_user_or_api_key(
                            authorization="Bearer valid-token", x_api_key=None
                        )

                    assert exc_info.value.status_code == 401
                    assert exc_info.value.detail == "Authentication required"

    @pytest.mark.asyncio
    async def test_get_current_user_jwt_exception(self):
        """Test JWT exception handling (lines 176-179)."""
        with patch("app.core.auth.verify_token") as mock_verify:
            mock_verify.side_effect = Exception("Token error")

            # Patch logging.debug to verify it's called
            with patch("app.api.documents.logging.debug") as mock_debug:
                with pytest.raises(HTTPException) as exc_info:
                    await get_current_user_or_api_key(
                        authorization="Bearer bad-token", x_api_key=None
                    )

                assert exc_info.value.status_code == 401
                assert exc_info.value.detail == "Authentication required"
                # Verify debug logging was called
                mock_debug.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_current_user_jwt_success(self):
        """Test successful JWT authentication."""
        with patch("app.core.auth.is_token_blacklisted") as mock_blacklist:
            with patch("app.core.auth.verify_token") as mock_verify:
                with patch("app.core.auth.users_storage") as mock_users:
                    mock_blacklist.return_value = False
                    mock_verify.return_value = {"sub": "test@example.com"}
                    mock_users.get.return_value = {
                        "user_id": "test_user",
                        "permissions": ["read", "write"],
                    }

                    result = await get_current_user_or_api_key(
                        authorization="Bearer valid-token", x_api_key=None
                    )

                    assert result["user_id"] == "test_user"
                    assert result["email"] == "test@example.com"
                    assert result["auth_type"] == "jwt"


class TestDocumentUpdate:
    """Test document update endpoint covering all branches."""

    @pytest.mark.asyncio
    async def test_update_document_no_permission(self):
        """Test update without write permission (lines 268-269)."""
        doc_update = DocumentUpdate(title="Updated Title")
        user = {"permissions": ["read"]}

        with pytest.raises(HTTPException) as exc_info:
            await update_document(
                document_id="test-id", document_update=doc_update, current_user=user
            )

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "Write permission required"

    @pytest.mark.asyncio
    async def test_update_document_not_found(self):
        """Test update non-existent document (lines 272-273)."""
        doc_update = DocumentUpdate(title="Updated Title")
        user = {"permissions": ["write"]}

        with pytest.raises(HTTPException) as exc_info:
            await update_document(
                document_id="non-existent",
                document_update=doc_update,
                current_user=user,
            )

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Document not found"

    @pytest.mark.asyncio
    async def test_update_document_all_fields(self):
        """Test updating all fields (lines 276, 284-294)."""
        user = {"permissions": ["write"]}

        # Update title only
        doc_update = DocumentUpdate(title="New Title")
        result = await update_document(
            document_id="test-id", document_update=doc_update, current_user=user
        )
        assert result.title == "New Title"
        assert result.content == "Test content"

        # Update content only
        doc_update = DocumentUpdate(content="New content")
        result = await update_document(
            document_id="test-id", document_update=doc_update, current_user=user
        )
        assert result.title == "Test Document"
        assert result.content == "New content"

        # Update source_type only
        doc_update = DocumentUpdate(source_type=SourceType.confluence)
        result = await update_document(
            document_id="test-id", document_update=doc_update, current_user=user
        )
        assert result.source_type == SourceType.confluence

        # Update all fields
        doc_update = DocumentUpdate(
            title="All New Title",
            content="All new content",
            source_type=SourceType.jira,
        )
        result = await update_document(
            document_id="test-id", document_update=doc_update, current_user=user
        )
        assert result.title == "All New Title"
        assert result.content == "All new content"
        assert result.source_type == SourceType.jira


class TestDocumentProcessingService:
    """Test document processing service dependency."""

    @pytest.mark.asyncio
    async def test_get_document_processing_service(self):
        """Test service creation (lines 303-307)."""
        mock_session = AsyncMock(spec=AsyncSession)

        service = await get_document_processing_service(db=mock_session)

        assert isinstance(service, DocumentProcessingService)
        assert service.document_repository is not None
        assert service.chunk_repository is not None


class TestProcessDocuments:
    """Test async document processing."""

    @pytest.mark.asyncio
    async def test_process_documents_no_permission(self):
        """Test without write permission (lines 321-322)."""
        config = ProcessingConfigRequest(
            source_type=SourceType.confluence, source_path="/path"
        )
        background_tasks = BackgroundTasks()
        user = {"permissions": ["read"]}
        mock_service = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await process_documents(
                config=config,
                background_tasks=background_tasks,
                current_user=user,
                processing_service=mock_service,
            )

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "Write permission required"

    @pytest.mark.asyncio
    async def test_process_documents_success(self):
        """Test successful processing (lines 324-344)."""
        config = ProcessingConfigRequest(
            source_type=SourceType.confluence, source_path="/path"
        )
        background_tasks = BackgroundTasks()
        user = {"permissions": ["write"]}
        mock_service = AsyncMock()

        result = await process_documents(
            config=config,
            background_tasks=background_tasks,
            current_user=user,
            processing_service=mock_service,
        )

        assert result.success is True
        assert result.total_documents == 0
        assert result.processing_time == 0.0

    @pytest.mark.asyncio
    async def test_process_documents_exception(self):
        """Test with exception (lines 346-350)."""
        config = ProcessingConfigRequest(
            source_type=SourceType.confluence, source_path="/path"
        )
        background_tasks = BackgroundTasks()
        user = {"permissions": ["write"]}

        with patch("app.api.documents._convert_to_processing_config") as mock_convert:
            mock_convert.side_effect = Exception("Conversion error")
            mock_service = AsyncMock()

            with pytest.raises(HTTPException) as exc_info:
                await process_documents(
                    config=config,
                    background_tasks=background_tasks,
                    current_user=user,
                    processing_service=mock_service,
                )

            assert exc_info.value.status_code == 500
            assert "Processing failed: Conversion error" in exc_info.value.detail


class TestProcessDocumentsSync:
    """Test sync document processing."""

    @pytest.mark.asyncio
    async def test_process_sync_no_permission(self):
        """Test without permission (lines 363-364)."""
        config = ProcessingConfigRequest(
            source_type=SourceType.confluence, source_path="/path"
        )
        user = {"permissions": ["read"]}
        mock_service = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await process_documents_sync(
                config=config, current_user=user, processing_service=mock_service
            )

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "Write permission required"

    @pytest.mark.asyncio
    async def test_process_sync_success(self):
        """Test successful sync (lines 366-383)."""
        config = ProcessingConfigRequest(
            source_type=SourceType.confluence, source_path="/path"
        )
        user = {"permissions": ["write"]}

        mock_result = Mock()
        mock_result.success = True
        mock_result.total_documents = 5
        mock_result.successful_documents = 4
        mock_result.failed_documents = 1
        mock_result.total_chunks = 20
        mock_result.successful_chunks = 19
        mock_result.failed_chunks = 1
        mock_result.processing_time = 10.5
        mock_result.errors = ["error1"]

        mock_service = AsyncMock()
        mock_service.process_documents.return_value = mock_result

        result = await process_documents_sync(
            config=config, current_user=user, processing_service=mock_service
        )

        assert result.success is True
        assert result.total_documents == 5
        assert result.error_count == 1

    @pytest.mark.asyncio
    async def test_process_sync_exception(self):
        """Test sync exception (lines 385-389)."""
        config = ProcessingConfigRequest(
            source_type=SourceType.confluence, source_path="/path"
        )
        user = {"permissions": ["write"]}

        mock_service = AsyncMock()
        mock_service.process_documents.side_effect = Exception("Process error")

        with pytest.raises(HTTPException) as exc_info:
            await process_documents_sync(
                config=config, current_user=user, processing_service=mock_service
            )

        assert exc_info.value.status_code == 500
        assert "Processing failed: Process error" in exc_info.value.detail


class TestProcessingStatus:
    """Test processing status endpoints."""

    @pytest.mark.asyncio
    async def test_get_status_success(self):
        """Test successful status retrieval (lines 401-414)."""
        user = {"permissions": ["read"]}
        mock_service = Mock()
        mock_service.get_processing_status.return_value = {
            "document_id": "doc123",
            "stage": "chunking",
            "progress": 0.75,
            "error_message": None,
            "chunks_processed": 15,
            "chunks_total": 20,
        }

        result = await get_processing_status(
            document_id="doc123", current_user=user, processing_service=mock_service
        )

        assert result.document_id == "doc123"
        assert result.stage == "chunking"
        assert result.progress == 0.75

    @pytest.mark.asyncio
    async def test_get_status_not_found(self):
        """Test status not found (lines 404-405)."""
        user = {"permissions": ["read"]}
        mock_service = Mock()
        mock_service.get_processing_status.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await get_processing_status(
                document_id="unknown",
                current_user=user,
                processing_service=mock_service,
            )

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Processing status not found"

    @pytest.mark.asyncio
    async def test_get_status_exception(self):
        """Test status exception (lines 416-422)."""
        user = {"permissions": ["read"]}
        mock_service = Mock()
        mock_service.get_processing_status.side_effect = Exception("Status error")

        with pytest.raises(HTTPException) as exc_info:
            await get_processing_status(
                document_id="doc123", current_user=user, processing_service=mock_service
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to get processing status"


class TestAllProcessingStatus:
    """Test getting all processing status."""

    @pytest.mark.asyncio
    async def test_get_all_status_success(self):
        """Test successful all status (lines 433-447)."""
        user = {"permissions": ["read"]}
        mock_service = Mock()
        mock_service.get_all_processing_status.return_value = {
            "doc1": {
                "document_id": "doc1",
                "stage": "completed",
                "progress": 1.0,
                "error_message": None,
                "chunks_processed": 10,
                "chunks_total": 10,
            },
            "doc2": {
                "document_id": "doc2",
                "stage": "chunking",
                "progress": 0.5,
                "error_message": "Timeout",
                "chunks_processed": 5,
                "chunks_total": 10,
            },
        }

        result = await get_all_processing_status(
            current_user=user, processing_service=mock_service
        )

        assert "doc1" in result
        assert result["doc1"].stage == "completed"
        assert result["doc2"].error_message == "Timeout"

    @pytest.mark.asyncio
    async def test_get_all_status_exception(self):
        """Test all status exception (lines 449-453)."""
        user = {"permissions": ["read"]}
        mock_service = Mock()
        mock_service.get_all_processing_status.side_effect = Exception(
            "All status error"
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_all_processing_status(
                current_user=user, processing_service=mock_service
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to get processing status"


class TestReprocessDocument:
    """Test document reprocessing."""

    @pytest.mark.asyncio
    async def test_reprocess_no_permission(self):
        """Test without permission (lines 469-470)."""
        config = ProcessingConfigRequest(
            source_type=SourceType.confluence, source_path="/path"
        )
        user = {"permissions": ["read"]}
        mock_service = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await reprocess_document(
                document_id="doc123",
                config=config,
                current_user=user,
                processing_service=mock_service,
            )

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "Write permission required"

    @pytest.mark.asyncio
    async def test_reprocess_success(self):
        """Test successful reprocess (lines 472-492)."""
        config = ProcessingConfigRequest(
            source_type=SourceType.confluence, source_path="/path"
        )
        user = {"permissions": ["write"]}

        mock_service = AsyncMock()
        mock_service.process_single_document_by_id.return_value = {
            "success": True,
            "total_chunks": 10,
            "successful_chunks": 10,
            "failed_chunks": 0,
        }

        result = await reprocess_document(
            document_id="doc123",
            config=config,
            current_user=user,
            processing_service=mock_service,
        )

        assert result.success is True
        assert result.total_documents == 1
        assert result.successful_documents == 1
        assert result.total_chunks == 10

    @pytest.mark.asyncio
    async def test_reprocess_failure(self):
        """Test failed reprocess (lines 494-504)."""
        config = ProcessingConfigRequest(
            source_type=SourceType.confluence, source_path="/path"
        )
        user = {"permissions": ["write"]}

        mock_service = AsyncMock()
        mock_service.process_single_document_by_id.return_value = {"success": False}

        result = await reprocess_document(
            document_id="doc123",
            config=config,
            current_user=user,
            processing_service=mock_service,
        )

        assert result.success is False
        assert result.failed_documents == 1
        assert result.error_count == 1

    @pytest.mark.asyncio
    async def test_reprocess_exception(self):
        """Test reprocess exception (lines 506-510)."""
        config = ProcessingConfigRequest(
            source_type=SourceType.confluence, source_path="/path"
        )
        user = {"permissions": ["write"]}

        mock_service = AsyncMock()
        mock_service.process_single_document_by_id.side_effect = Exception(
            "Reprocess error"
        )

        with pytest.raises(HTTPException) as exc_info:
            await reprocess_document(
                document_id="doc123",
                config=config,
                current_user=user,
                processing_service=mock_service,
            )

        assert exc_info.value.status_code == 500
        assert "Reprocessing failed: Reprocess error" in exc_info.value.detail


class TestConversionHelpers:
    """Test conversion helper functions."""

    def test_convert_config_all_strategies(self):
        """Test all chunking strategies (lines 516-551)."""
        # Test semantic strategy
        request = ProcessingConfigRequest(
            source_type=SourceType.confluence,
            source_path="/path",
            chunking_strategy="semantic",
            chunk_size=500,
            overlap_size=100,
            extract_structure=True,
            extract_entities=True,
            extract_keywords=True,
        )

        config = _convert_to_processing_config(request)

        assert config.collection_config.source_type.value == "confluence"
        assert config.extraction_config.extract_structure is True
        assert config.chunking_config.strategy.value == "semantic"
        assert config.chunking_config.chunk_size == 500

        # Test hierarchical strategy
        request.chunking_strategy = "hierarchical"
        config = _convert_to_processing_config(request)
        assert config.chunking_config.strategy.value == "hierarchical"

        # Test default strategy
        request.chunking_strategy = "unknown"
        config = _convert_to_processing_config(request)
        assert config.chunking_config.strategy.value == "fixed_size"


class TestBackgroundProcessing:
    """Test background processing."""

    @pytest.mark.asyncio
    async def test_background_success(self):
        """Test successful background (lines 558-560)."""
        mock_service = AsyncMock()
        mock_service.process_documents.return_value = Mock(
            get_summary=Mock(return_value="Processing completed")
        )

        mock_config = Mock()

        # Should complete without exception
        await _background_document_processing(mock_service, mock_config)

        mock_service.process_documents.assert_called_once_with(mock_config)

    @pytest.mark.asyncio
    async def test_background_exception(self):
        """Test background exception (lines 561-562)."""
        mock_service = AsyncMock()
        mock_service.process_documents.side_effect = Exception("Background error")

        mock_config = Mock()

        # Should complete without raising exception (errors are logged)
        await _background_document_processing(mock_service, mock_config)

        mock_service.process_documents.assert_called_once_with(mock_config)


class TestRepositories:
    """Test repository dependencies."""

    @pytest.mark.asyncio
    async def test_get_document_repository(self):
        """Test document repository (line 569)."""
        mock_session = AsyncMock(spec=AsyncSession)

        repo = await get_document_repository(db=mock_session)

        assert isinstance(repo, DocumentRepository)


class TestDatabaseSession:
    """Test database session management."""

    @pytest.mark.asyncio
    async def test_get_db_cleanup(self):
        """Test database cleanup (lines 47-51)."""
        mock_session = AsyncMock()
        mock_session.close = AsyncMock()

        with patch("app.api.documents.SessionLocal") as mock_local:
            async_cm = AsyncMock()
            async_cm.__aenter__ = AsyncMock(return_value=mock_session)
            async_cm.__aexit__ = AsyncMock(return_value=None)
            mock_local.return_value = async_cm

            # Normal flow
            async for session in get_db():
                assert session == mock_session

            # Verify close called
            mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_db_exception_cleanup(self):
        """Test cleanup on exception."""
        mock_session = AsyncMock()
        mock_session.close = AsyncMock()

        with patch("app.api.documents.SessionLocal") as mock_local:
            async_cm = AsyncMock()
            async_cm.__aenter__ = AsyncMock(return_value=mock_session)
            async_cm.__aexit__ = AsyncMock(return_value=None)
            mock_local.return_value = async_cm

            # Exception flow
            try:
                async for _ in get_db():
                    raise Exception("Test exception")
            except Exception:  # noqa: S110
                pass  # Expected exception for test

            # Verify close still called
            mock_session.close.assert_called_once()
