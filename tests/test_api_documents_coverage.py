"""
Comprehensive test coverage for app/api/documents.py to achieve 100% coverage.
This file focuses on covering all missing lines identified in the coverage report.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi import HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
import os

from app.api.documents import (
    router,
    get_db,
    get_current_user_or_api_key,
    list_documents,
    create_document,
    delete_document,
    update_document,
    get_document,
    process_documents,
    process_documents_sync,
    get_processing_status,
    get_all_processing_status,
    reprocess_document,
    get_document_processing_service,
    get_document_repository,
    _convert_to_processing_config,
    _background_document_processing,
    DocumentCreate,
    DocumentUpdate,
    DocumentResponse,
    DocumentList,
    SourceType,
    ProcessingConfigRequest,
    ProcessingStatusResponse,
    ProcessingResultResponse,
)
from app.repositories.document_repository import DocumentRepository
from app.repositories.chunk_repository import DocumentChunkRepository
from app.services.document_processing_service import DocumentProcessingService


@pytest.fixture
def mock_session():
    """Mock database session."""
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.fixture
def mock_current_user():
    """Mock current user with permissions."""
    return {
        "user_id": "test_user",
        "email": "test@example.com",
        "permissions": ["read", "write", "delete", "admin"],
        "auth_type": "jwt",
    }


@pytest.fixture
def mock_current_user_api_key():
    """Mock current user authenticated via API key."""
    return {
        "user_id": "api_user",
        "permissions": ["read", "write"],
        "auth_type": "api_key",
    }


@pytest.fixture
def mock_current_user_limited():
    """Mock current user with limited permissions."""
    return {
        "user_id": "limited_user",
        "email": "limited@example.com",
        "permissions": ["read"],
        "auth_type": "jwt",
    }


class TestAuthentication:
    """Test authentication function."""

    @pytest.mark.asyncio
    async def test_get_current_user_jwt_blacklisted(self):
        """Test JWT authentication with blacklisted token (line 165)."""
        # We need to patch the entire module since imports are done inside the function
        with patch("app.api.documents.is_token_blacklisted", create=True) as mock_blacklist:
            with patch("app.api.documents.verify_token", create=True) as mock_verify:
                with patch("app.api.documents.users_storage", create=True) as mock_users:
                    mock_blacklist.return_value = True
                    
                    with pytest.raises(HTTPException) as exc_info:
                        await get_current_user_or_api_key(
                            authorization="Bearer blacklisted-token", x_api_key=None
                        )

                    # Due to the exception handling in the function, it returns generic error
                    assert exc_info.value.status_code == 401
                    assert exc_info.value.detail == "Authentication required"

    @pytest.mark.asyncio
    async def test_get_current_user_jwt_no_user_in_storage(self):
        """Test JWT authentication when user not found in storage (lines 176)."""
        with patch("app.core.auth.verify_token") as mock_verify:
            mock_verify.return_value = {"sub": "unknown@example.com"}
            
            with patch("app.core.auth.is_token_blacklisted") as mock_blacklist:
                mock_blacklist.return_value = False
                
                with patch("app.core.auth.users_storage") as mock_users:
                    mock_users.get.return_value = None

                    with pytest.raises(HTTPException) as exc_info:
                        await get_current_user_or_api_key(
                            authorization="Bearer valid-token", x_api_key=None
                        )

                    assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_get_current_user_jwt_exception(self):
        """Test JWT authentication with exception (lines 176-179)."""
        with patch("app.core.auth.verify_token") as mock_verify:
            mock_verify.side_effect = Exception("Token verification failed")

            with pytest.raises(HTTPException) as exc_info:
                await get_current_user_or_api_key(
                    authorization="Bearer invalid-token", x_api_key=None
                )

            assert exc_info.value.status_code == 401
            assert exc_info.value.detail == "Authentication required"


class TestDocumentUpdate:
    """Test document update endpoint."""

    @pytest.mark.asyncio
    async def test_update_document_no_permission(self, mock_current_user_limited):
        """Test updating document without write permission (lines 268-269)."""
        doc_update = DocumentUpdate(title="Updated Title")

        with pytest.raises(HTTPException) as exc_info:
            await update_document(
                document_id="test-id",
                document_update=doc_update,
                current_user=mock_current_user_limited
            )

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "Write permission required"

    @pytest.mark.asyncio
    async def test_update_document_not_found(self, mock_current_user):
        """Test updating non-existent document (lines 272-273)."""
        doc_update = DocumentUpdate(title="Updated Title")

        with pytest.raises(HTTPException) as exc_info:
            await update_document(
                document_id="non-existent-id",
                document_update=doc_update,
                current_user=mock_current_user
            )

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Document not found"

    @pytest.mark.asyncio
    async def test_update_document_partial_update(self, mock_current_user):
        """Test partial document update (lines 276, 284-294)."""
        # Test updating only title
        doc_update = DocumentUpdate(title="New Title")
        result = await update_document(
            document_id="test-id",
            document_update=doc_update,
            current_user=mock_current_user
        )
        assert result.title == "New Title"
        assert result.content == "Test content"  # Original content preserved

        # Test updating only content
        doc_update = DocumentUpdate(content="New content")
        result = await update_document(
            document_id="test-id",
            document_update=doc_update,
            current_user=mock_current_user
        )
        assert result.title == "Test Document"  # Original title preserved
        assert result.content == "New content"

        # Test updating source_type
        doc_update = DocumentUpdate(source_type=SourceType.confluence)
        result = await update_document(
            document_id="test-id",
            document_update=doc_update,
            current_user=mock_current_user
        )
        assert result.source_type == SourceType.confluence


class TestDocumentProcessingService:
    """Test document processing service dependency."""

    @pytest.mark.asyncio
    async def test_get_document_processing_service(self, mock_session):
        """Test getting document processing service (lines 303-307)."""
        async def mock_get_db():
            yield mock_session

        service = await get_document_processing_service(db=mock_session)
        
        assert isinstance(service, DocumentProcessingService)
        assert service.document_repository is not None
        assert service.chunk_repository is not None


class TestProcessDocuments:
    """Test document processing endpoints."""

    @pytest.mark.asyncio
    async def test_process_documents_no_permission(self, mock_current_user_limited):
        """Test processing documents without write permission (lines 321-322)."""
        config = ProcessingConfigRequest(
            source_type=SourceType.confluence,
            source_path="/path/to/docs"
        )
        background_tasks = BackgroundTasks()
        mock_service = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await process_documents(
                config=config,
                background_tasks=background_tasks,
                current_user=mock_current_user_limited,
                processing_service=mock_service
            )

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "Write permission required"

    @pytest.mark.asyncio
    async def test_process_documents_success(self, mock_current_user):
        """Test successful document processing (lines 324-344)."""
        config = ProcessingConfigRequest(
            source_type=SourceType.confluence,
            source_path="/path/to/docs"
        )
        background_tasks = BackgroundTasks()
        mock_service = AsyncMock()

        result = await process_documents(
            config=config,
            background_tasks=background_tasks,
            current_user=mock_current_user,
            processing_service=mock_service
        )

        assert result.success is True
        assert result.total_documents == 0
        assert result.processing_time == 0.0

    @pytest.mark.asyncio
    async def test_process_documents_exception(self, mock_current_user):
        """Test document processing with exception (lines 346-350)."""
        config = ProcessingConfigRequest(
            source_type=SourceType.confluence,
            source_path="/path/to/docs"
        )
        background_tasks = BackgroundTasks()
        
        with patch("app.api.documents._convert_to_processing_config") as mock_convert:
            mock_convert.side_effect = Exception("Conversion error")
            mock_service = AsyncMock()

            with pytest.raises(HTTPException) as exc_info:
                await process_documents(
                    config=config,
                    background_tasks=background_tasks,
                    current_user=mock_current_user,
                    processing_service=mock_service
                )

            assert exc_info.value.status_code == 500
            assert "Processing failed" in exc_info.value.detail


class TestProcessDocumentsSync:
    """Test synchronous document processing."""

    @pytest.mark.asyncio
    async def test_process_documents_sync_no_permission(self, mock_current_user_limited):
        """Test sync processing without permission (lines 363-364)."""
        config = ProcessingConfigRequest(
            source_type=SourceType.confluence,
            source_path="/path/to/docs"
        )
        mock_service = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await process_documents_sync(
                config=config,
                current_user=mock_current_user_limited,
                processing_service=mock_service
            )

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "Write permission required"

    @pytest.mark.asyncio
    async def test_process_documents_sync_success(self, mock_current_user):
        """Test successful sync processing (lines 366-383)."""
        config = ProcessingConfigRequest(
            source_type=SourceType.confluence,
            source_path="/path/to/docs"
        )
        
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
            config=config,
            current_user=mock_current_user,
            processing_service=mock_service
        )

        assert result.success is True
        assert result.total_documents == 5
        assert result.error_count == 1

    @pytest.mark.asyncio
    async def test_process_documents_sync_exception(self, mock_current_user):
        """Test sync processing with exception (lines 385-389)."""
        config = ProcessingConfigRequest(
            source_type=SourceType.confluence,
            source_path="/path/to/docs"
        )
        
        mock_service = AsyncMock()
        mock_service.process_documents.side_effect = Exception("Processing error")

        with pytest.raises(HTTPException) as exc_info:
            await process_documents_sync(
                config=config,
                current_user=mock_current_user,
                processing_service=mock_service
            )

        assert exc_info.value.status_code == 500
        assert "Processing failed" in exc_info.value.detail


class TestProcessingStatus:
    """Test processing status endpoints."""

    @pytest.mark.asyncio
    async def test_get_processing_status_success(self, mock_current_user):
        """Test getting processing status (lines 401-414)."""
        mock_service = Mock()
        mock_service.get_processing_status.return_value = {
            "document_id": "doc123",
            "stage": "chunking",
            "progress": 0.75,
            "error_message": None,
            "chunks_processed": 15,
            "chunks_total": 20
        }

        result = await get_processing_status(
            document_id="doc123",
            current_user=mock_current_user,
            processing_service=mock_service
        )

        assert result.document_id == "doc123"
        assert result.stage == "chunking"
        assert result.progress == 0.75

    @pytest.mark.asyncio
    async def test_get_processing_status_not_found(self, mock_current_user):
        """Test getting status when not found (lines 404-405)."""
        mock_service = Mock()
        mock_service.get_processing_status.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await get_processing_status(
                document_id="unknown",
                current_user=mock_current_user,
                processing_service=mock_service
            )

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Processing status not found"

    @pytest.mark.asyncio
    async def test_get_processing_status_exception(self, mock_current_user):
        """Test getting status with exception (lines 416-422)."""
        mock_service = Mock()
        mock_service.get_processing_status.side_effect = Exception("Status error")

        with pytest.raises(HTTPException) as exc_info:
            await get_processing_status(
                document_id="doc123",
                current_user=mock_current_user,
                processing_service=mock_service
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to get processing status"


class TestAllProcessingStatus:
    """Test getting all processing status."""

    @pytest.mark.asyncio
    async def test_get_all_processing_status_success(self, mock_current_user):
        """Test getting all processing status (lines 433-447)."""
        mock_service = Mock()
        mock_service.get_all_processing_status.return_value = {
            "doc1": {
                "document_id": "doc1",
                "stage": "completed",
                "progress": 1.0,
                "error_message": None,
                "chunks_processed": 10,
                "chunks_total": 10
            },
            "doc2": {
                "document_id": "doc2",
                "stage": "chunking",
                "progress": 0.5,
                "error_message": "Timeout",
                "chunks_processed": 5,
                "chunks_total": 10
            }
        }

        result = await get_all_processing_status(
            current_user=mock_current_user,
            processing_service=mock_service
        )

        assert "doc1" in result
        assert result["doc1"].stage == "completed"
        assert result["doc2"].error_message == "Timeout"

    @pytest.mark.asyncio
    async def test_get_all_processing_status_exception(self, mock_current_user):
        """Test getting all status with exception (lines 449-453)."""
        mock_service = Mock()
        mock_service.get_all_processing_status.side_effect = Exception("Status error")

        with pytest.raises(HTTPException) as exc_info:
            await get_all_processing_status(
                current_user=mock_current_user,
                processing_service=mock_service
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to get processing status"


class TestReprocessDocument:
    """Test document reprocessing."""

    @pytest.mark.asyncio
    async def test_reprocess_document_no_permission(self, mock_current_user_limited):
        """Test reprocessing without permission (lines 469-470)."""
        config = ProcessingConfigRequest(
            source_type=SourceType.confluence,
            source_path="/path/to/docs"
        )
        mock_service = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await reprocess_document(
                document_id="doc123",
                config=config,
                current_user=mock_current_user_limited,
                processing_service=mock_service
            )

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "Write permission required"

    @pytest.mark.asyncio
    async def test_reprocess_document_success(self, mock_current_user):
        """Test successful reprocessing (lines 472-492)."""
        config = ProcessingConfigRequest(
            source_type=SourceType.confluence,
            source_path="/path/to/docs"
        )
        
        mock_service = AsyncMock()
        mock_service.process_single_document_by_id.return_value = {
            "success": True,
            "total_chunks": 10,
            "successful_chunks": 10,
            "failed_chunks": 0
        }

        result = await reprocess_document(
            document_id="doc123",
            config=config,
            current_user=mock_current_user,
            processing_service=mock_service
        )

        assert result.success is True
        assert result.total_documents == 1
        assert result.successful_documents == 1
        assert result.total_chunks == 10

    @pytest.mark.asyncio
    async def test_reprocess_document_failure(self, mock_current_user):
        """Test failed reprocessing (lines 494-504)."""
        config = ProcessingConfigRequest(
            source_type=SourceType.confluence,
            source_path="/path/to/docs"
        )
        
        mock_service = AsyncMock()
        mock_service.process_single_document_by_id.return_value = {
            "success": False
        }

        result = await reprocess_document(
            document_id="doc123",
            config=config,
            current_user=mock_current_user,
            processing_service=mock_service
        )

        assert result.success is False
        assert result.failed_documents == 1
        assert result.error_count == 1

    @pytest.mark.asyncio
    async def test_reprocess_document_exception(self, mock_current_user):
        """Test reprocessing with exception (lines 506-510)."""
        config = ProcessingConfigRequest(
            source_type=SourceType.confluence,
            source_path="/path/to/docs"
        )
        
        mock_service = AsyncMock()
        mock_service.process_single_document_by_id.side_effect = Exception("Reprocess error")

        with pytest.raises(HTTPException) as exc_info:
            await reprocess_document(
                document_id="doc123",
                config=config,
                current_user=mock_current_user,
                processing_service=mock_service
            )

        assert exc_info.value.status_code == 500
        assert "Reprocessing failed" in exc_info.value.detail


class TestConversionHelpers:
    """Test conversion helper functions."""

    def test_convert_to_processing_config_all_strategies(self):
        """Test config conversion with all chunking strategies (lines 516-551)."""
        # Test with semantic strategy
        request = ProcessingConfigRequest(
            source_type=SourceType.confluence,
            source_path="/path/to/docs",
            chunking_strategy="semantic",
            chunk_size=500,
            overlap_size=100,
            extract_structure=True,
            extract_entities=True,
            extract_keywords=True
        )
        
        config = _convert_to_processing_config(request)
        
        assert config.collection_config.source_type.value == "confluence"
        assert config.collection_config.batch_size == 10
        assert config.extraction_config.extract_structure is True
        assert config.chunking_config.strategy.value == "semantic"
        assert config.chunking_config.chunk_size == 500

        # Test with hierarchical strategy
        request.chunking_strategy = "hierarchical"
        config = _convert_to_processing_config(request)
        assert config.chunking_config.strategy.value == "hierarchical"

        # Test with default (fixed_size) strategy
        request.chunking_strategy = "fixed_size"
        config = _convert_to_processing_config(request)
        assert config.chunking_config.strategy.value == "fixed_size"


class TestBackgroundProcessing:
    """Test background processing function."""

    @pytest.mark.asyncio
    async def test_background_processing_success(self):
        """Test successful background processing (lines 558-560)."""
        mock_service = AsyncMock()
        mock_service.process_documents.return_value = Mock(
            get_summary=Mock(return_value="Processing completed: 10 documents")
        )
        
        mock_config = Mock()
        
        # Should not raise any exception
        await _background_document_processing(mock_service, mock_config)
        
        mock_service.process_documents.assert_called_once_with(mock_config)

    @pytest.mark.asyncio
    async def test_background_processing_exception(self):
        """Test background processing with exception (lines 561-562)."""
        mock_service = AsyncMock()
        mock_service.process_documents.side_effect = Exception("Background error")
        
        mock_config = Mock()
        
        # Should not raise exception (errors are logged)
        await _background_document_processing(mock_service, mock_config)
        
        mock_service.process_documents.assert_called_once_with(mock_config)


class TestDocumentRepository:
    """Test document repository dependency."""

    @pytest.mark.asyncio
    async def test_get_document_repository(self, mock_session):
        """Test getting document repository (line 569)."""
        repo = await get_document_repository(db=mock_session)
        assert isinstance(repo, DocumentRepository)