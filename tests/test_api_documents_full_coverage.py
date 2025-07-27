"""
Comprehensive test coverage for app/api/documents.py to achieve 100% coverage.
This file focuses on covering all missing lines identified in the coverage report.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import BackgroundTasks, HTTPException
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.documents import (
    DocumentUpdate,
    ProcessingConfigRequest,
    SourceType,
    _background_document_processing,
    _convert_to_processing_config,
    get_current_user_or_api_key,
    get_db,
    get_document_processing_service,
    get_document_repository,
    get_processing_status,
    process_documents_sync,
    reprocess_document,
    update_document,
)
from app.repositories.document_repository import DocumentRepository
from app.services.document_processing_service import DocumentProcessingService


@pytest.fixture
def mock_session():
    """Mock database session."""
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.fixture
def mock_document_repo():
    """Mock document repository."""
    repo = Mock(spec=DocumentRepository)
    repo.get_by_id = AsyncMock(return_value=None)
    repo.update = AsyncMock()
    repo.get_all = AsyncMock(return_value=[])
    repo.create = AsyncMock()
    repo.delete = AsyncMock()
    return repo


@pytest.fixture
def mock_processing_service():
    """Mock document processing service."""
    service = Mock(spec=DocumentProcessingService)
    service.process_document = AsyncMock()
    service.process_documents = AsyncMock()
    service.reprocess_document = AsyncMock()
    service.get_processing_status = AsyncMock(return_value=None)
    service.get_all_processing_status = AsyncMock(return_value=[])
    return service


@pytest.fixture
def mock_auth_user():
    """Mock authenticated user."""
    return {"sub": "test@example.com", "permissions": ["read", "write"]}


class TestDocumentEndpointsCoverage:
    """Test coverage for document endpoints."""

    @pytest.mark.asyncio
    async def test_update_document_not_found_line_107(
        self, mock_session, mock_document_repo
    ):
        """Test document not found (line 107)."""
        # Mock dependencies
        with patch("app.api.documents.get_db", return_value=mock_session):
            with patch(
                "app.api.documents.get_document_repository",
                return_value=mock_document_repo,
            ):
                with patch(
                    "app.api.documents.get_current_user_or_api_key",
                    return_value={"permissions": ["write"]},
                ):
                    # Repository returns None (document not found)
                    mock_document_repo.get_by_id.return_value = None

                    # Should raise 404
                    with pytest.raises(HTTPException) as exc_info:
                        await update_document(
                            document_id="non-existent-id",
                            document_update=DocumentUpdate(title="New Title"),
                            current_user={"permissions": ["write"]},
                        )

                    assert exc_info.value.status_code == 404
                    assert exc_info.value.detail == "Document not found"

    @pytest.mark.asyncio
    async def test_process_documents_invalid_source_type_line_287(self):
        """Test invalid source type (line 287)."""
        # Pydantic validates the source_type at creation time
        with pytest.raises(ValidationError) as exc_info:
            ProcessingConfigRequest(
                source_type="invalid_type",  # This should trigger the validation
                parameters={"path": "/test"},
            )

        # Check that the validation error is about the source_type
        assert "source_type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_background_document_processing_exception_line_243(
        self, mock_session, mock_processing_service
    ):
        """Test exception in background processing (line 243)."""
        # Make process_document raise an exception
        mock_processing_service.process_document.side_effect = Exception(
            "Processing failed"
        )

        # Mock background tasks
        BackgroundTasks()

        # Create a mock ProcessingConfig
        mock_config = Mock()

        # Run background processing (it should handle the exception)
        await _background_document_processing(
            processing_service=mock_processing_service,
            config=mock_config,
        )

        # The exception should be handled, not propagated

    @pytest.mark.asyncio
    async def test_process_documents_sync_validation_error_line_318(self):
        """Test validation error in sync processing (line 318)."""
        with patch(
            "app.api.documents.get_document_processing_service"
        ) as mock_service_getter:
            mock_service = Mock()
            # Simulate a validation error during processing
            mock_service.process_documents.side_effect = ValueError(
                "Invalid document format"
            )
            mock_service_getter.return_value = mock_service

            with patch("app.api.documents.get_current_user_or_api_key"):
                with pytest.raises(HTTPException) as exc_info:
                    await process_documents_sync(
                        config=ProcessingConfigRequest(
                            source_type=SourceType.test, parameters={"path": "/test"}
                        ),
                        current_user={"permissions": ["write"]},
                        processing_service=mock_service,
                    )

                assert exc_info.value.status_code == 500
                assert "Invalid document format" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_processing_status_none_line_370(self):
        """Test processing status returns None (line 370)."""
        with patch(
            "app.api.documents.get_document_processing_service"
        ) as mock_service_getter:
            mock_service = Mock()
            mock_service.get_processing_status.return_value = None
            mock_service_getter.return_value = mock_service

            with patch("app.api.documents.get_current_user_or_api_key"):
                # When service returns None, endpoint should handle it
                result = await get_processing_status(
                    document_id="unknown-task",
                    current_user={"permissions": ["read"]},
                    processing_service=mock_service,
                )

                # Should return None or handle appropriately
                assert result is None or result == {"status": "not_found"}

    @pytest.mark.asyncio
    async def test_reprocess_document_exception_line_398(
        self, mock_session, mock_processing_service
    ):
        """Test exception during reprocessing (line 398)."""
        with patch("app.api.documents.get_db", return_value=mock_session):
            with patch(
                "app.api.documents.get_document_processing_service",
                return_value=mock_processing_service,
            ):
                # Make reprocess_document raise a general exception
                mock_processing_service.reprocess_document.side_effect = Exception(
                    "Reprocessing failed"
                )

                with patch("app.api.documents.get_current_user_or_api_key"):
                    with pytest.raises(HTTPException) as exc_info:
                        await reprocess_document(
                            document_id="test-id",
                            db=mock_session,
                            current_user={"permissions": ["write"]},
                        )

                    assert exc_info.value.status_code == 500
                    assert "Error reprocessing document" in exc_info.value.detail


class TestAuthenticationCoverage:
    """Test coverage for authentication edge cases."""

    @pytest.mark.asyncio
    async def test_get_current_user_no_auth_header_line_147(self):
        """Test no authorization header (line 147)."""
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_or_api_key(authorization=None, x_api_key=None)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Authentication required"

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_scheme_line_156(self):
        """Test invalid authorization scheme (line 156)."""
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_or_api_key(
                authorization="Basic dGVzdDp0ZXN0",  # Not Bearer
                x_api_key=None,
            )

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid authentication scheme"

    @pytest.mark.asyncio
    async def test_get_current_user_jwt_blacklisted_line_165(self):
        """Test blacklisted JWT token (line 165)."""
        with patch("app.core.auth.is_token_blacklisted", return_value=True):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user_or_api_key(
                    authorization="Bearer blacklisted-token", x_api_key=None
                )

            assert exc_info.value.status_code == 401
            assert exc_info.value.detail == "Authentication required"

    @pytest.mark.asyncio
    async def test_get_current_user_jwt_no_user_line_173(self):
        """Test JWT token with no corresponding user (line 173)."""
        with patch("app.core.auth.is_token_blacklisted", return_value=False):
            with patch(
                "app.core.auth.verify_token",
                return_value={"sub": "nonexistent@test.com"},
            ):
                with patch("app.core.auth.users_storage.get", return_value=None):
                    with pytest.raises(HTTPException) as exc_info:
                        await get_current_user_or_api_key(
                            authorization="Bearer valid-token", x_api_key=None
                        )

                    assert exc_info.value.status_code == 401
                    assert exc_info.value.detail == "User not found"

    @pytest.mark.asyncio
    async def test_get_current_user_jwt_exception_line_179(self):
        """Test JWT processing exception (line 179)."""
        with patch("app.core.auth.verify_token", side_effect=Exception("JWT error")):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user_or_api_key(
                    authorization="Bearer invalid-token", x_api_key=None
                )

            assert exc_info.value.status_code == 401
            assert exc_info.value.detail == "Authentication required"

    @pytest.mark.asyncio
    async def test_get_current_user_api_key_validation_error_line_189(self):
        """Test API key validation error (line 189)."""
        with patch(
            "app.core.auth.validate_api_key",
            side_effect=ValueError("Invalid API key format"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user_or_api_key(
                    authorization=None, x_api_key="invalid-format-key"
                )

            assert exc_info.value.status_code == 401
            assert exc_info.value.detail == "Invalid API key"

    @pytest.mark.asyncio
    async def test_get_current_user_api_key_exception_line_193(self):
        """Test API key processing exception (line 193)."""
        with patch(
            "app.core.auth.validate_api_key", side_effect=Exception("API key error")
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user_or_api_key(
                    authorization=None, x_api_key="error-key"
                )

            assert exc_info.value.status_code == 401
            assert exc_info.value.detail == "Authentication required"


class TestDependencyCoverage:
    """Test coverage for dependency injection."""

    @pytest.mark.asyncio
    async def test_get_db_exception_handling(self):
        """Test database session exception handling."""
        # Mock the sessionmaker
        with patch("app.api.documents.SessionLocal") as mock_local:
            mock_session = AsyncMock()
            mock_session.close = AsyncMock()

            # Create async context manager mock
            async_cm = AsyncMock()
            async_cm.__aenter__ = AsyncMock(return_value=mock_session)
            async_cm.__aexit__ = AsyncMock(return_value=None)
            mock_local.return_value = async_cm

            # Test normal flow
            async for session in get_db():
                assert session == mock_session

            # Verify close was called
            mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_document_repository(self, mock_session):
        """Test document repository creation."""
        repo = get_document_repository(mock_session)
        assert isinstance(repo, DocumentRepository)

    @pytest.mark.asyncio
    async def test_get_document_processing_service(self):
        """Test processing service creation."""
        service = await get_document_processing_service()
        assert isinstance(service, DocumentProcessingService)


class TestProcessingConfigConversion:
    """Test coverage for processing config conversion."""

    def test_convert_to_processing_config_all_source_types(self):
        """Test conversion for all source types."""
        # Test FILE source type
        file_request = ProcessingConfigRequest(
            source_type=SourceType.test,
            parameters={"path": "/test/file.pdf"},
            chunk_size=1000,
            chunk_overlap=200,
        )
        file_config = _convert_to_processing_config(file_request)
        assert file_config.collection_config.source_type.value == SourceType.test.value
        assert file_config.chunking_config.chunk_size == 1000
        assert file_config.chunking_config.overlap_size == 200

        # Test confluence source type
        confluence_request = ProcessingConfigRequest(
            source_type=SourceType.confluence,
            parameters={"url": "https://confluence.example.com"},
            chunk_size=2000,
        )
        confluence_config = _convert_to_processing_config(confluence_request)
        assert (
            confluence_config.collection_config.source_type.value
            == SourceType.confluence.value
        )
        assert confluence_config.chunking_config.chunk_size == 2000

        # Test jira source type
        jira_request = ProcessingConfigRequest(
            source_type=SourceType.jira,
            parameters={"url": "https://jira.example.com", "api_key": "secret"},
        )
        jira_config = _convert_to_processing_config(jira_request)
        assert jira_config.collection_config.source_type.value == SourceType.jira.value

    def test_convert_to_processing_config_optional_params(self):
        """Test conversion with optional parameters."""
        # Test with minimal parameters
        request = ProcessingConfigRequest(
            source_type=SourceType.test, parameters={"path": "/test"}
        )
        config = _convert_to_processing_config(request)

        # Check defaults are applied
        assert hasattr(config.chunking_config, "chunk_size")
        assert hasattr(config.chunking_config, "overlap_size")


# No cleanup needed since we're not modifying sys.modules anymore
