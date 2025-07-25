"""
Test coverage for mock endpoints in app/api/documents.py.
These endpoints return mock data for testing purposes.
"""

import pytest
from app.api.documents import (
    list_documents,
    create_document,
    delete_document,
    get_document,
    DocumentCreate,
    SourceType,
    DocumentList,
    DocumentResponse,
)
from fastapi import HTTPException


class TestMockEndpoints:
    """Test mock endpoint implementations."""

    @pytest.mark.asyncio
    async def test_list_documents_mock(self):
        """Test list documents returns mock data (lines 190-204)."""
        user = {"permissions": ["read"]}
        
        result = await list_documents(current_user=user)
        
        assert isinstance(result, DocumentList)
        assert len(result.documents) == 2
        
        # Check first document
        assert result.documents[0].id == "doc1"
        assert result.documents[0].title == "Sample Document 1"
        assert result.documents[0].content == "Sample content 1"
        assert result.documents[0].source_type == SourceType.test
        
        # Check second document  
        assert result.documents[1].id == "doc2"
        assert result.documents[1].title == "Sample Document 2"
        assert result.documents[1].content == "Sample content 2"
        assert result.documents[1].source_type == SourceType.confluence

    @pytest.mark.asyncio
    async def test_create_document_mock(self):
        """Test create document returns mock response (lines 214-218)."""
        user = {"permissions": ["write"]}
        
        doc_create = DocumentCreate(
            title="Test Title",
            content="Test Content",
            source_type=SourceType.jira
        )
        
        result = await create_document(document=doc_create, current_user=user)
        
        assert isinstance(result, DocumentResponse)
        assert result.id == "mock-id"
        assert result.title == "Test Title"
        assert result.content == "Test Content"
        assert result.source_type == SourceType.jira
        
    @pytest.mark.asyncio
    async def test_create_document_no_write_permission(self):
        """Test create document without write permission."""
        user = {"permissions": ["read"]}
        
        doc_create = DocumentCreate(
            title="Test Title",
            content="Test Content",
            source_type=SourceType.test
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await create_document(document=doc_create, current_user=user)
        
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "Write permission required"

    @pytest.mark.asyncio
    async def test_delete_document_mock(self):
        """Test delete document returns mock response (lines 233-239)."""
        # Test with delete permission
        user = {"permissions": ["delete"]}
        
        result = await delete_document(
            document_id="test-doc-123",
            current_user=user
        )
        
        assert isinstance(result, dict)
        assert result["message"] == "Document test-doc-123 deleted successfully"
        
        # Test with admin permission
        admin_user = {"permissions": ["admin"]}
        
        result = await delete_document(
            document_id="admin-doc-456",
            current_user=admin_user
        )
        
        assert result["message"] == "Document admin-doc-456 deleted successfully"
        
    @pytest.mark.asyncio
    async def test_delete_document_no_permission(self):
        """Test delete document without proper permissions."""
        user = {"permissions": ["read", "write"]}
        
        with pytest.raises(HTTPException) as exc_info:
            await delete_document(
                document_id="no-perm-doc",
                current_user=user
            )
        
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "Delete permission required"

    @pytest.mark.asyncio
    async def test_get_document_mock(self):
        """Test get document returns mock response (lines 250-257)."""
        user = {"permissions": ["read"]}
        
        # Test with existing test document
        result = await get_document(
            document_id="test-id",
            current_user=user
        )
        
        assert isinstance(result, DocumentResponse)
        assert result.id == "test-id"
        assert result.title == "Test Document"
        assert result.content == "Test content"
        assert result.source_type == SourceType.test
        
    @pytest.mark.asyncio
    async def test_get_document_not_found(self):
        """Test get document with non-existent ID."""
        user = {"permissions": ["read"]}
        
        with pytest.raises(HTTPException) as exc_info:
            await get_document(
                document_id="non-existent-doc",
                current_user=user
            )
        
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Document not found"