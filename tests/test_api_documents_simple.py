"""
Simple coverage tests for app/api/documents.py focusing on missing lines.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.documents import (
    get_db,
    get_current_user_or_api_key,
    list_documents,
    create_document,
    delete_document,
    DocumentCreate,
    SourceType,
)


class TestDocumentsAPICoverage:
    """Test missing coverage in documents.py."""

    @pytest.mark.asyncio
    async def test_get_db_cleanup(self):
        """Test database session cleanup (lines 47-51)."""
        mock_session = AsyncMock(spec=AsyncSession)
        
        with patch("app.api.documents.SessionLocal") as mock_session_local:
            mock_session_local.return_value.__aenter__.return_value = mock_session
            mock_session_local.return_value.__aexit__.return_value = None
            
            # Test normal flow
            async for session in get_db():
                assert session == mock_session
            
            # Verify close was called
            mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_current_user_jwt_blacklisted(self):
        """Test JWT auth with blacklisted token (line 165)."""
        with patch("app.api.documents.is_token_blacklisted") as mock_blacklist:
            mock_blacklist.return_value = True
            
            with pytest.raises(HTTPException) as exc:
                await get_current_user_or_api_key(
                    authorization="Bearer blacklisted-token",
                    x_api_key=None
                )
            
            assert exc.value.status_code == 401
            assert exc.value.detail == "Token has been revoked"

    @pytest.mark.asyncio
    async def test_get_current_user_jwt_exception(self):
        """Test JWT auth exception handling (lines 176-179)."""
        with patch("app.api.documents.verify_token") as mock_verify:
            mock_verify.side_effect = Exception("Token error")
            
            # Should raise HTTPException, not the original exception
            with pytest.raises(HTTPException) as exc:
                await get_current_user_or_api_key(
                    authorization="Bearer bad-token",
                    x_api_key=None
                )
            
            assert exc.value.status_code == 401
            assert exc.value.detail == "Authentication required"

    @pytest.mark.asyncio
    async def test_delete_document_permissions(self):
        """Test delete document permission checks (lines 232-239)."""
        # Test with only delete permission
        user_with_delete = {"permissions": ["delete"]}
        result = await delete_document("doc1", current_user=user_with_delete)
        assert "deleted successfully" in result["message"]
        
        # Test with only admin permission
        user_with_admin = {"permissions": ["admin"]}
        result = await delete_document("doc2", current_user=user_with_admin)
        assert "deleted successfully" in result["message"]
        
        # Test without required permissions
        user_without_perms = {"permissions": ["read", "write"]}
        with pytest.raises(HTTPException) as exc:
            await delete_document("doc3", current_user=user_without_perms)
        assert exc.value.status_code == 403