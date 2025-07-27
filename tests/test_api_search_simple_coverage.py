"""
Simple focused tests for app/api/search.py to cover missing lines.
This minimal test file targets specific missing coverage.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException

# We'll test the specific functions that need coverage


class TestSearchAPIMissingCoverage:
    """Test specific missing lines in search.py."""

    def test_convert_enhanced_filters_with_all_fields(self):
        """Test convert_enhanced_filters_to_legacy with all filter types."""
        from app.api.search import EnhancedFilters, convert_enhanced_filters_to_legacy

        # Test with source_types (line 273)
        filters = EnhancedFilters(
            source_types=["manual", "api"], languages=[], date_range=None, tags=[]
        )
        result = convert_enhanced_filters_to_legacy(filters)
        assert len(result) == 1
        assert result[0].field == "source_type"

        # Test with languages (line 280)
        filters = EnhancedFilters(
            source_types=[], languages=["en", "ja"], date_range=None, tags=[]
        )
        result = convert_enhanced_filters_to_legacy(filters)
        assert len(result) == 1
        assert result[0].field == "language"

        # Test with tags (line 287)
        filters = EnhancedFilters(
            source_types=[], languages=[], date_range=None, tags=["test", "example"]
        )
        result = convert_enhanced_filters_to_legacy(filters)
        assert len(result) == 1
        assert result[0].field == "metadata.tags"

        # Test with date_range (line 296)
        from app.api.search import DateRange

        filters = EnhancedFilters(
            source_types=[],
            languages=[],
            date_range=DateRange(**{"from": "2024-01-01", "to": "2024-12-31"}),
            tags=[],
        )
        result = convert_enhanced_filters_to_legacy(filters)
        assert len(result) == 2  # start and end filters

    @pytest.mark.asyncio
    async def test_search_documents_missing_coverage(self):
        """Test search_documents to cover lines 393-595."""
        from app.api.search import (
            SearchOptions,
            SearchRequest,
            search_documents,
        )
        from app.services.hybrid_search_engine import SearchResult

        # Mock dependencies
        mock_user = {"permissions": ["read"]}
        mock_engine = Mock()
        mock_engine.config = Mock()

        # Create a failed search result to cover lines 574-595
        failed_result = Mock(spec=SearchResult)
        failed_result.success = False
        failed_result.error_message = "Search failed"
        failed_result.total_hits = 0
        failed_result.documents = []
        failed_result.facets = None

        mock_engine.search = AsyncMock(return_value=failed_result)

        request = SearchRequest(
            query="test",
            filters=None,
            search_options=SearchOptions(
                search_type="hybrid",
                max_results=10,
                min_score=0.0,
                highlight=True,
                include_metadata=True,
            ),
            ranking_options=None,
            max_results=None,
            offset=0,
            search_mode=None,
            dense_weight=None,
            sparse_weight=None,
            similarity_threshold=None,
            enable_reranking=None,
        )

        response = await search_documents(
            request=request, current_user=mock_user, search_engine=mock_engine
        )

        assert response.success is False
        assert response.error_message == "Search failed"
        assert response.total_results == 0

    @pytest.mark.asyncio
    async def test_search_endpoints_coverage(self):
        """Test semantic and keyword search endpoints (lines 609-721, 737-849)."""
        from app.api.search import SearchRequest, keyword_search, semantic_search

        mock_user = {"permissions": ["read"]}
        mock_engine = Mock()
        mock_engine.config = Mock()

        # Test semantic search exception (lines 609-721)
        mock_engine.search = AsyncMock(side_effect=Exception("Semantic error"))

        request = SearchRequest(
            query="test",
            filters=None,
            search_options=None,
            ranking_options=None,
            max_results=None,
            offset=0,
            search_mode=None,
            dense_weight=None,
            sparse_weight=None,
            similarity_threshold=None,
            enable_reranking=None,
        )

        with pytest.raises(HTTPException) as exc:
            await semantic_search(
                request=request, current_user=mock_user, search_engine=mock_engine
            )
        assert exc.value.status_code == 500
        assert "Semantic search failed" in str(exc.value.detail)

        # Test keyword search exception (lines 737-849)
        with pytest.raises(HTTPException) as exc:
            await keyword_search(
                request=request, current_user=mock_user, search_engine=mock_engine
            )
        assert exc.value.status_code == 500
        assert "Keyword search failed" in str(exc.value.detail)

    @pytest.mark.asyncio
    async def test_search_suggestions_coverage(self):
        """Test search suggestions endpoint (lines 864-899)."""
        from app.api.search import get_search_suggestions

        # Test with no read permission (lines 866-867)
        mock_user = {"permissions": ["write"]}  # No read permission

        with pytest.raises(HTTPException) as exc:
            await get_search_suggestions(q="test", limit=5, current_user=mock_user)
        assert exc.value.status_code == 403

        # Test exception handling (lines 897-901)
        mock_user = {"permissions": ["read"]}

        # We can't easily trigger a real exception in this simple function,
        # but we can test the normal flow
        result = await get_search_suggestions(q="data", limit=3, current_user=mock_user)

        assert hasattr(result, "suggestions")
        assert hasattr(result, "query")
        assert result.query == "data"

    @pytest.mark.asyncio
    async def test_search_config_coverage(self):
        """Test search config endpoint (lines 912-953)."""
        from app.api.search import get_search_config

        # Test no permission (line 915)
        mock_user = {"permissions": ["write"]}  # No read permission

        with pytest.raises(HTTPException) as exc:
            await get_search_config(current_user=mock_user)
        assert exc.value.status_code == 403

        # Test normal success case since get_search_config doesn't take search_engine parameter
        mock_user = {"permissions": ["read"]}

        # We can't easily trigger the exception in lines 951-953 without modifying the function,
        # so we'll test the success path
        result = await get_search_config(current_user=mock_user)

        assert "config" in result
        assert isinstance(result["config"], dict)

    @pytest.mark.asyncio
    async def test_search_aliases_coverage(self):
        """Test search alias endpoints (lines 969-972, 986-989)."""
        from app.api.search import SearchRequest, search_keyword, search_semantic

        mock_user = {"permissions": ["read"]}
        mock_engine = Mock()

        # Mock the actual search functions
        with patch("app.api.search.semantic_search") as mock_semantic:
            mock_semantic.return_value = {"success": True}

            request = SearchRequest(
                query="test",
                filters=None,
                search_options=None,
                ranking_options=None,
                max_results=None,
                offset=0,
                search_mode=None,
                dense_weight=None,
                sparse_weight=None,
                similarity_threshold=None,
                enable_reranking=None,
            )
            await search_semantic(
                request=request, current_user=mock_user, search_engine=mock_engine
            )

            mock_semantic.assert_called_once()

        with patch("app.api.search.keyword_search") as mock_keyword:
            mock_keyword.return_value = {"success": True}

            await search_keyword(
                request=request, current_user=mock_user, search_engine=mock_engine
            )

            mock_keyword.assert_called_once()
