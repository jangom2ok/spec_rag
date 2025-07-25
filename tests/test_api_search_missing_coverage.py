"""Test module for achieving 100% coverage of app/api/search.py.

This module contains targeted tests for all uncovered lines in the search API,
including authentication, search endpoints, and error handling scenarios.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.search import (
    DateRange,
    EnhancedFilters,
    convert_enhanced_filters_to_legacy,
    generate_search_suggestions,
    get_current_user_or_api_key,
    get_hybrid_search_engine,
    highlight_content,
)
from app.main import app
from app.services.hybrid_search_engine import (
    HybridSearchEngine,
    SearchConfig,
    SearchMode,
    SearchResult,
)


class TestAuthenticationDependency:
    """Test cases for get_current_user_or_api_key dependency."""

    @pytest.mark.asyncio
    async def test_api_key_authentication_success(self):
        """Test successful API key authentication."""
        with patch("app.api.search.validate_api_key") as mock_validate:
            mock_validate.return_value = {
                "user_id": "test_user",
                "permissions": ["read", "write"],
            }

            result = await get_current_user_or_api_key(
                authorization=None, x_api_key="test_api_key"
            )

            assert result["user_id"] == "test_user"
            assert result["auth_type"] == "api_key"
            assert "read" in result["permissions"]

    @pytest.mark.asyncio
    async def test_jwt_authentication_success(self):
        """Test successful JWT authentication."""
        with (
            patch("app.api.search.verify_token") as mock_verify,
            patch("app.api.search.is_token_blacklisted") as mock_blacklist,
            patch("app.api.search.users_storage") as mock_storage,
        ):

            mock_blacklist.return_value = False
            mock_verify.return_value = {"sub": "test@example.com"}
            mock_storage.get.return_value = {
                "user_id": "jwt_user",
                "permissions": ["read"],
            }

            result = await get_current_user_or_api_key(
                authorization="Bearer test_token", x_api_key=None
            )

            assert result["email"] == "test@example.com"
            assert result["auth_type"] == "jwt"
            assert result["user_id"] == "jwt_user"

    @pytest.mark.asyncio
    async def test_jwt_authentication_blacklisted_token(self):
        """Test JWT authentication with blacklisted token."""
        with (
            patch("app.api.search.is_token_blacklisted") as mock_blacklist,
            patch("app.api.search.verify_token") as mock_verify,
        ):
            mock_blacklist.return_value = True
            mock_verify.return_value = {"sub": "test@example.com"}

            with pytest.raises(HTTPException) as exc_info:
                await get_current_user_or_api_key(
                    authorization="Bearer blacklisted_token", x_api_key=None
                )

            assert exc_info.value.status_code == 401
            assert "Token has been revoked" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_jwt_authentication_user_not_found(self):
        """Test JWT authentication when user is not found."""
        with (
            patch("app.api.search.verify_token") as mock_verify,
            patch("app.api.search.is_token_blacklisted") as mock_blacklist,
            patch("app.api.search.users_storage") as mock_storage,
        ):

            mock_blacklist.return_value = False
            mock_verify.return_value = {"sub": "nonexistent@example.com"}
            mock_storage.get.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                await get_current_user_or_api_key(
                    authorization="Bearer test_token", x_api_key=None
                )

            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_jwt_authentication_invalid_payload(self):
        """Test JWT authentication with invalid payload (no email)."""
        with (
            patch("app.api.search.verify_token") as mock_verify,
            patch("app.api.search.is_token_blacklisted") as mock_blacklist,
        ):

            mock_blacklist.return_value = False
            mock_verify.return_value = {}  # No 'sub' field

            with pytest.raises(HTTPException) as exc_info:
                await get_current_user_or_api_key(
                    authorization="Bearer test_token", x_api_key=None
                )

            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_jwt_authentication_exception_handling(self):
        """Test JWT authentication exception handling and fallback."""
        with patch("app.api.search.verify_token") as mock_verify:
            mock_verify.side_effect = Exception("Token verification failed")

            with pytest.raises(HTTPException) as exc_info:
                await get_current_user_or_api_key(
                    authorization="Bearer invalid_token", x_api_key=None
                )

            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_no_authentication_provided(self):
        """Test when no authentication is provided."""
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_or_api_key(authorization=None, x_api_key=None)

        assert exc_info.value.status_code == 401
        assert "Authentication required" in str(exc_info.value.detail)


class TestSearchEngineDependency:
    """Test cases for get_hybrid_search_engine dependency."""

    @pytest.mark.asyncio
    async def test_get_hybrid_search_engine_initialization(self):
        """Test hybrid search engine dependency initialization."""
        with (
            patch("app.services.embedding_service.EmbeddingService") as MockEmbedding,
            patch(
                "app.repositories.document_repository.DocumentRepository"
            ) as MockDocRepo,
            patch(
                "app.repositories.chunk_repository.DocumentChunkRepository"
            ) as MockChunkRepo,
        ):

            # Mock embedding service
            mock_embedding_instance = AsyncMock()
            mock_embedding_instance.initialize = AsyncMock()
            MockEmbedding.return_value = mock_embedding_instance

            # Mock repositories
            MockDocRepo.return_value = MagicMock()
            MockChunkRepo.return_value = MagicMock()

            # Get search engine
            engine = await get_hybrid_search_engine()

            assert isinstance(engine, HybridSearchEngine)
            assert engine.config.dense_weight == 0.7
            assert engine.config.sparse_weight == 0.3
            assert engine.config.top_k == 20
            assert engine.config.search_mode == SearchMode.HYBRID
            assert engine.config.enable_reranking is True

            # Verify embedding service was initialized
            mock_embedding_instance.initialize.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions in search module."""

    def test_highlight_content_basic(self):
        """Test basic content highlighting."""
        content = "This is a test document about machine learning and AI."
        query = "machine learning"

        result = highlight_content(content, query)

        assert "**machine**" in result
        assert "**learning**" in result

    def test_highlight_content_empty_query(self):
        """Test highlighting with empty query."""
        content = "Test content"

        result = highlight_content(content, "")
        assert result == content

        result = highlight_content(content, None)
        assert result == content

    def test_highlight_content_empty_content(self):
        """Test highlighting with empty content."""
        result = highlight_content("", "query")
        assert result == ""

        result = highlight_content(None, "query")
        assert result is None

    def test_highlight_content_short_words_skipped(self):
        """Test that short words are skipped in highlighting."""
        content = "This is a test of highlighting"
        query = "is a of"  # All words are 2 chars or less

        result = highlight_content(content, query)
        assert "**" not in result  # No highlighting should occur

    def test_convert_enhanced_filters_comprehensive(self):
        """Test comprehensive filter conversion."""
        enhanced_filters = EnhancedFilters(
            source_types=["api", "docs"],
            languages=["en", "ja"],
            tags=["python", "fastapi"],
            date_range=DateRange(from_date="2024-01-01", to_date="2024-12-31"),
        )

        result = convert_enhanced_filters_to_legacy(enhanced_filters)

        assert len(result) == 5  # 2 source_types + 2 languages + 1 tags + 2 date_range

        # Check source types filter
        source_filter = next(f for f in result if f.field == "source_type")
        assert source_filter.value == ["api", "docs"]
        assert source_filter.operator == "in"

        # Check language filter
        lang_filter = next(f for f in result if f.field == "language")
        assert lang_filter.value == ["en", "ja"]
        assert lang_filter.operator == "in"

        # Check tags filter
        tags_filter = next(f for f in result if f.field == "metadata.tags")
        assert tags_filter.value == ["python", "fastapi"]
        assert tags_filter.operator == "contains_any"

        # Check date range filters
        date_filters = [f for f in result if f.field == "updated_at"]
        assert len(date_filters) == 2
        assert any(f.operator == "gte" for f in date_filters)
        assert any(f.operator == "lte" for f in date_filters)

    def test_convert_enhanced_filters_none(self):
        """Test filter conversion with None input."""
        result = convert_enhanced_filters_to_legacy(None)
        assert result == []

    def test_convert_enhanced_filters_empty(self):
        """Test filter conversion with empty filters."""
        result = convert_enhanced_filters_to_legacy(EnhancedFilters())
        assert result == []

    def test_generate_search_suggestions_with_tags(self):
        """Test search suggestion generation with tags."""
        query = "python"
        results = [
            {
                "metadata": {
                    "tags": ["tutorial", "beginner", "advanced", "api", "framework"]
                }
            },
            {"metadata": {"tags": ["guide", "intermediate"]}},
        ]

        suggestions = generate_search_suggestions(query, results)

        assert len(suggestions) <= 5
        assert all(query in s or s in query for s in suggestions)

    def test_generate_search_suggestions_no_metadata(self):
        """Test suggestion generation with results lacking metadata."""
        query = "test"
        results = [
            {"content": "Test content"},
            {"metadata": "invalid"},  # Invalid metadata type
            {"metadata": {"no_tags": True}},  # No tags field
        ]

        suggestions = generate_search_suggestions(query, results)

        assert isinstance(suggestions, list)
        assert len(suggestions) <= 5


class TestSearchEndpoints:
    """Test search API endpoints for missing coverage."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers."""
        return {"X-API-Key": "test_api_key"}

    @pytest.fixture
    def mock_search_engine(self):
        """Create mock search engine."""
        engine = AsyncMock(spec=HybridSearchEngine)
        engine.config = SearchConfig()
        engine.search = AsyncMock()
        return engine

    @pytest.mark.asyncio
    async def test_search_documents_comprehensive(self, client, auth_headers):
        """Test comprehensive search with all options."""
        with (
            patch("app.api.search.get_current_user_or_api_key") as mock_auth,
            patch("app.api.search.get_hybrid_search_engine") as mock_get_engine,
        ):

            # Mock authentication
            mock_auth.return_value = {"user_id": "test_user", "permissions": ["read"]}

            # Mock search engine
            mock_engine = AsyncMock()
            mock_engine.config = SearchConfig()
            mock_engine.search = AsyncMock(
                return_value=SearchResult(
                    success=True,
                    query="test query",
                    total_hits=2,
                    search_time=0.5,
                    documents=[
                        {
                            "id": "doc1",
                            "document_id": "doc1",
                            "chunk_id": "chunk1",
                            "title": "Test Document",
                            "content": "Test content about machine learning",
                            "search_score": 0.95,
                            "source_type": "api",
                            "language": "en",
                            "document_type": "guide",
                            "metadata": {
                                "tags": ["ml", "ai"],
                                "url": "https://example.com",
                                "author": "Test Author",
                            },
                            "updated_at": datetime.now(),
                            "hierarchy_path": ["docs", "guides", "ml"],
                            "chunk_type": "section",
                        },
                        {
                            "id": "doc2",
                            "document_id": "doc2",
                            "title": "Another Document",
                            "content": "Another test content",
                            "search_score": 0.85,
                            "metadata": {},
                        },
                    ],
                    facets={
                        "source_type": [
                            {"value": "api", "count": 10},
                            {"value": "docs", "count": 5},
                        ]
                    },
                )
            )
            mock_get_engine.return_value = mock_engine

            # Make request with all options
            request_data = {
                "query": "test query",
                "filters": {
                    "source_types": ["api", "docs"],
                    "languages": ["en"],
                    "tags": ["ml"],
                    "date_range": {
                        "from_date": "2024-01-01T00:00:00",
                        "to_date": "2024-12-31T23:59:59",
                    },
                },
                "search_options": {
                    "search_type": "hybrid",
                    "max_results": 20,
                    "min_score": 0.5,
                    "include_metadata": True,
                    "highlight": True,
                },
                "ranking_options": {
                    "dense_weight": 0.6,
                    "sparse_weight": 0.4,
                    "rerank": True,
                    "diversity": True,
                },
                "facets": ["source_type", "language"],
            }

            response = client.post(
                "/api/v1/search/", json=request_data, headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()

            assert data["query"] == "test query"
            assert data["total_results"] == 2
            assert len(data["results"]) == 2

            # Check first result structure
            first_result = data["results"][0]
            assert first_result["document_id"] == "doc1"
            assert first_result["score"] == 0.95
            assert first_result["highlighted_content"] is not None
            assert "**" in first_result["highlighted_content"]  # Highlighting applied

            # Check source info
            assert first_result["source"] is not None
            assert first_result["source"]["type"] == "api"
            assert first_result["source"]["url"] == "https://example.com"

            # Check context info
            assert first_result["context"] is not None
            assert first_result["context"]["hierarchy_path"] == ["docs", "guides", "ml"]

    @pytest.mark.asyncio
    async def test_search_documents_weight_normalization(self, client, auth_headers):
        """Test search with weight normalization."""
        with (
            patch("app.api.search.get_current_user_or_api_key") as mock_auth,
            patch("app.api.search.get_hybrid_search_engine") as mock_get_engine,
        ):

            mock_auth.return_value = {"user_id": "test", "permissions": ["read"]}

            mock_engine = AsyncMock()
            mock_engine.config = SearchConfig()
            mock_engine.search = AsyncMock(
                return_value=SearchResult(
                    success=True,
                    query="test",
                    total_hits=0,
                    search_time=0.1,
                    documents=[],
                )
            )
            mock_get_engine.return_value = mock_engine

            # Test with weights that don't sum to 1.0
            request_data = {
                "query": "test",
                "ranking_options": {
                    "dense_weight": 0.3,
                    "sparse_weight": 0.2,  # Sum is 0.5
                },
            }

            response = client.post(
                "/api/v1/search/", json=request_data, headers=auth_headers
            )

            assert response.status_code == 200
            # Check that weights were normalized
            assert mock_engine.config.dense_weight == 0.6  # 0.3 / 0.5
            assert mock_engine.config.sparse_weight == 0.4  # 0.2 / 0.5

    @pytest.mark.asyncio
    async def test_search_documents_zero_weights(self, client, auth_headers):
        """Test search with zero weights."""
        with (
            patch("app.api.search.get_current_user_or_api_key") as mock_auth,
            patch("app.api.search.get_hybrid_search_engine") as mock_get_engine,
        ):

            mock_auth.return_value = {"user_id": "test", "permissions": ["read"]}

            mock_engine = AsyncMock()
            mock_engine.config = SearchConfig()
            mock_engine.search = AsyncMock(
                return_value=SearchResult(
                    success=True,
                    query="test",
                    total_hits=0,
                    search_time=0.1,
                    documents=[],
                )
            )
            mock_get_engine.return_value = mock_engine

            # Test with zero weights
            request_data = {
                "query": "test",
                "ranking_options": {"dense_weight": 0.0, "sparse_weight": 0.0},
            }

            response = client.post(
                "/api/v1/search/", json=request_data, headers=auth_headers
            )

            assert response.status_code == 200
            # Check that default weights were applied
            assert mock_engine.config.dense_weight == 0.7
            assert mock_engine.config.sparse_weight == 0.3

    @pytest.mark.asyncio
    async def test_search_documents_legacy_compatibility(self, client, auth_headers):
        """Test search with legacy fields."""
        with (
            patch("app.api.search.get_current_user_or_api_key") as mock_auth,
            patch("app.api.search.get_hybrid_search_engine") as mock_get_engine,
        ):

            mock_auth.return_value = {"user_id": "test", "permissions": ["read"]}

            mock_engine = AsyncMock()
            mock_engine.config = SearchConfig()
            mock_engine.search = AsyncMock(
                return_value=SearchResult(
                    success=True,
                    query="test",
                    total_hits=1,
                    search_time=0.1,
                    documents=[{"id": "doc1", "content": "test"}],
                )
            )
            mock_get_engine.return_value = mock_engine

            # Test with legacy fields
            request_data = {
                "query": "test",
                "max_results": 50,  # Legacy field
                "dense_weight": 0.8,  # Legacy field
                "sparse_weight": 0.2,  # Legacy field
                "similarity_threshold": 0.7,  # Legacy field
                "enable_reranking": False,  # Legacy field
                "search_mode": "semantic",  # Legacy field
                "legacy_filters": [{"field": "type", "value": "api", "operator": "eq"}],
            }

            response = client.post(
                "/api/v1/search/", json=request_data, headers=auth_headers
            )

            assert response.status_code == 200
            # Verify legacy fields were processed
            assert mock_engine.config.dense_weight == 0.8
            assert mock_engine.config.sparse_weight == 0.2
            assert mock_engine.config.similarity_threshold == 0.7
            assert mock_engine.config.enable_reranking is False

    @pytest.mark.asyncio
    async def test_search_documents_no_permission(self, client, auth_headers):
        """Test search without read permission."""
        with patch("app.api.search.get_current_user_or_api_key") as mock_auth:
            mock_auth.return_value = {
                "user_id": "test",
                "permissions": ["write"],
            }  # No read

            response = client.post(
                "/api/v1/search/", json={"query": "test"}, headers=auth_headers
            )

            assert response.status_code == 403
            assert "Read permission required" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_search_documents_search_failure(self, client, auth_headers):
        """Test search when search engine returns failure."""
        with (
            patch("app.api.search.get_current_user_or_api_key") as mock_auth,
            patch("app.api.search.get_hybrid_search_engine") as mock_get_engine,
        ):

            mock_auth.return_value = {"user_id": "test", "permissions": ["read"]}

            mock_engine = AsyncMock()
            mock_engine.config = SearchConfig()
            mock_engine.search = AsyncMock(
                return_value=SearchResult(
                    success=False,
                    query="test",
                    total_hits=0,
                    search_time=0.1,
                    documents=[],
                    error_message="Search index unavailable",
                )
            )
            mock_get_engine.return_value = mock_engine

            response = client.post(
                "/api/v1/search/", json={"query": "test"}, headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert data["error_message"] == "Search index unavailable"
            assert data["total_results"] == 0

    @pytest.mark.asyncio
    async def test_search_documents_exception(self, client, auth_headers):
        """Test search when an exception occurs."""
        with (
            patch("app.api.search.get_current_user_or_api_key") as mock_auth,
            patch("app.api.search.get_hybrid_search_engine") as mock_get_engine,
        ):

            mock_auth.return_value = {"user_id": "test", "permissions": ["read"]}

            mock_engine = AsyncMock()
            mock_engine.config = SearchConfig()
            mock_engine.search = AsyncMock(
                side_effect=Exception("Database connection failed")
            )
            mock_get_engine.return_value = mock_engine

            response = client.post(
                "/api/v1/search/", json={"query": "test"}, headers=auth_headers
            )

            assert response.status_code == 500
            assert "Search failed" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_semantic_search_endpoint(self, client, auth_headers):
        """Test semantic search endpoint."""
        with (
            patch("app.api.search.get_current_user_or_api_key") as mock_auth,
            patch("app.api.search.get_hybrid_search_engine") as mock_get_engine,
        ):

            mock_auth.return_value = {"user_id": "test", "permissions": ["read"]}

            mock_engine = AsyncMock()
            mock_engine.config = SearchConfig()
            mock_engine.search = AsyncMock(
                return_value=SearchResult(
                    success=True,
                    query="semantic test",
                    total_hits=1,
                    search_time=0.2,
                    documents=[
                        {
                            "id": "doc1",
                            "title": "Semantic Result",
                            "content": "Content found by semantic search",
                            "search_score": 0.92,
                            "chunk_id": "chunk1",
                            "chunk_type": "paragraph",
                            "source_type": "docs",
                            "language": "en",
                            "document_type": "tutorial",
                            "metadata": {"category": "ml"},
                            "rerank_score": 0.95,
                            "ranking_explanation": "High semantic similarity",
                        }
                    ],
                    facets={
                        "category": [
                            {"value": "ml", "count": 5},
                            {"value": "ai", "count": 3},
                        ]
                    },
                )
            )
            mock_get_engine.return_value = mock_engine

            request_data = {
                "query": "semantic test",
                "legacy_filters": [
                    {"field": "type", "value": "docs", "operator": "eq"}
                ],
                "facets": ["category"],
                "similarity_threshold": 0.8,
                "enable_reranking": True,
            }

            response = client.post(
                "/api/v1/search/semantic", json=request_data, headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()

            # Verify semantic search configuration
            assert mock_engine.config.search_mode == SearchMode.SEMANTIC
            assert mock_engine.config.dense_weight == 1.0
            assert mock_engine.config.sparse_weight == 0.0

            # Check response
            assert data["total_results"] == 1
            assert data["results"][0]["title"] == "Semantic Result"
            assert data["results"][0]["rerank_score"] == 0.95

    @pytest.mark.asyncio
    async def test_semantic_search_failure(self, client, auth_headers):
        """Test semantic search with failure."""
        with (
            patch("app.api.search.get_current_user_or_api_key") as mock_auth,
            patch("app.api.search.get_hybrid_search_engine") as mock_get_engine,
        ):

            mock_auth.return_value = {"user_id": "test", "permissions": ["read"]}

            mock_engine = AsyncMock()
            mock_engine.config = SearchConfig()
            mock_engine.search = AsyncMock(
                return_value=SearchResult(
                    success=False,
                    query="test",
                    total_hits=0,
                    search_time=0.1,
                    documents=[],
                    error_message="Embedding service unavailable",
                )
            )
            mock_get_engine.return_value = mock_engine

            response = client.post(
                "/api/v1/search/semantic", json={"query": "test"}, headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert data["error_message"] == "Embedding service unavailable"

    @pytest.mark.asyncio
    async def test_keyword_search_endpoint(self, client, auth_headers):
        """Test keyword search endpoint."""
        with (
            patch("app.api.search.get_current_user_or_api_key") as mock_auth,
            patch("app.api.search.get_hybrid_search_engine") as mock_get_engine,
        ):

            mock_auth.return_value = {"user_id": "test", "permissions": ["read"]}

            mock_engine = AsyncMock()
            mock_engine.config = SearchConfig()
            mock_engine.search = AsyncMock(
                return_value=SearchResult(
                    success=True,
                    query="keyword test",
                    total_hits=2,
                    search_time=0.1,
                    documents=[
                        {
                            "id": "doc1",
                            "title": "Exact Match",
                            "content": "Document with keyword test",
                            "search_score": 1.0,
                            "metadata": {},
                        },
                        {
                            "id": "doc2",
                            "title": "Partial Match",
                            "content": "Another keyword document",
                            "search_score": 0.8,
                            "metadata": {},
                        },
                    ],
                )
            )
            mock_get_engine.return_value = mock_engine

            response = client.post(
                "/api/v1/search/keyword",
                json={"query": "keyword test"},
                headers=auth_headers,
            )

            assert response.status_code == 200

            # Verify keyword search configuration
            assert mock_engine.config.search_mode == SearchMode.KEYWORD
            assert mock_engine.config.dense_weight == 0.0
            assert mock_engine.config.sparse_weight == 1.0

    @pytest.mark.asyncio
    async def test_search_suggestions_endpoint(self, client, auth_headers):
        """Test search suggestions endpoint."""
        with patch("app.api.search.get_current_user_or_api_key") as mock_auth:
            mock_auth.return_value = {"user_id": "test", "permissions": ["read"]}

            response = client.get(
                "/api/v1/search/suggestions?q=machine&limit=3", headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()

            assert "suggestions" in data
            assert data["query"] == "machine"
            assert len(data["suggestions"]) <= 3
            # Should contain "machine learning" based on base suggestions
            assert any("machine" in s.lower() for s in data["suggestions"])

    @pytest.mark.asyncio
    async def test_search_suggestions_empty_query(self, client, auth_headers):
        """Test search suggestions with empty query."""
        with patch("app.api.search.get_current_user_or_api_key") as mock_auth:
            mock_auth.return_value = {"user_id": "test", "permissions": ["read"]}

            response = client.get("/api/v1/search/suggestions?q=", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert data["suggestions"] == []

    @pytest.mark.asyncio
    async def test_search_suggestions_no_permission(self, client, auth_headers):
        """Test search suggestions without read permission."""
        with patch("app.api.search.get_current_user_or_api_key") as mock_auth:
            mock_auth.return_value = {"user_id": "test", "permissions": ["write"]}

            response = client.get(
                "/api/v1/search/suggestions?q=test", headers=auth_headers
            )

            assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_search_suggestions_exception(self, client, auth_headers):
        """Test search suggestions with exception."""
        with patch("app.api.search.get_current_user_or_api_key") as mock_auth:
            mock_auth.return_value = {"user_id": "test", "permissions": ["read"]}

            # Mock to raise exception in the endpoint
            with patch("app.api.search.logger.error"):
                # Force an exception by patching something inside the endpoint
                with patch.object(str, "lower", side_effect=Exception("Test error")):
                    response = client.get(
                        "/api/v1/search/suggestions?q=test", headers=auth_headers
                    )

                assert response.status_code == 500
                assert "Suggestions failed" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_search_config_endpoint(self, client, auth_headers):
        """Test search configuration endpoint."""
        with patch("app.api.search.get_current_user_or_api_key") as mock_auth:
            mock_auth.return_value = {"user_id": "test", "permissions": ["read"]}

            response = client.get("/api/v1/search/config", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()

            # Check structure
            assert "search_modes" in data
            assert "ranking_algorithms" in data
            assert "default_config" in data
            assert "available_filters" in data
            assert "available_facets" in data

            # Check default config values
            default_config = data["default_config"]
            assert default_config["dense_weight"] == 0.7
            assert default_config["sparse_weight"] == 0.3
            assert default_config["enable_reranking"] is True

    @pytest.mark.asyncio
    async def test_search_config_no_permission(self, client, auth_headers):
        """Test search config without permission."""
        with patch("app.api.search.get_current_user_or_api_key") as mock_auth:
            mock_auth.return_value = {"user_id": "test", "permissions": []}

            response = client.get("/api/v1/search/config", headers=auth_headers)

            assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_search_config_exception(self, client, auth_headers):
        """Test search config with exception."""
        with patch("app.api.search.get_current_user_or_api_key") as mock_auth:
            mock_auth.return_value = {"user_id": "test", "permissions": ["read"]}

            with patch(
                "app.api.search.SearchMode", side_effect=Exception("Enum error")
            ):
                response = client.get("/api/v1/search/config", headers=auth_headers)

            assert response.status_code == 500
            assert "Config retrieval failed" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_search_semantic_alias_endpoint(self, client, auth_headers):
        """Test /search/semantic alias endpoint."""
        with patch("app.api.search.search_documents") as mock_search_documents:
            mock_search_documents.return_value = {
                "success": True,
                "query": "test",
                "total_results": 1,
                "results": [],
            }

            request_data = {"query": "test semantic alias"}
            response = client.post(
                "/api/v1/search/semantic", json=request_data, headers=auth_headers
            )

            # The alias should call search_documents with SEMANTIC mode
            assert response.status_code == 200

            # Verify the request was modified to use SEMANTIC mode
            call_args = mock_search_documents.call_args
            request_arg = call_args[0][0]
            assert request_arg.search_mode == SearchMode.SEMANTIC

    @pytest.mark.asyncio
    async def test_search_keyword_alias_endpoint(self, client, auth_headers):
        """Test /search/keyword alias endpoint."""
        with patch("app.api.search.search_documents") as mock_search_documents:
            mock_search_documents.return_value = {
                "success": True,
                "query": "test",
                "total_results": 1,
                "results": [],
            }

            request_data = {"query": "test keyword alias"}
            response = client.post(
                "/api/v1/search/keyword", json=request_data, headers=auth_headers
            )

            # The alias should call search_documents with KEYWORD mode
            assert response.status_code == 200

            # Verify the request was modified to use KEYWORD mode
            call_args = mock_search_documents.call_args
            request_arg = call_args[0][0]
            assert request_arg.search_mode == SearchMode.KEYWORD


class TestSearchModeConfiguration:
    """Test search mode configuration in different endpoints."""

    @pytest.mark.asyncio
    async def test_search_mode_dense(self):
        """Test search options with dense search type."""
        client = TestClient(app)

        with (
            patch("app.api.search.get_current_user_or_api_key") as mock_auth,
            patch("app.api.search.get_hybrid_search_engine") as mock_get_engine,
        ):

            mock_auth.return_value = {"user_id": "test", "permissions": ["read"]}

            mock_engine = AsyncMock()
            mock_engine.config = SearchConfig()
            mock_engine.search = AsyncMock(
                return_value=SearchResult(
                    success=True,
                    query="test",
                    total_hits=0,
                    search_time=0.1,
                    documents=[],
                )
            )
            mock_get_engine.return_value = mock_engine

            request_data = {"query": "test", "search_options": {"search_type": "dense"}}

            response = client.post(
                "/api/v1/search/", json=request_data, headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200
            # Verify search was called with SEMANTIC mode
            search_call = mock_engine.search.call_args[0][0]
            assert search_call.search_mode == SearchMode.SEMANTIC

    @pytest.mark.asyncio
    async def test_search_mode_sparse(self):
        """Test search options with sparse search type."""
        client = TestClient(app)

        with (
            patch("app.api.search.get_current_user_or_api_key") as mock_auth,
            patch("app.api.search.get_hybrid_search_engine") as mock_get_engine,
        ):

            mock_auth.return_value = {"user_id": "test", "permissions": ["read"]}

            mock_engine = AsyncMock()
            mock_engine.config = SearchConfig()
            mock_engine.search = AsyncMock(
                return_value=SearchResult(
                    success=True,
                    query="test",
                    total_hits=0,
                    search_time=0.1,
                    documents=[],
                )
            )
            mock_get_engine.return_value = mock_engine

            request_data = {
                "query": "test",
                "search_options": {"search_type": "sparse"},
            }

            response = client.post(
                "/api/v1/search/", json=request_data, headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200
            # Verify search was called with KEYWORD mode
            search_call = mock_engine.search.call_args[0][0]
            assert search_call.search_mode == SearchMode.KEYWORD


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
