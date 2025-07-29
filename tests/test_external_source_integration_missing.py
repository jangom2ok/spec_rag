"""Additional tests to achieve 100% coverage for external_source_integration.py

Missing lines: 203, 205-206, 234-246, 277-282, 305-306, 338, 511, 516, 521, 563, 565, 571-573, 581-596
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.services.external_source_integration import (
    AuthenticationError,
    AuthType,
    ConfluenceConnector,
    ExternalSourceIntegrator,
    JiraConnector,
    RateLimitError,
    SourceConfig,
    SourceType,
)


class TestMissingCoverage:
    """Tests for missing coverage lines"""

    @pytest.fixture
    def confluence_config(self):
        """Confluence config"""
        return SourceConfig(
            source_type=SourceType.CONFLUENCE,
            base_url="https://example.com",
            auth_type=AuthType.API_TOKEN,
            api_token="token",
            username="user",
            max_pages=2,
            batch_size=2,
        )

    @pytest.fixture
    def jira_config(self):
        """JIRA config"""
        return SourceConfig(
            source_type=SourceType.JIRA,
            base_url="https://example.com",
            auth_type=AuthType.API_TOKEN,
            api_token="token",
            username="user",
            project_key=None,  # No project key
            search_query=None,  # No search query
            filters={},  # No filters
        )

    @pytest.mark.unit
    async def test_authentication_error_in_make_request(self, confluence_config):
        """Test AuthenticationError (line 203)"""
        connector = ConfluenceConnector(confluence_config)

        async with connector:
            mock_response = AsyncMock()
            mock_response.status = 401
            if connector._session:
                connector._session.request = AsyncMock(return_value=mock_response)  # type: ignore

            with pytest.raises(AuthenticationError, match="Authentication failed"):
                await connector._make_request("GET", "https://example.com")

    @pytest.mark.unit
    async def test_rate_limit_error_in_make_request(self, confluence_config):
        """Test RateLimitError with Retry-After header (lines 205-206)"""
        connector = ConfluenceConnector(confluence_config)

        async with connector:
            mock_response = AsyncMock()
            mock_response.status = 429
            mock_response.headers = {"Retry-After": "120"}
            if connector._session:
                connector._session.request = AsyncMock(return_value=mock_response)  # type: ignore

            with pytest.raises(
                RateLimitError, match="Rate limit exceeded. Retry after 120 seconds"
            ):
                await connector._make_request("GET", "https://example.com")

    @pytest.mark.unit
    async def test_confluence_test_connection_success(self, confluence_config):
        """Test Confluence test_connection success (lines 234-246)"""
        connector = ConfluenceConnector(confluence_config)

        async with connector:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "type": "user",
                "username": "test_user",
                "email": "test@example.com",
            }
            if connector._session:
                connector._session.request = AsyncMock(return_value=mock_response)  # type: ignore

            result = await connector.test_connection()

            assert result["success"] is True
            assert result["message"] == "Authentication successful"
            assert result["user"] == "test_user"

    @pytest.mark.unit
    async def test_confluence_test_connection_failure(self, confluence_config):
        """Test Confluence test_connection failure (line 246)"""
        connector = ConfluenceConnector(confluence_config)

        async with connector:
            if connector._session is not None:
                connector._session.request = AsyncMock(
                    side_effect=Exception("Network error")
                )

            result = await connector.test_connection()

            assert result["success"] is False
            assert "Authentication failed: Network error" in result["message"]

    @pytest.mark.unit
    async def test_confluence_fetch_pages_pagination_limit(self, confluence_config):
        """Test Confluence pagination reaching max_pages limit (lines 277-282)"""
        connector = ConfluenceConnector(confluence_config)

        # Create 3 batches of 2 pages each, but max_pages is 2
        batch_calls = 0

        async def mock_fetch_batch(start, limit, incremental):
            nonlocal batch_calls
            batch_calls += 1
            if batch_calls <= 3:
                return [{"id": f"page-{start+i}"} for i in range(limit)]
            return []

        with patch.object(
            connector, "_fetch_pages_batch", side_effect=mock_fetch_batch
        ):
            pages = await connector._fetch_all_pages()

            assert len(pages) == 2  # Limited by max_pages
            assert pages[0]["id"] == "page-0"
            assert pages[1]["id"] == "page-1"

    @pytest.mark.unit
    async def test_confluence_fetch_pages_without_search_query(self, confluence_config):
        """Test fetching pages without search query (lines 305-306)"""
        confluence_config.search_query = None  # No search query
        connector = ConfluenceConnector(confluence_config)

        async with connector:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "results": [{"id": "123", "title": "Test"}]
            }

            mock_request = AsyncMock(return_value=mock_response)
            if connector._session is not None:
                connector._session.request = mock_request

            await connector._fetch_pages_batch(0, 10)

            # Check URL and params for regular content API
            call_args = mock_request.call_args
            assert "/rest/api/content" in call_args[0][1]
            assert call_args[1]["params"]["type"] == "page"

    @pytest.mark.unit
    def test_confluence_page_with_custom_properties(self, confluence_config):
        """Test page transformation with custom properties (line 338)"""
        connector = ConfluenceConnector(confluence_config)

        page_with_properties = {
            "id": "123",
            "title": "Test Page",
            "body": {"storage": {"value": "<p>content</p>"}},
            "version": {"number": 1, "when": "2024-01-01T00:00:00Z"},
            "_links": {"webui": "/pages/123"},
            "metadata": {"properties": {"custom1": "value1", "custom2": "value2"}},
        }

        document = connector._transform_page_to_document(page_with_properties)

        assert document["metadata"]["custom_properties"] == {
            "custom1": "value1",
            "custom2": "value2",
        }

    @pytest.mark.unit
    def test_jira_issue_with_priority(self, jira_config):
        """Test JIRA issue transformation with priority (lines 510-513)"""
        connector = JiraConnector(jira_config)

        issue = {
            "key": "TEST-1",
            "fields": {
                "summary": "Test Issue",
                "description": "Test description",
                "issuetype": {"name": "Bug"},
                "status": {"name": "Open"},
                "priority": {"name": "High", "id": "1"},  # With priority
                "project": {"key": "TEST", "name": "Test Project"},
                "created": "2024-01-01T00:00:00.000+0000",
                "updated": "2024-01-01T00:00:00.000+0000",
            },
        }

        document = connector._transform_issue_to_document(issue)

        assert "Priority: High" in document["content"]
        assert document["metadata"]["priority"] == "High"

    @pytest.mark.unit
    def test_jira_issue_with_assignee(self, jira_config):
        """Test JIRA issue transformation with assignee (lines 515-518)"""
        connector = JiraConnector(jira_config)

        issue = {
            "key": "TEST-1",
            "fields": {
                "summary": "Test Issue",
                "issuetype": {"name": "Bug"},
                "status": {"name": "Open"},
                "assignee": {
                    "displayName": "John Doe",
                    "emailAddress": "john@example.com",
                },
                "project": {"key": "TEST", "name": "Test Project"},
                "created": "2024-01-01T00:00:00.000+0000",
                "updated": "2024-01-01T00:00:00.000+0000",
            },
        }

        document = connector._transform_issue_to_document(issue)

        assert "Assignee: John Doe" in document["content"]
        assert document["metadata"]["assignee"] == "John Doe"

    @pytest.mark.unit
    def test_jira_issue_with_reporter(self, jira_config):
        """Test JIRA issue transformation with reporter (lines 520-523)"""
        connector = JiraConnector(jira_config)

        issue = {
            "key": "TEST-1",
            "fields": {
                "summary": "Test Issue",
                "issuetype": {"name": "Bug"},
                "status": {"name": "Open"},
                "reporter": {
                    "displayName": "Jane Smith",
                    "emailAddress": "jane@example.com",
                },
                "project": {"key": "TEST", "name": "Test Project"},
                "created": "2024-01-01T00:00:00.000+0000",
                "updated": "2024-01-01T00:00:00.000+0000",
            },
        }

        document = connector._transform_issue_to_document(issue)

        assert "Reporter: Jane Smith" in document["content"]
        assert document["metadata"]["reporter"] == "Jane Smith"

    @pytest.mark.unit
    def test_create_confluence_connector(self, confluence_config):
        """Test creating Confluence connector (line 563)"""
        integrator = ExternalSourceIntegrator(confluence_config)
        connector = integrator._create_connector()

        assert isinstance(connector, ConfluenceConnector)
        assert connector.config == confluence_config

    @pytest.mark.unit
    def test_create_jira_connector(self, jira_config):
        """Test creating JIRA connector (line 565)"""
        integrator = ExternalSourceIntegrator(jira_config)
        connector = integrator._create_connector()

        assert isinstance(connector, JiraConnector)
        assert connector.config == jira_config

    @pytest.mark.unit
    async def test_test_connection_wrapper(self, confluence_config):
        """Test test_connection wrapper method (lines 571-573)"""
        integrator = ExternalSourceIntegrator(confluence_config)

        mock_result = {"success": True, "message": "Connected"}

        with patch.object(integrator, "_create_connector") as mock_create:
            mock_connector = AsyncMock()
            mock_connector.__aenter__.return_value = mock_connector
            mock_connector.__aexit__.return_value = None
            mock_connector.test_connection.return_value = mock_result
            mock_create.return_value = mock_connector

            result = await integrator.test_connection()

            assert result == mock_result

    @pytest.mark.unit
    async def test_fetch_documents_success_with_batches(self, confluence_config):
        """Test successful document fetch with batch processing (lines 581-596)"""
        integrator = ExternalSourceIntegrator(confluence_config)

        # Mock documents from connector
        mock_docs = [
            {
                "id": f"doc-{i}",
                "title": f"Document {i}",
                "content": f"Content {i}",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
            }
            for i in range(5)
        ]

        with patch.object(integrator, "_create_connector") as mock_create:
            mock_connector = AsyncMock()
            mock_connector.__aenter__.return_value = mock_connector
            mock_connector.__aexit__.return_value = None
            mock_connector.fetch_documents.return_value = mock_docs
            mock_create.return_value = mock_connector

            result = await integrator.fetch_documents(incremental=True)

            assert result.success is True
            assert result.total_count == 5
            assert len(result.documents) == 5
            assert result.batch_count == 3  # 5 docs with batch_size=2
            assert result.processing_time > 0
            assert result.source_type == SourceType.CONFLUENCE
            assert result.last_sync_time is not None

            # Check that documents were processed
            for doc in result.documents:
                assert "metadata" in doc
                assert doc["metadata"]["integration_timestamp"]
                assert doc["metadata"]["source_system"] == SourceType.CONFLUENCE
                assert doc["created_at"].endswith("Z")  # Normalized
                assert doc["updated_at"].endswith("Z")  # Normalized

    @pytest.mark.unit
    async def test_jira_fetch_issues_no_jql_filters(self, jira_config):
        """Test JIRA fetch with no JQL filters - default ordering"""
        connector = JiraConnector(jira_config)

        async with connector:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"issues": []}

            if connector._session:
                connector._session.request = AsyncMock(return_value=mock_response)  # type: ignore

            await connector._fetch_issues_batch(0, 50, incremental=False)

            # Check that default JQL is used
            if connector._session:
                # Access call_args on the mock object
                call_args = connector._session.request.call_args  # type: ignore[attr-defined]
                payload = call_args[1]["json"]
                assert payload["jql"] == "order by created DESC"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
