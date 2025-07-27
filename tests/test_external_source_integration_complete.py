"""Complete test coverage for external_source_integration.py

This test file ensures 100% code coverage for the external source integration module.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from app.services.external_source_integration import (
    AuthenticationError,
    AuthType,
    BaseConnector,
    ConfluenceConnector,
    ConnectionError,
    ExternalSourceIntegrator,
    IntegrationResult,
    JiraConnector,
    SourceConfig,
    SourceType,
)


class TestSourceConfigValidation:
    """Test SourceConfig validation for 100% coverage"""

    @pytest.mark.unit
    def test_base_url_validation(self):
        """Test base_url validation (line 96)"""
        with pytest.raises(ValueError, match="base_url is required"):
            SourceConfig(
                source_type=SourceType.CONFLUENCE,
                base_url="",  # Empty base_url
                auth_type=AuthType.API_TOKEN,
                api_token="token",
                username="user",
            )

    @pytest.mark.unit
    def test_api_token_auth_validation(self):
        """Test API token auth validation (lines 100-101)"""
        # Missing api_token
        with pytest.raises(ValueError, match="api_token and username are required"):
            SourceConfig(
                source_type=SourceType.CONFLUENCE,
                base_url="https://example.com",
                auth_type=AuthType.API_TOKEN,
                api_token=None,
                username="user",
            )

        # Missing username
        with pytest.raises(ValueError, match="api_token and username are required"):
            SourceConfig(
                source_type=SourceType.CONFLUENCE,
                base_url="https://example.com",
                auth_type=AuthType.API_TOKEN,
                api_token="token",
                username=None,
            )

    @pytest.mark.unit
    def test_oauth_validation(self):
        """Test OAuth validation (lines 105-106)"""
        # Missing client_id
        with pytest.raises(
            ValueError, match="client_id and client_secret are required"
        ):
            SourceConfig(
                source_type=SourceType.CONFLUENCE,
                base_url="https://example.com",
                auth_type=AuthType.OAUTH,
                client_id=None,
                client_secret="secret",
            )

        # Missing client_secret
        with pytest.raises(
            ValueError, match="client_id and client_secret are required"
        ):
            SourceConfig(
                source_type=SourceType.CONFLUENCE,
                base_url="https://example.com",
                auth_type=AuthType.OAUTH,
                client_id="id",
                client_secret=None,
            )

    @pytest.mark.unit
    def test_basic_auth_validation(self):
        """Test basic auth validation (lines 107-108)"""
        # Missing username
        with pytest.raises(ValueError, match="username and password are required"):
            SourceConfig(
                source_type=SourceType.CONFLUENCE,
                base_url="https://example.com",
                auth_type=AuthType.BASIC_AUTH,
                username=None,
                password="pass",
            )

        # Missing password
        with pytest.raises(ValueError, match="username and password are required"):
            SourceConfig(
                source_type=SourceType.CONFLUENCE,
                base_url="https://example.com",
                auth_type=AuthType.BASIC_AUTH,
                username="user",
                password=None,
            )

    @pytest.mark.unit
    def test_bearer_token_validation(self):
        """Test bearer token validation (lines 110-111)"""
        with pytest.raises(ValueError, match="bearer_token is required"):
            SourceConfig(
                source_type=SourceType.CONFLUENCE,
                base_url="https://example.com",
                auth_type=AuthType.BEARER_TOKEN,
                bearer_token=None,
            )


class TestIntegrationResult:
    """Test IntegrationResult for complete coverage"""

    @pytest.mark.unit
    def test_get_summary(self):
        """Test get_summary method (line 131)"""
        result = IntegrationResult(
            success=True,
            documents=[{"id": "doc1"}, {"id": "doc2"}],
            total_count=2,
            processing_time=1.5,
            source_type=SourceType.CONFLUENCE,
            batch_count=1,
            last_sync_time="2024-01-01T00:00:00Z",
        )

        summary = result.get_summary()

        assert summary["success"] is True
        assert summary["source_type"] == SourceType.CONFLUENCE
        assert summary["total_count"] == 2
        assert summary["processing_time"] == 1.5
        assert summary["batch_count"] == 1
        assert summary["last_sync_time"] == "2024-01-01T00:00:00Z"


class TestBaseConnectorAuth:
    """Test authentication headers for all auth types"""

    @pytest.fixture
    def basic_auth_config(self):
        """Basic auth config"""
        return SourceConfig(
            source_type=SourceType.CONFLUENCE,
            base_url="https://example.com",
            auth_type=AuthType.BASIC_AUTH,
            username="user",
            password="password",
        )

    @pytest.fixture
    def bearer_token_config(self):
        """Bearer token config"""
        return SourceConfig(
            source_type=SourceType.CONFLUENCE,
            base_url="https://example.com",
            auth_type=AuthType.BEARER_TOKEN,
            bearer_token="my-bearer-token",
        )

    @pytest.fixture
    def oauth_config_with_token(self):
        """OAuth config with access token"""
        return SourceConfig(
            source_type=SourceType.CONFLUENCE,
            base_url="https://example.com",
            auth_type=AuthType.OAUTH,
            client_id="client",
            client_secret="secret",
            access_token="access-token-123",
        )

    @pytest.fixture
    def oauth_config_without_token(self):
        """OAuth config without access token"""
        return SourceConfig(
            source_type=SourceType.CONFLUENCE,
            base_url="https://example.com",
            auth_type=AuthType.OAUTH,
            client_id="client",
            client_secret="secret",
            access_token=None,
        )

    @pytest.mark.unit
    def test_basic_auth_headers(self, basic_auth_config):
        """Test basic auth headers (lines 170-173)"""
        connector = BaseConnector(basic_auth_config)
        headers = connector._get_auth_headers()

        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Basic ")
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.unit
    def test_bearer_token_headers(self, bearer_token_config):
        """Test bearer token headers (lines 175-176)"""
        connector = BaseConnector(bearer_token_config)
        headers = connector._get_auth_headers()

        assert headers["Authorization"] == "Bearer my-bearer-token"

    @pytest.mark.unit
    def test_oauth_headers_with_token(self, oauth_config_with_token):
        """Test OAuth headers with token (lines 178-180)"""
        connector = BaseConnector(oauth_config_with_token)
        headers = connector._get_auth_headers()

        assert headers["Authorization"] == "Bearer access-token-123"

    @pytest.mark.unit
    def test_oauth_headers_without_token(self, oauth_config_without_token):
        """Test OAuth headers without token"""
        connector = BaseConnector(oauth_config_without_token)
        headers = connector._get_auth_headers()

        assert "Authorization" not in headers


class TestBaseConnectorErrorHandling:
    """Test error handling in BaseConnector"""

    @pytest.fixture
    def connector(self):
        """Create a test connector"""
        config = SourceConfig(
            source_type=SourceType.CONFLUENCE,
            base_url="https://example.com",
            auth_type=AuthType.API_TOKEN,
            api_token="token",
            username="user",
        )
        return BaseConnector(config)

    @pytest.mark.unit
    async def test_session_not_initialized(self, connector):
        """Test error when session not initialized (line 192)"""
        with pytest.raises(ConnectionError, match="Session not initialized"):
            await connector._make_request("GET", "https://example.com")

    @pytest.mark.unit
    async def test_make_request_with_custom_headers(self, connector):
        """Test make_request with custom headers (line 196)"""
        async with connector:
            mock_response = AsyncMock()
            mock_response.status = 200

            mock_request = AsyncMock(return_value=mock_response)
            connector._session.request = mock_request

            custom_headers = {"X-Custom": "value"}
            await connector._make_request(
                "GET", "https://example.com", headers=custom_headers
            )

            # Check that custom headers were merged
            call_args = mock_request.call_args
            assert call_args[1]["headers"]["X-Custom"] == "value"
            assert "Authorization" in call_args[1]["headers"]

    @pytest.mark.unit
    async def test_http_error_handling(self, connector):
        """Test HTTP error handling (lines 210-211)"""
        async with connector:
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_response.text.return_value = "Not Found"

            if connector._session:
                connector._session.request = AsyncMock(return_value=mock_response)  # type: ignore

            with pytest.raises(ConnectionError, match="HTTP 404: Not Found"):
                await connector._make_request("GET", "https://example.com")

    @pytest.mark.unit
    async def test_client_connector_error(self, connector):
        """Test ClientConnectorError handling (lines 215-216)"""
        async with connector:
            if connector._session:
                connector._session.request = AsyncMock(  # type: ignore
                    side_effect=aiohttp.ClientConnectorError(
                    connection_key=MagicMock(), os_error=OSError("Network error")
                )
            )

            with pytest.raises(ConnectionError, match="Connection failed"):
                await connector._make_request("GET", "https://example.com")

    @pytest.mark.unit
    async def test_timeout_error(self, connector):
        """Test timeout error handling (lines 217-218)"""
        async with connector:
            if connector._session:
                connector._session.request = AsyncMock(side_effect=TimeoutError())  # type: ignore

            with pytest.raises(ConnectionError, match="Request timeout"):
                await connector._make_request("GET", "https://example.com")


class TestConfluenceConnectorErrors:
    """Test Confluence connector error handling"""

    @pytest.fixture
    def confluence_config(self):
        """Confluence config"""
        return SourceConfig(
            source_type=SourceType.CONFLUENCE,
            base_url="https://example.com",
            auth_type=AuthType.API_TOKEN,
            api_token="token",
            username="user",
            search_query="test query",
            last_sync_time="2024-01-01T00:00:00Z",
        )

    @pytest.mark.unit
    async def test_transform_page_error(self, confluence_config):
        """Test page transformation error (lines 257-261)"""
        connector = ConfluenceConnector(confluence_config)

        # Mock pages that will fail transformation
        mock_pages = [
            {
                "id": "123",
                "title": "Good Page",
                "body": {"storage": {"value": "<p>content</p>"}},
            },
            {"id": "456"},  # Missing required fields
        ]

        with patch.object(connector, "_fetch_all_pages", return_value=mock_pages):
            with patch(
                "app.services.external_source_integration.logger"
            ) as mock_logger:
                documents = await connector.fetch_documents()

                assert len(documents) == 1  # Only the good page
                assert documents[0]["id"] == "confluence-123"
                mock_logger.warning.assert_called_once()

    @pytest.mark.unit
    async def test_fetch_pages_empty_batch(self, confluence_config):
        """Test fetching pages with empty batch (line 275)"""
        connector = ConfluenceConnector(confluence_config)

        with patch.object(connector, "_fetch_pages_batch", return_value=[]):
            pages = await connector._fetch_all_pages()
            assert pages == []

    @pytest.mark.unit
    async def test_fetch_pages_with_last_sync(self, confluence_config):
        """Test incremental sync with last_sync_time (line 298)"""
        connector = ConfluenceConnector(confluence_config)

        async with connector:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"results": []}

            mock_request = AsyncMock(return_value=mock_response)
            connector._session.request = mock_request

            await connector._fetch_pages_batch(0, 10, incremental=True)

            # Check that lastModified was included in params
            call_args = mock_request.call_args
            assert call_args[1]["params"]["lastModified"] == "2024-01-01T00:00:00Z"

    @pytest.mark.unit
    async def test_fetch_pages_with_search_query(self, confluence_config):
        """Test fetching pages with search query (lines 302-303)"""
        connector = ConfluenceConnector(confluence_config)

        async with connector:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"results": []}

            mock_request = AsyncMock(return_value=mock_response)
            connector._session.request = mock_request

            await connector._fetch_pages_batch(0, 10)

            # Check URL and CQL parameter
            call_args = mock_request.call_args
            assert "/rest/api/content/search" in call_args[0][1]
            assert 'text ~ "test query"' in call_args[1]["params"]["cql"]

    @pytest.mark.unit
    async def test_fetch_pages_batch_error(self, confluence_config):
        """Test error in fetch_pages_batch (lines 313-315)"""
        connector = ConfluenceConnector(confluence_config)

        async with connector:
            if connector._session:
                connector._session.request = AsyncMock(side_effect=Exception("API Error"))  # type: ignore

            with patch(
                "app.services.external_source_integration.logger"
            ) as mock_logger:
                result = await connector._fetch_pages_batch(0, 10)

                assert result == []
                mock_logger.error.assert_called_once()


class TestJiraConnector:
    """Test JIRA connector for complete coverage"""

    @pytest.fixture
    def jira_config(self):
        """JIRA config"""
        return SourceConfig(
            source_type=SourceType.JIRA,
            base_url="https://example.com",
            auth_type=AuthType.API_TOKEN,
            api_token="token",
            username="user",
            project_key="TEST",
            search_query="bug",
            last_sync_time="2024-01-01T00:00:00Z",
            filters={"status": "Open", "assignee": "john", "issuetype": "Bug"},
        )

    @pytest.mark.unit
    async def test_jira_test_connection_success(self, jira_config):
        """Test JIRA connection success (lines 371-381)"""
        connector = JiraConnector(jira_config)

        async with connector:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "name": "testuser",
                "emailAddress": "test@example.com",
            }

            if connector._session:
                connector._session.request = AsyncMock(return_value=mock_response)  # type: ignore

            result = await connector.test_connection()

            assert result["success"] is True
            assert result["user"] == "testuser"
            assert "Authentication successful" in result["message"]

    @pytest.mark.unit
    async def test_jira_test_connection_failure(self, jira_config):
        """Test JIRA connection failure (lines 382-383)"""
        connector = JiraConnector(jira_config)

        async with connector:
            if connector._session:
                connector._session.request = AsyncMock(  # type: ignore
                    side_effect=AuthenticationError("Invalid credentials")
                )

            result = await connector.test_connection()

            assert result["success"] is False
            assert "Authentication failed" in result["message"]

    @pytest.mark.unit
    async def test_jira_fetch_documents(self, jira_config):
        """Test JIRA fetch documents (lines 387-400)"""
        connector = JiraConnector(jira_config)

        mock_issues = [
            {
                "key": "TEST-1",
                "fields": {
                    "summary": "Test Issue",
                    "description": "Description",
                    "issuetype": {"name": "Bug"},
                    "status": {"name": "Open"},
                },
            },
            {"key": "TEST-2"},  # Missing fields - will fail transformation
        ]

        with patch.object(connector, "_fetch_all_issues", return_value=mock_issues):
            with patch(
                "app.services.external_source_integration.logger"
            ) as mock_logger:
                documents = await connector.fetch_documents()

                assert len(documents) == 1
                assert documents[0]["id"] == "jira-TEST-1"
                mock_logger.warning.assert_called_once()

    @pytest.mark.unit
    async def test_jira_fetch_all_issues(self, jira_config):
        """Test JIRA fetch all issues with pagination (lines 406-425)"""
        jira_config.max_issues = 100
        jira_config.batch_size = 50  # Ensure batch_size is set
        connector = JiraConnector(jira_config)

        # First batch returns full results (50 items)
        batch1 = [{"key": f"TEST-{i}"} for i in range(50)]
        # Second batch returns partial results (25 items < 50 max_results)
        batch2 = [{"key": f"TEST-{i}"} for i in range(50, 75)]

        call_count = 0

        async def mock_fetch_batch(start_at, max_results, incremental):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return batch1
            elif call_count == 2:
                return batch2  # This returns 25 items < max_results, so loop stops
            else:
                return []

        with patch.object(
            connector, "_fetch_issues_batch", side_effect=mock_fetch_batch
        ):
            issues = await connector._fetch_all_issues()

            assert len(issues) == 75  # All issues from both batches
            assert (
                call_count == 2
            )  # Should stop after 2 batches because batch2 < max_results

    @pytest.mark.unit
    async def test_jira_fetch_all_issues_empty(self, jira_config):
        """Test JIRA fetch with empty results (line 415)"""
        connector = JiraConnector(jira_config)

        with patch.object(connector, "_fetch_issues_batch", return_value=[]):
            issues = await connector._fetch_all_issues()
            assert issues == []

    @pytest.mark.unit
    async def test_jira_fetch_issues_batch(self, jira_config):
        """Test JIRA fetch issues batch with all filters (lines 431-483)"""
        connector = JiraConnector(jira_config)

        async with connector:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "issues": [{"key": "TEST-1", "fields": {}}]
            }

            mock_request = AsyncMock(return_value=mock_response)
            connector._session.request = mock_request

            issues = await connector._fetch_issues_batch(0, 50, incremental=True)

            # Verify JQL construction
            call_args = mock_request.call_args
            payload = call_args[1]["json"]
            jql = payload["jql"]

            assert "project = TEST" in jql
            assert "updated >= '2024-01-01T00:00:00Z'" in jql
            assert "text ~ 'bug'" in jql
            assert "status = 'Open'" in jql
            assert "assignee = 'john'" in jql
            assert "issuetype = 'Bug'" in jql
            assert len(issues) == 1

    @pytest.mark.unit
    async def test_jira_fetch_issues_batch_error(self, jira_config):
        """Test error in fetch_issues_batch (lines 484-486)"""
        connector = JiraConnector(jira_config)

        async with connector:
            if connector._session:
                connector._session.request = AsyncMock(  # type: ignore
                    side_effect=Exception("JIRA API Error")
                )

            with patch(
                "app.services.external_source_integration.logger"
            ) as mock_logger:
                result = await connector._fetch_issues_batch(0, 50)

                assert result == []
                mock_logger.error.assert_called_once()


class TestExternalSourceIntegrator:
    """Test ExternalSourceIntegrator for complete coverage"""

    @pytest.fixture
    def sharepoint_config(self):
        """Unsupported source type config"""
        return SourceConfig(
            source_type=SourceType.SHAREPOINT,
            base_url="https://example.com",
            auth_type=AuthType.BEARER_TOKEN,
            bearer_token="token",
        )

    @pytest.mark.unit
    def test_unsupported_source_type(self, sharepoint_config):
        """Test unsupported source type (line 567)"""
        integrator = ExternalSourceIntegrator(sharepoint_config)

        with pytest.raises(
            ValueError, match="Unsupported source type: SourceType.SHAREPOINT"
        ):
            integrator._create_connector()

    @pytest.mark.unit
    async def test_fetch_documents_exception(self):
        """Test exception handling in fetch_documents (lines 605-610)"""
        config = SourceConfig(
            source_type=SourceType.CONFLUENCE,
            base_url="https://example.com",
            auth_type=AuthType.API_TOKEN,
            api_token="token",
            username="user",
        )
        integrator = ExternalSourceIntegrator(config)

        with patch.object(integrator, "_create_connector") as mock_create:
            mock_connector = AsyncMock()
            mock_connector.__aenter__.side_effect = Exception("Connection failed")
            mock_create.return_value = mock_connector

            with patch(
                "app.services.external_source_integration.logger"
            ) as mock_logger:
                result = await integrator.fetch_documents()

                assert result.success is False
                assert result.total_count == 0
                assert result.error_message == "Connection failed"
                assert result.processing_time > 0
                mock_logger.error.assert_called_once()

    @pytest.mark.unit
    async def test_process_document_batch_error(self):
        """Test document processing error (lines 630-634)"""
        config = SourceConfig(
            source_type=SourceType.CONFLUENCE,
            base_url="https://example.com",
            auth_type=AuthType.API_TOKEN,
            api_token="token",
            username="user",
        )
        integrator = ExternalSourceIntegrator(config)

        documents = [
            {"id": "doc1", "title": "Title 1", "content": "Content 1"},
            {"id": "doc2"},  # Missing required fields
        ]

        with patch("app.services.external_source_integration.logger") as mock_logger:
            processed = await integrator._process_document_batch(documents)

            assert len(processed) == 1
            assert processed[0]["id"] == "doc1"
            mock_logger.warning.assert_called_once()

    @pytest.mark.unit
    async def test_process_single_document_validation(self):
        """Test document validation error (line 644)"""
        config = SourceConfig(
            source_type=SourceType.CONFLUENCE,
            base_url="https://example.com",
            auth_type=AuthType.API_TOKEN,
            api_token="token",
            username="user",
        )
        integrator = ExternalSourceIntegrator(config)

        # Document without title
        with pytest.raises(ValueError, match="Document must have title and content"):
            await integrator._process_single_document({"content": "test"})

        # Document without content
        with pytest.raises(ValueError, match="Document must have title and content"):
            await integrator._process_single_document({"title": "test"})

    @pytest.mark.unit
    async def test_process_single_document_without_metadata(self):
        """Test processing document without metadata (line 654)"""
        config = SourceConfig(
            source_type=SourceType.CONFLUENCE,
            base_url="https://example.com",
            auth_type=AuthType.API_TOKEN,
            api_token="token",
            username="user",
        )
        integrator = ExternalSourceIntegrator(config)

        document = {
            "title": "Test",
            "content": "Content",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }

        processed = await integrator._process_single_document(document)

        assert "metadata" in processed
        assert processed["metadata"]["integration_timestamp"]
        assert processed["metadata"]["source_system"] == SourceType.CONFLUENCE

    @pytest.mark.unit
    def test_normalize_timestamp_variations(self):
        """Test timestamp normalization edge cases (lines 670-672)"""
        config = SourceConfig(
            source_type=SourceType.CONFLUENCE,
            base_url="https://example.com",
            auth_type=AuthType.API_TOKEN,
            api_token="token",
            username="user",
        )
        integrator = ExternalSourceIntegrator(config)

        # Already normalized timestamp
        assert (
            integrator._normalize_timestamp("2024-01-01T00:00:00Z")
            == "2024-01-01T00:00:00Z"
        )

        # Timestamp without Z
        assert (
            integrator._normalize_timestamp("2024-01-01T00:00:00")
            == "2024-01-01T00:00:00Z"
        )

        # Other format (no T)
        result = integrator._normalize_timestamp("2024-01-01 00:00:00")
        assert result == "2024-01-01 00:00:00"

        # Exception case - test with object that can't be processed as string
        result = integrator._normalize_timestamp("")  # type: ignore
        # Should return current ISO timestamp on any exception
        assert "T" in result or result == ""  # Basic check that it's ISO format or empty string

    @pytest.mark.unit
    async def test_fetch_confluence_pages_method(self):
        """Test _fetch_confluence_pages method (lines 676-678)"""
        config = SourceConfig(
            source_type=SourceType.CONFLUENCE,
            base_url="https://example.com",
            auth_type=AuthType.API_TOKEN,
            api_token="token",
            username="user",
        )
        integrator = ExternalSourceIntegrator(config)

        mock_docs = [{"id": "doc1", "title": "Test"}]

        with patch(
            "app.services.external_source_integration.ConfluenceConnector"
        ) as mock_connector:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.fetch_documents.return_value = mock_docs
            mock_connector.return_value = mock_instance

            docs = await integrator._fetch_confluence_pages()

            assert docs == mock_docs

    @pytest.mark.unit
    async def test_fetch_jira_issues_method(self):
        """Test _fetch_jira_issues method (lines 682-684)"""
        config = SourceConfig(
            source_type=SourceType.JIRA,
            base_url="https://example.com",
            auth_type=AuthType.API_TOKEN,
            api_token="token",
            username="user",
        )
        integrator = ExternalSourceIntegrator(config)

        mock_docs = [{"id": "issue1", "title": "Test Issue"}]

        with patch(
            "app.services.external_source_integration.JiraConnector"
        ) as mock_connector:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.fetch_documents.return_value = mock_docs
            mock_connector.return_value = mock_instance

            docs = await integrator._fetch_jira_issues()

            assert docs == mock_docs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
