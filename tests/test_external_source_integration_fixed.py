"""外部ソース統合テスト（修正版）

外部API（Confluence, JIRA）統合テスト - HTTPリクエストをモックで解決
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.services.external_source_integration import (
    ExternalSourceIntegrator,
    ExternalSourceType,
    SourceConfig,
)


class TestExternalSourceIntegrator:
    """外部ソース統合テスト"""

    @pytest.fixture
    def confluence_config(self):
        """Confluence設定"""
        return SourceConfig(
            source_type=ExternalSourceType.CONFLUENCE,
            base_url="https://confluence.example.com",
            authentication={
                "type": "bearer",
                "token": "test-token-123",
            },
            sync_config={
                "spaces": ["TEST"],
                "page_limit": 10,
            },
        )

    @pytest.fixture
    def jira_config(self):
        """JIRA設定"""
        return SourceConfig(
            source_type=ExternalSourceType.JIRA,
            base_url="https://jira.example.com",
            authentication={
                "type": "basic",
                "username": "test@example.com",
                "password": "test-password",
            },
            sync_config={
                "projects": ["TEST"],
                "issue_types": ["Task", "Bug"],
                "max_results": 50,
            },
        )

    @pytest.mark.asyncio
    async def test_authentication_api_token(
        self, confluence_config, mock_httpx_client, mock_environment_variables
    ):
        """APIトークン認証テスト"""
        integrator = ExternalSourceIntegrator(confluence_config)

        # Test authentication setup
        headers = integrator._get_auth_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-token-123"

    @pytest.mark.asyncio
    async def test_authentication_failure(
        self, confluence_config, mock_httpx_client, mock_environment_variables
    ):
        """認証失敗テスト"""
        # Mock 401 response
        mock_httpx_client.get.side_effect = AsyncMock(
            return_value=AsyncMock(
                status_code=401,
                text="Unauthorized",
                json=AsyncMock(
                    side_effect=ValueError("Not JSON")
                ),  # Non-JSON response
            )
        )

        integrator = ExternalSourceIntegrator(confluence_config)

        with pytest.raises(Exception) as exc_info:
            await integrator.test_connection()

        assert "401" in str(exc_info.value) or "Unauthorized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_confluence_fetch_pages(
        self, confluence_config, mock_httpx_client, mock_environment_variables
    ):
        """Confluenceページ取得テスト"""
        integrator = ExternalSourceIntegrator(confluence_config)

        # Fetch pages
        pages = await integrator.fetch_confluence_pages(space_key="TEST")

        assert len(pages) == 1
        assert pages[0]["id"] == "123456"
        assert pages[0]["title"] == "Test Page"
        assert "Test content" in pages[0]["body"]["storage"]["value"]

    @pytest.mark.asyncio
    async def test_jira_fetch_issues(
        self, jira_config, mock_httpx_client, mock_environment_variables
    ):
        """JIRA課題取得テスト"""
        integrator = ExternalSourceIntegrator(jira_config)

        # Fetch issues
        issues = await integrator.fetch_jira_issues(project="TEST")

        assert len(issues) == 1
        assert issues[0]["key"] == "TEST-1"
        assert issues[0]["fields"]["summary"] == "Test Issue"
        assert issues[0]["fields"]["status"]["name"] == "Open"

    @pytest.mark.asyncio
    async def test_pagination_handling(
        self, confluence_config, mock_httpx_client, mock_environment_variables
    ):
        """ページネーション処理テスト"""
        # Mock multiple pages
        page_responses = [
            {
                "results": [{"id": f"page{i}", "title": f"Page {i}"} for i in range(5)],
                "size": 5,
                "_links": {"next": "/rest/api/content?start=5"},
            },
            {
                "results": [
                    {"id": f"page{i}", "title": f"Page {i}"} for i in range(5, 8)
                ],
                "size": 3,
                "_links": {"next": None},
            },
        ]

        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            response_data = page_responses[min(call_count, len(page_responses) - 1)]
            call_count += 1
            return AsyncMock(
                status_code=200,
                json=AsyncMock(return_value=response_data),
            )

        mock_httpx_client.get.side_effect = mock_get

        integrator = ExternalSourceIntegrator(confluence_config)
        all_pages = []

        # Simulate pagination
        async for batch in integrator._paginate_results(
            "/rest/api/content", params={"spaceKey": "TEST"}
        ):
            all_pages.extend(batch)

        assert len(all_pages) == 8
        assert all_pages[0]["id"] == "page0"
        assert all_pages[7]["id"] == "page7"

    @pytest.mark.asyncio
    async def test_rate_limit_handling(
        self, confluence_config, mock_httpx_client, mock_environment_variables
    ):
        """レート制限処理テスト"""
        # Mock rate limit response followed by success
        responses = [
            AsyncMock(
                status_code=429,
                headers={"Retry-After": "1"},
                json=AsyncMock(return_value={"message": "Rate limit exceeded"}),
            ),
            AsyncMock(
                status_code=200,
                json=AsyncMock(
                    return_value={
                        "results": [{"id": "123", "title": "Success after retry"}],
                        "_links": {"next": None},
                    }
                ),
            ),
        ]

        mock_httpx_client.get.side_effect = responses

        integrator = ExternalSourceIntegrator(confluence_config)

        with patch("asyncio.sleep") as mock_sleep:
            pages = await integrator.fetch_confluence_pages(space_key="TEST")

            # Should have retried after rate limit
            mock_sleep.assert_called_once_with(1)
            assert len(pages) == 1
            assert pages[0]["title"] == "Success after retry"

    @pytest.mark.asyncio
    async def test_connection_error_handling(
        self, confluence_config, mock_httpx_client, mock_environment_variables
    ):
        """接続エラー処理テスト"""
        # Mock connection error
        mock_httpx_client.get.side_effect = Exception("Connection timeout")

        integrator = ExternalSourceIntegrator(confluence_config)

        with pytest.raises(Exception) as exc_info:
            await integrator.fetch_confluence_pages(space_key="TEST")

        assert "Connection timeout" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_incremental_sync(
        self, jira_config, mock_httpx_client, mock_environment_variables
    ):
        """増分同期テスト"""
        # Mock response with updated issues
        mock_httpx_client.post.return_value = AsyncMock(
            status_code=200,
            json=AsyncMock(
                return_value={
                    "issues": [
                        {
                            "id": "10001",
                            "key": "TEST-1",
                            "fields": {
                                "summary": "Updated Issue",
                                "updated": "2024-01-02T00:00:00.000+0000",
                            },
                        },
                        {
                            "id": "10002",
                            "key": "TEST-2",
                            "fields": {
                                "summary": "New Issue",
                                "updated": "2024-01-02T12:00:00.000+0000",
                            },
                        },
                    ],
                    "total": 2,
                }
            ),
        )

        integrator = ExternalSourceIntegrator(jira_config)

        # Perform incremental sync
        last_sync = datetime(2024, 1, 1)
        updated_issues = await integrator.sync_incremental(last_sync_time=last_sync)

        assert len(updated_issues) == 2
        assert updated_issues[0]["key"] == "TEST-1"
        assert updated_issues[1]["key"] == "TEST-2"

        # Verify JQL query includes updated date
        call_args = mock_httpx_client.post.call_args
        assert "jql" in call_args[1]["json"]

    @pytest.mark.asyncio
    async def test_filtering_and_search(
        self, jira_config, mock_httpx_client, mock_environment_variables
    ):
        """フィルタリングと検索テスト"""
        integrator = ExternalSourceIntegrator(jira_config)

        # Test with filters
        filters = {
            "project": "TEST",
            "issuetype": "Bug",
            "status": "Open",
        }

        issues = await integrator.search_issues(filters=filters)

        # Verify request was made with correct JQL
        call_args = mock_httpx_client.post.call_args
        jql = call_args[1]["json"]["jql"]
        assert "project" in jql
        assert "issuetype" in jql

    @pytest.mark.asyncio
    async def test_batch_processing(
        self, confluence_config, mock_httpx_client, mock_environment_variables
    ):
        """バッチ処理テスト"""
        # Mock multiple pages for batch processing
        pages = [
            {"id": f"page{i}", "title": f"Page {i}", "body": {"storage": {"value": f"Content {i}"}}}
            for i in range(20)
        ]

        mock_httpx_client.get.return_value = AsyncMock(
            status_code=200,
            json=AsyncMock(
                return_value={
                    "results": pages,
                    "_links": {"next": None},
                }
            ),
        )

        integrator = ExternalSourceIntegrator(confluence_config)

        # Process in batches
        processed_count = 0
        batch_size = 5

        all_pages = await integrator.fetch_confluence_pages(space_key="TEST")

        for i in range(0, len(all_pages), batch_size):
            batch = all_pages[i : i + batch_size]
            processed_count += len(batch)

            # Verify batch processing
            assert len(batch) <= batch_size

        assert processed_count == 20

    @pytest.mark.asyncio
    async def test_end_to_end_confluence_integration(
        self, confluence_config, mock_httpx_client, mock_environment_variables
    ):
        """Confluence統合エンドツーエンドテスト"""
        integrator = ExternalSourceIntegrator(confluence_config)

        # Test full sync flow
        sync_result = await integrator.sync_all()

        assert sync_result["success"] is True
        assert sync_result["source_type"] == "confluence"
        assert "total_fetched" in sync_result
        assert "sync_time" in sync_result
        assert sync_result["total_fetched"] > 0


class TestSourceConfig:
    """SourceConfig設定テスト"""

    def test_config_validation(self):
        """設定バリデーションテスト"""
        # Valid config
        config = SourceConfig(
            source_type=ExternalSourceType.CONFLUENCE,
            base_url="https://confluence.example.com",
            authentication={"type": "bearer", "token": "test-token"},
        )

        assert config.source_type == ExternalSourceType.CONFLUENCE
        assert config.base_url == "https://confluence.example.com"
        assert config.authentication["type"] == "bearer"

        # Test with sync config
        config_with_sync = SourceConfig(
            source_type=ExternalSourceType.JIRA,
            base_url="https://jira.example.com",
            authentication={"type": "basic", "username": "user", "password": "pass"},
            sync_config={
                "projects": ["PROJ1", "PROJ2"],
                "sync_interval": 3600,
            },
        )

        assert len(config_with_sync.sync_config["projects"]) == 2
        assert config_with_sync.sync_config["sync_interval"] == 3600


# Ensure fixtures from conftest_extended.py are available
pytest_plugins = ["conftest_extended"]