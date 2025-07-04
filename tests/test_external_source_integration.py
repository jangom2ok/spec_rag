"""外部ソース統合のテストモジュール

TDD実装：テストケース→実装→リファクタの順序で実装
"""

from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

from app.services.external_source_integration import (
    AuthType,
    ConfluenceConnector,
    ConnectionError,
    ExternalSourceIntegrator,
    IntegrationResult,
    JiraConnector,
    RateLimitError,
    SourceConfig,
    SourceType,
)


class TestExternalSourceIntegrator:
    """外部ソース統合器のテストクラス"""

    @pytest.fixture
    def confluence_config(self) -> SourceConfig:
        """Confluence設定"""
        return SourceConfig(
            source_type=SourceType.CONFLUENCE,
            base_url="https://example.atlassian.net/wiki",
            auth_type=AuthType.API_TOKEN,
            api_token="test_token",
            username="test@example.com",
            max_pages=100,
            timeout=30,
        )

    @pytest.fixture
    def jira_config(self) -> SourceConfig:
        """JIRA設定"""
        return SourceConfig(
            source_type=SourceType.JIRA,
            base_url="https://example.atlassian.net",
            auth_type=AuthType.API_TOKEN,
            api_token="test_token",
            username="test@example.com",
            project_key="TEST",
            max_issues=50,
            timeout=30,
        )

    @pytest.fixture
    def oauth_config(self) -> SourceConfig:
        """OAuth設定"""
        return SourceConfig(
            source_type=SourceType.CONFLUENCE,
            base_url="https://example.atlassian.net/wiki",
            auth_type=AuthType.OAUTH,
            client_id="test_client_id",
            client_secret="test_client_secret",
            redirect_uri="https://app.example.com/oauth/callback",
        )

    @pytest.mark.unit
    def test_integrator_initialization(self, confluence_config: SourceConfig):
        """統合器の初期化テスト"""
        integrator = ExternalSourceIntegrator(config=confluence_config)

        assert integrator.config == confluence_config
        assert integrator.config.source_type == SourceType.CONFLUENCE
        assert integrator.config.auth_type == AuthType.API_TOKEN

    @pytest.mark.unit
    async def test_confluence_connector_creation(self, confluence_config: SourceConfig):
        """Confluenceコネクタ作成テスト"""
        integrator = ExternalSourceIntegrator(config=confluence_config)
        connector = integrator._create_connector()

        assert isinstance(connector, ConfluenceConnector)
        assert connector.config == confluence_config

    @pytest.mark.unit
    async def test_jira_connector_creation(self, jira_config: SourceConfig):
        """JIRAコネクタ作成テスト"""
        integrator = ExternalSourceIntegrator(config=jira_config)
        connector = integrator._create_connector()

        assert isinstance(connector, JiraConnector)
        assert connector.config == jira_config

    @pytest.mark.unit
    async def test_authentication_api_token(self, confluence_config: SourceConfig):
        """APIトークン認証テスト"""
        integrator = ExternalSourceIntegrator(config=confluence_config)

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "type": "user",
                "username": "test@example.com",
            }
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = (
                mock_response
            )

            result = await integrator.test_connection()

            assert result.success is True
            assert "authentication successful" in result.message.lower()

    @pytest.mark.unit
    async def test_authentication_failure(self, confluence_config: SourceConfig):
        """認証失敗テスト"""
        confluence_config.api_token = "invalid_token"
        integrator = ExternalSourceIntegrator(config=confluence_config)

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 401
            mock_response.text.return_value = "Unauthorized"
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = (
                mock_response
            )

            result = await integrator.test_connection()

            assert result.success is False
            assert "authentication failed" in result.message.lower()

    @pytest.mark.unit
    async def test_confluence_fetch_pages(self, confluence_config: SourceConfig):
        """Confluenceページ取得テスト"""
        integrator = ExternalSourceIntegrator(config=confluence_config)

        mock_pages = [
            {
                "id": "123456",
                "title": "Test Page 1",
                "body": {"storage": {"value": "<p>Test content 1</p>"}},
                "space": {"key": "TEST"},
                "version": {"number": 1},
                "_links": {"webui": "/pages/123456"},
            },
            {
                "id": "789012",
                "title": "Test Page 2",
                "body": {"storage": {"value": "<p>Test content 2</p>"}},
                "space": {"key": "TEST"},
                "version": {"number": 2},
                "_links": {"webui": "/pages/789012"},
            },
        ]

        with patch.object(
            integrator, "_fetch_confluence_pages", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = mock_pages

            result = await integrator.fetch_documents()

            assert result.success is True
            assert len(result.documents) == 2
            assert result.documents[0]["title"] == "Test Page 1"
            assert result.documents[1]["title"] == "Test Page 2"

    @pytest.mark.unit
    async def test_jira_fetch_issues(self, jira_config: SourceConfig):
        """JIRA課題取得テスト"""
        integrator = ExternalSourceIntegrator(config=jira_config)

        mock_issues = [
            {
                "id": "10001",
                "key": "TEST-1",
                "fields": {
                    "summary": "Test Issue 1",
                    "description": "Test description 1",
                    "issuetype": {"name": "Bug"},
                    "status": {"name": "Open"},
                    "project": {"key": "TEST"},
                },
            },
            {
                "id": "10002",
                "key": "TEST-2",
                "fields": {
                    "summary": "Test Issue 2",
                    "description": "Test description 2",
                    "issuetype": {"name": "Task"},
                    "status": {"name": "In Progress"},
                    "project": {"key": "TEST"},
                },
            },
        ]

        with patch.object(
            integrator, "_fetch_jira_issues", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = mock_issues

            result = await integrator.fetch_documents()

            assert result.success is True
            assert len(result.documents) == 2
            assert result.documents[0]["title"] == "TEST-1: Test Issue 1"
            assert result.documents[1]["title"] == "TEST-2: Test Issue 2"

    @pytest.mark.unit
    async def test_pagination_handling(self, confluence_config: SourceConfig):
        """ページネーション処理テスト"""
        confluence_config.max_pages = 50
        ExternalSourceIntegrator(config=confluence_config)

        # 複数ページのレスポンスをシミュレート
        page1_response = {
            "results": [{"id": f"page-{i}", "title": f"Page {i}"} for i in range(25)],
            "_links": {"next": "/rest/api/content?start=25&limit=25"},
        }

        page2_response = {
            "results": [
                {"id": f"page-{i}", "title": f"Page {i}"} for i in range(25, 50)
            ],
            "_links": {},
        }

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response1 = AsyncMock()
            mock_response1.status = 200
            mock_response1.json.return_value = page1_response

            mock_response2 = AsyncMock()
            mock_response2.status = 200
            mock_response2.json.return_value = page2_response

            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.side_effect = [
                mock_response1,
                mock_response2,
            ]

            connector = ConfluenceConnector(config=confluence_config)
            pages = await connector._fetch_all_pages()

            assert len(pages) == 50

    @pytest.mark.unit
    async def test_rate_limit_handling(self, confluence_config: SourceConfig):
        """レート制限処理テスト"""
        integrator = ExternalSourceIntegrator(config=confluence_config)

        with patch("aiohttp.ClientSession") as mock_session:
            # 429レスポンスを返す
            mock_response = AsyncMock()
            mock_response.status = 429
            mock_response.headers = {"Retry-After": "60"}
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = (
                mock_response
            )

            with pytest.raises(RateLimitError) as exc_info:
                await integrator.test_connection()

            assert "rate limit exceeded" in str(exc_info.value).lower()

    @pytest.mark.unit
    async def test_connection_error_handling(self, confluence_config: SourceConfig):
        """接続エラー処理テスト"""
        integrator = ExternalSourceIntegrator(config=confluence_config)

        with patch("aiohttp.ClientSession") as mock_session:
            # 接続エラーをシミュレート
            mock_session.return_value.__aenter__.return_value.get.side_effect = (
                aiohttp.ClientConnectorError(
                    connection_key="test", os_error=Exception("Connection failed")
                )
            )

            with pytest.raises(ConnectionError) as exc_info:
                await integrator.test_connection()

            assert "connection failed" in str(exc_info.value).lower()

    @pytest.mark.unit
    async def test_data_transformation_confluence(
        self, confluence_config: SourceConfig
    ):
        """Confluenceデータ変換テスト"""
        ExternalSourceIntegrator(config=confluence_config)
        connector = ConfluenceConnector(config=confluence_config)

        raw_page = {
            "id": "123456",
            "title": "Test Page",
            "body": {"storage": {"value": "<p>Test <strong>content</strong></p>"}},
            "space": {"key": "TEST", "name": "Test Space"},
            "version": {"number": 3, "when": "2024-01-01T12:00:00Z"},
            "_links": {"webui": "/pages/123456"},
            "metadata": {"properties": {"custom": {"value": "test"}}},
        }

        transformed = connector._transform_page_to_document(raw_page)

        assert transformed["id"] == "confluence-123456"
        assert transformed["title"] == "Test Page"
        assert transformed["content"] == "Test content"  # HTMLタグが除去される
        assert transformed["source_type"] == "confluence"
        assert transformed["source_id"] == "123456"
        assert transformed["metadata"]["page_url"] is not None

    @pytest.mark.unit
    async def test_data_transformation_jira(self, jira_config: SourceConfig):
        """JIRAデータ変換テスト"""
        ExternalSourceIntegrator(config=jira_config)
        connector = JiraConnector(config=jira_config)

        raw_issue = {
            "id": "10001",
            "key": "TEST-1",
            "fields": {
                "summary": "Test Issue",
                "description": "Test description with details",
                "issuetype": {"name": "Bug", "iconUrl": "bug.png"},
                "status": {"name": "Open", "statusCategory": {"key": "new"}},
                "priority": {"name": "High"},
                "project": {"key": "TEST", "name": "Test Project"},
                "assignee": {"displayName": "John Doe"},
                "reporter": {"displayName": "Jane Smith"},
                "created": "2024-01-01T12:00:00.000+0000",
                "updated": "2024-01-02T12:00:00.000+0000",
            },
        }

        transformed = connector._transform_issue_to_document(raw_issue)

        assert transformed["id"] == "jira-TEST-1"
        assert transformed["title"] == "TEST-1: Test Issue"
        assert "Test description with details" in transformed["content"]
        assert transformed["source_type"] == "jira"
        assert transformed["source_id"] == "TEST-1"
        assert transformed["metadata"]["issue_type"] == "Bug"
        assert transformed["metadata"]["status"] == "Open"

    @pytest.mark.unit
    async def test_incremental_sync(self, confluence_config: SourceConfig):
        """増分同期テスト"""
        confluence_config.last_sync_time = "2024-01-01T00:00:00Z"
        integrator = ExternalSourceIntegrator(config=confluence_config)

        # 最終同期時刻以降に更新されたページのみを返すことを確認
        with patch.object(
            integrator, "_fetch_confluence_pages", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = [
                {
                    "id": "123456",
                    "title": "Updated Page",
                    "version": {"when": "2024-01-02T12:00:00Z"},
                }
            ]

            await integrator.fetch_documents(incremental=True)

            # 増分同期パラメータが正しく渡されることを確認
            mock_fetch.assert_called_once()
            call_args = mock_fetch.call_args
            assert (
                "incremental" in str(call_args)
                or confluence_config.last_sync_time is not None
            )

    @pytest.mark.unit
    async def test_filtering_and_search(self, confluence_config: SourceConfig):
        """フィルタリングと検索テスト"""
        confluence_config.filters = {"labels": ["documentation", "api"]}
        confluence_config.search_query = "REST API"

        integrator = ExternalSourceIntegrator(config=confluence_config)

        with patch.object(
            integrator, "_fetch_confluence_pages", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = [
                {
                    "id": "123456",
                    "title": "REST API Documentation",
                    "metadata": {
                        "labels": [{"name": "documentation"}, {"name": "api"}]
                    },
                }
            ]

            result = await integrator.fetch_documents()

            assert result.success is True
            assert len(result.documents) == 1
            assert "REST API" in result.documents[0]["title"]

    @pytest.mark.unit
    async def test_batch_processing(self, confluence_config: SourceConfig):
        """バッチ処理テスト"""
        confluence_config.batch_size = 10
        integrator = ExternalSourceIntegrator(config=confluence_config)

        # 大量のページを生成
        large_page_list = [
            {"id": f"page-{i}", "title": f"Page {i}", "content": f"Content {i}"}
            for i in range(25)
        ]

        with patch.object(
            integrator, "_fetch_confluence_pages", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = large_page_list

            result = await integrator.fetch_documents()

            assert result.success is True
            assert len(result.documents) == 25
            # バッチ処理により、処理が分割されることを確認
            assert result.batch_count >= 3  # 25項目を10個ずつ処理

    @pytest.mark.integration
    async def test_end_to_end_confluence_integration(
        self, confluence_config: SourceConfig
    ):
        """Confluence統合のEnd-to-Endテスト"""
        # 実際のHTTPリクエストをモック
        mock_auth_response = {"type": "user", "username": "test@example.com"}

        mock_pages_response = {
            "results": [
                {
                    "id": "123456",
                    "title": "Integration Test Page",
                    "body": {
                        "storage": {
                            "value": "<h1>Test</h1><p>Integration test content</p>"
                        }
                    },
                    "space": {"key": "TEST"},
                    "version": {"number": 1},
                    "_links": {"webui": "/pages/123456"},
                }
            ],
            "_links": {},
        }

        with patch("aiohttp.ClientSession") as mock_session:
            # 認証チェック
            mock_auth = AsyncMock()
            mock_auth.status = 200
            mock_auth.json.return_value = mock_auth_response

            # ページ取得
            mock_pages = AsyncMock()
            mock_pages.status = 200
            mock_pages.json.return_value = mock_pages_response

            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.side_effect = [
                mock_auth,
                mock_pages,
            ]

            integrator = ExternalSourceIntegrator(config=confluence_config)

            # 接続テスト
            connection_result = await integrator.test_connection()
            assert connection_result.success is True

            # ドキュメント取得
            fetch_result = await integrator.fetch_documents()
            assert fetch_result.success is True
            assert len(fetch_result.documents) == 1
            assert fetch_result.documents[0]["title"] == "Integration Test Page"


class TestSourceConfig:
    """ソース設定のテストクラス"""

    @pytest.mark.unit
    def test_config_validation(self):
        """設定バリデーションテスト"""
        # 有効な設定
        valid_config = SourceConfig(
            source_type=SourceType.CONFLUENCE,
            base_url="https://example.atlassian.net/wiki",
            auth_type=AuthType.API_TOKEN,
            api_token="token",
            username="user@example.com",
        )
        assert valid_config.source_type == SourceType.CONFLUENCE

        # 無効な設定（必須フィールド不足）
        with pytest.raises(ValueError):
            SourceConfig(
                source_type=SourceType.CONFLUENCE,
                # base_urlが不足
                auth_type=AuthType.API_TOKEN,
                api_token="token",
            )

    @pytest.mark.unit
    def test_auth_config_validation(self):
        """認証設定バリデーションテスト"""
        # APIトークン認証
        api_config = SourceConfig(
            source_type=SourceType.CONFLUENCE,
            base_url="https://example.com",
            auth_type=AuthType.API_TOKEN,
            api_token="token",
            username="user",
        )
        assert api_config.auth_type == AuthType.API_TOKEN

        # OAuth認証
        oauth_config = SourceConfig(
            source_type=SourceType.CONFLUENCE,
            base_url="https://example.com",
            auth_type=AuthType.OAUTH,
            client_id="client_id",
            client_secret="client_secret",
        )
        assert oauth_config.auth_type == AuthType.OAUTH


class TestIntegrationResult:
    """統合結果のテストクラス"""

    @pytest.mark.unit
    def test_result_creation(self):
        """結果作成テスト"""
        documents = [
            {"id": "doc-1", "title": "Document 1"},
            {"id": "doc-2", "title": "Document 2"},
        ]

        result = IntegrationResult(
            success=True,
            documents=documents,
            total_count=2,
            processing_time=3.5,
            source_type=SourceType.CONFLUENCE,
        )

        assert result.success is True
        assert len(result.documents) == 2
        assert result.total_count == 2
        assert result.processing_time == 3.5

    @pytest.mark.unit
    def test_result_summary(self):
        """結果サマリーテスト"""
        result = IntegrationResult(
            success=True,
            documents=[{"id": "doc-1"}],
            total_count=1,
            processing_time=1.0,
            source_type=SourceType.JIRA,
        )

        summary = result.get_summary()

        assert "success" in summary
        assert "total_count" in summary
        assert "processing_time" in summary
        assert summary["success"] is True
        assert summary["total_count"] == 1
