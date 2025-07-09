"""外部ソース統合サービス

TDD実装：テストケースに基づいた外部システム統合機能
"""

import base64
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from urllib.parse import urljoin

import aiohttp

logger = logging.getLogger(__name__)


class SourceType(str, Enum):
    """外部ソースタイプ"""

    CONFLUENCE = "confluence"
    JIRA = "jira"
    SHAREPOINT = "sharepoint"
    NOTION = "notion"


class AuthType(str, Enum):
    """認証タイプ"""

    API_TOKEN = "api_token"
    OAUTH = "oauth"
    BASIC_AUTH = "basic_auth"
    BEARER_TOKEN = "bearer_token"


class ConnectionError(Exception):
    """接続エラー"""

    pass


class AuthenticationError(Exception):
    """認証エラー"""

    pass


class RateLimitError(Exception):
    """レート制限エラー"""

    pass


@dataclass
class SourceConfig:
    """外部ソース設定"""

    source_type: SourceType
    base_url: str
    auth_type: AuthType

    # API Token認証
    api_token: str | None = None
    username: str | None = None

    # OAuth認証
    client_id: str | None = None
    client_secret: str | None = None
    redirect_uri: str | None = None
    access_token: str | None = None
    refresh_token: str | None = None

    # Basic認証
    password: str | None = None

    # Bearer Token認証
    bearer_token: str | None = None

    # 取得設定
    project_key: str | None = None
    max_pages: int = 100
    max_issues: int = 100
    batch_size: int = 10
    timeout: int = 30

    # フィルタリング
    filters: dict[str, Any] = field(default_factory=dict)
    search_query: str | None = None
    last_sync_time: str | None = None

    def __post_init__(self):
        """設定値のバリデーション"""
        if not self.base_url:
            raise ValueError("base_url is required")

        if self.auth_type == AuthType.API_TOKEN:
            if not self.api_token or not self.username:
                raise ValueError(
                    "api_token and username are required for API token auth"
                )
        elif self.auth_type == AuthType.OAUTH:
            if not self.client_id or not self.client_secret:
                raise ValueError("client_id and client_secret are required for OAuth")
        elif self.auth_type == AuthType.BASIC_AUTH:
            if not self.username or not self.password:
                raise ValueError("username and password are required for basic auth")
        elif self.auth_type == AuthType.BEARER_TOKEN:
            if not self.bearer_token:
                raise ValueError("bearer_token is required for bearer token auth")


@dataclass
class IntegrationResult:
    """統合結果"""

    success: bool
    documents: list[dict[str, Any]]
    total_count: int
    processing_time: float
    source_type: SourceType
    batch_count: int = 1
    error_message: str | None = None
    last_sync_time: str | None = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    def get_summary(self) -> dict[str, Any]:
        """統合結果のサマリーを取得"""
        return {
            "success": self.success,
            "source_type": self.source_type,
            "total_count": self.total_count,
            "processing_time": self.processing_time,
            "batch_count": self.batch_count,
            "last_sync_time": self.last_sync_time,
        }


class BaseConnector:
    """外部ソースコネクタ基底クラス"""

    def __init__(self, config: SourceConfig):
        self.config = config
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        """非同期コンテキスト開始"""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキスト終了"""
        if self._session:
            await self._session.close()

    def _get_auth_headers(self) -> dict[str, str]:
        """認証ヘッダーを取得"""
        headers = {}

        if self.config.auth_type == AuthType.API_TOKEN:
            # Atlassian API Token認証
            credentials = f"{self.config.username}:{self.config.api_token}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded_credentials}"

        elif self.config.auth_type == AuthType.BASIC_AUTH:
            credentials = f"{self.config.username}:{self.config.password}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded_credentials}"

        elif self.config.auth_type == AuthType.BEARER_TOKEN:
            headers["Authorization"] = f"Bearer {self.config.bearer_token}"

        elif self.config.auth_type == AuthType.OAUTH:
            if self.config.access_token:
                headers["Authorization"] = f"Bearer {self.config.access_token}"

        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"

        return headers

    async def _make_request(
        self, method: str, url: str, **kwargs
    ) -> aiohttp.ClientResponse:
        """HTTPリクエストを実行"""
        if not self._session:
            raise ConnectionError("Session not initialized")

        headers = self._get_auth_headers()
        if "headers" in kwargs:
            headers.update(kwargs["headers"])
        kwargs["headers"] = headers

        try:
            response = await self._session.request(method, url, **kwargs)

            if response.status == 401:
                raise AuthenticationError("Authentication failed")
            elif response.status == 429:
                retry_after = response.headers.get("Retry-After", "60")
                raise RateLimitError(
                    f"Rate limit exceeded. Retry after {retry_after} seconds"
                )
            elif response.status >= 400:
                error_text = await response.text()
                raise ConnectionError(f"HTTP {response.status}: {error_text}")

            return response

        except aiohttp.ClientConnectorError as e:
            raise ConnectionError(f"Connection failed: {str(e)}") from e
        except TimeoutError as e:
            raise ConnectionError("Request timeout") from e

    async def test_connection(self) -> dict[str, Any]:
        """接続テスト（サブクラスで実装）"""
        raise NotImplementedError

    async def fetch_documents(self, incremental: bool = False) -> list[dict[str, Any]]:
        """ドキュメント取得（サブクラスで実装）"""
        raise NotImplementedError


class ConfluenceConnector(BaseConnector):
    """Confluenceコネクタ"""

    async def test_connection(self) -> dict[str, Any]:
        """Confluence接続テスト"""
        url = urljoin(self.config.base_url, "/rest/api/user/current")

        try:
            response = await self._make_request("GET", url)
            user_info = await response.json()

            return {
                "success": True,
                "message": "Authentication successful",
                "user": user_info.get("username", "unknown"),
            }
        except Exception as e:
            return {"success": False, "message": f"Authentication failed: {str(e)}"}

    async def fetch_documents(self, incremental: bool = False) -> list[dict[str, Any]]:
        """Confluenceページを取得"""
        pages = await self._fetch_all_pages(incremental)
        documents = []

        for page in pages:
            try:
                document = self._transform_page_to_document(page)
                documents.append(document)
            except Exception as e:
                logger.warning(
                    f"Failed to transform page {page.get('id', 'unknown')}: {e}"
                )
                continue

        return documents

    async def _fetch_all_pages(self, incremental: bool = False) -> list[dict[str, Any]]:
        """すべてのページを取得（ページネーション対応）"""
        all_pages = []
        start = 0
        limit = min(self.config.batch_size, 25)  # Confluence APIの制限

        while len(all_pages) < self.config.max_pages:
            pages_batch = await self._fetch_pages_batch(start, limit, incremental)

            if not pages_batch:
                break

            all_pages.extend(pages_batch)

            if len(pages_batch) < limit:
                break

            start += limit

        return all_pages[: self.config.max_pages]

    async def _fetch_pages_batch(
        self, start: int, limit: int, incremental: bool = False
    ) -> list[dict[str, Any]]:
        """ページのバッチを取得"""
        params = {
            "start": start,
            "limit": limit,
            "expand": "body.storage,space,version,metadata.properties",
        }

        # 増分同期
        if incremental and self.config.last_sync_time:
            params["lastModified"] = self.config.last_sync_time

        # 検索クエリ
        if self.config.search_query:
            url = urljoin(self.config.base_url, "/rest/api/content/search")
            params["cql"] = f'text ~ "{self.config.search_query}"'
        else:
            url = urljoin(self.config.base_url, "/rest/api/content")
            params["type"] = "page"

        try:
            response = await self._make_request("GET", url, params=params)
            data = await response.json()
            return data.get("results", [])
        except Exception as e:
            logger.error(f"Failed to fetch pages batch: {e}")
            return []

    def _transform_page_to_document(self, page: dict[str, Any]) -> dict[str, Any]:
        """Confluenceページをドキュメント形式に変換"""
        page_id = page["id"]
        title = page["title"]

        # HTMLコンテンツからプレーンテキストを抽出
        html_content = page.get("body", {}).get("storage", {}).get("value", "")
        plain_content = self._html_to_plain_text(html_content)

        # メタデータの構築
        metadata = {
            "page_url": urljoin(
                self.config.base_url, page.get("_links", {}).get("webui", "")
            ),
            "version": page.get("version", {}).get("number"),
            "last_modified": page.get("version", {}).get("when"),
        }

        # カスタムプロパティの追加
        properties = page.get("metadata", {}).get("properties", {})
        if properties:
            metadata["custom_properties"] = properties

        return {
            "id": f"confluence-{page_id}",
            "title": title,
            "content": plain_content,
            "source_type": "confluence",
            "source_id": page_id,
            "metadata": metadata,
            "created_at": page.get("version", {}).get("when"),
            "updated_at": page.get("version", {}).get("when"),
        }

    def _html_to_plain_text(self, html: str) -> str:
        """HTMLをプレーンテキストに変換"""
        # 簡易的なHTML除去
        # より高度な変換が必要な場合はBeautifulSoupなどを使用
        text = re.sub(
            r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(
            r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


class JiraConnector(BaseConnector):
    """JIRAコネクタ"""

    async def test_connection(self) -> dict[str, Any]:
        """JIRA接続テスト"""
        url = urljoin(self.config.base_url, "/rest/api/2/myself")

        try:
            response = await self._make_request("GET", url)
            user_info = await response.json()

            return {
                "success": True,
                "message": "Authentication successful",
                "user": user_info.get("name", "unknown"),
            }
        except Exception as e:
            return {"success": False, "message": f"Authentication failed: {str(e)}"}

    async def fetch_documents(self, incremental: bool = False) -> list[dict[str, Any]]:
        """JIRA課題を取得"""
        issues = await self._fetch_all_issues(incremental)
        documents = []

        for issue in issues:
            try:
                document = self._transform_issue_to_document(issue)
                documents.append(document)
            except Exception as e:
                logger.warning(
                    f"Failed to transform issue {issue.get('key', 'unknown')}: {e}"
                )
                continue

        return documents

    async def _fetch_all_issues(
        self, incremental: bool = False
    ) -> list[dict[str, Any]]:
        """すべての課題を取得（ページネーション対応）"""
        all_issues = []
        start_at = 0
        max_results = min(self.config.batch_size, 50)  # JIRA APIの制限

        while len(all_issues) < self.config.max_issues:
            issues_batch = await self._fetch_issues_batch(
                start_at, max_results, incremental
            )

            if not issues_batch:
                break

            all_issues.extend(issues_batch)

            if len(issues_batch) < max_results:
                break

            start_at += max_results

        return all_issues[: self.config.max_issues]

    async def _fetch_issues_batch(
        self, start_at: int, max_results: int, incremental: bool = False
    ) -> list[dict[str, Any]]:
        """課題のバッチを取得"""
        jql_parts = []

        # プロジェクトフィルタ
        if self.config.project_key:
            jql_parts.append(f"project = {self.config.project_key}")

        # 増分同期
        if incremental and self.config.last_sync_time:
            jql_parts.append(f"updated >= '{self.config.last_sync_time}'")

        # テキスト検索
        if self.config.search_query:
            jql_parts.append(f"text ~ '{self.config.search_query}'")

        # フィルタの適用
        for key, value in self.config.filters.items():
            if key == "status":
                jql_parts.append(f"status = '{value}'")
            elif key == "assignee":
                jql_parts.append(f"assignee = '{value}'")
            elif key == "issuetype":
                jql_parts.append(f"issuetype = '{value}'")

        jql = " AND ".join(jql_parts) if jql_parts else "order by created DESC"

        payload = {
            "jql": jql,
            "startAt": start_at,
            "maxResults": max_results,
            "fields": [
                "summary",
                "description",
                "issuetype",
                "status",
                "priority",
                "project",
                "assignee",
                "reporter",
                "created",
                "updated",
                "labels",
                "components",
                "fixVersions",
            ],
        }

        url = urljoin(self.config.base_url, "/rest/api/2/search")

        try:
            response = await self._make_request("POST", url, json=payload)
            data = await response.json()
            return data.get("issues", [])
        except Exception as e:
            logger.error(f"Failed to fetch issues batch: {e}")
            return []

    def _transform_issue_to_document(self, issue: dict[str, Any]) -> dict[str, Any]:
        """JIRA課題をドキュメント形式に変換"""
        issue_key = issue["key"]
        fields = issue["fields"]

        title = f"{issue_key}: {fields.get('summary', '')}"

        # コンテンツの構築
        content_parts = []
        content_parts.append(f"Issue Key: {issue_key}")
        content_parts.append(f"Summary: {fields.get('summary', '')}")

        if fields.get("description"):
            content_parts.append(f"Description: {fields['description']}")

        content_parts.append(
            f"Issue Type: {fields.get('issuetype', {}).get('name', 'Unknown')}"
        )
        content_parts.append(
            f"Status: {fields.get('status', {}).get('name', 'Unknown')}"
        )

        if fields.get("priority"):
            content_parts.append(
                f"Priority: {fields['priority'].get('name', 'Unknown')}"
            )

        if fields.get("assignee"):
            content_parts.append(
                f"Assignee: {fields['assignee'].get('displayName', 'Unknown')}"
            )

        if fields.get("reporter"):
            content_parts.append(
                f"Reporter: {fields['reporter'].get('displayName', 'Unknown')}"
            )

        content = "\n".join(content_parts)

        # メタデータの構築
        metadata = {
            "issue_type": fields.get("issuetype", {}).get("name"),
            "status": fields.get("status", {}).get("name"),
            "priority": fields.get("priority", {}).get("name"),
            "project_key": fields.get("project", {}).get("key"),
            "project_name": fields.get("project", {}).get("name"),
            "assignee": fields.get("assignee", {}).get("displayName"),
            "reporter": fields.get("reporter", {}).get("displayName"),
            "labels": list(fields.get("labels", [])),
            "components": [comp.get("name") for comp in fields.get("components", [])],
            "fix_versions": [ver.get("name") for ver in fields.get("fixVersions", [])],
        }

        return {
            "id": f"jira-{issue_key}",
            "title": title,
            "content": content,
            "source_type": "jira",
            "source_id": issue_key,
            "metadata": metadata,
            "created_at": fields.get("created"),
            "updated_at": fields.get("updated"),
        }


class ExternalSourceIntegrator:
    """外部ソース統合器メインクラス"""

    def __init__(self, config: SourceConfig):
        self.config = config
        self._connector: BaseConnector | None = None

    def _create_connector(self) -> BaseConnector:
        """ソースタイプに応じたコネクタを作成"""
        if self.config.source_type == SourceType.CONFLUENCE:
            return ConfluenceConnector(self.config)
        elif self.config.source_type == SourceType.JIRA:
            return JiraConnector(self.config)
        else:
            raise ValueError(f"Unsupported source type: {self.config.source_type}")

    async def test_connection(self) -> dict[str, Any]:
        """接続テスト"""
        async with self._create_connector() as connector:
            return await connector.test_connection()

    async def fetch_documents(self, incremental: bool = False) -> IntegrationResult:
        """ドキュメントを取得"""
        start_time = datetime.now()

        try:
            async with self._create_connector() as connector:
                documents = await connector.fetch_documents(incremental)

                # バッチ処理
                processed_documents = []
                batch_count = 0

                for i in range(0, len(documents), self.config.batch_size):
                    batch = documents[i : i + self.config.batch_size]
                    processed_batch = await self._process_document_batch(batch)
                    processed_documents.extend(processed_batch)
                    batch_count += 1

                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()

                return IntegrationResult(
                    success=True,
                    documents=processed_documents,
                    total_count=len(processed_documents),
                    processing_time=processing_time,
                    source_type=self.config.source_type,
                    batch_count=batch_count,
                )

        except Exception as e:
            logger.error(f"Document fetch failed: {e}")
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            return IntegrationResult(
                success=False,
                documents=[],
                total_count=0,
                processing_time=processing_time,
                source_type=self.config.source_type,
                error_message=str(e),
            )

    async def _process_document_batch(
        self, documents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """ドキュメントバッチを処理"""
        processed = []

        for doc in documents:
            try:
                # 追加的な処理（バリデーション、正規化など）
                processed_doc = await self._process_single_document(doc)
                processed.append(processed_doc)
            except Exception as e:
                logger.warning(
                    f"Failed to process document {doc.get('id', 'unknown')}: {e}"
                )
                continue

        return processed

    async def _process_single_document(
        self, document: dict[str, Any]
    ) -> dict[str, Any]:
        """単一ドキュメントを処理"""
        # 基本的なバリデーション
        if not document.get("title") or not document.get("content"):
            raise ValueError("Document must have title and content")

        # タイムスタンプの正規化
        if document.get("created_at"):
            document["created_at"] = self._normalize_timestamp(document["created_at"])
        if document.get("updated_at"):
            document["updated_at"] = self._normalize_timestamp(document["updated_at"])

        # 統合メタデータの追加
        if "metadata" not in document:
            document["metadata"] = {}

        document["metadata"]["integration_timestamp"] = datetime.now().isoformat()
        document["metadata"]["source_system"] = self.config.source_type

        return document

    def _normalize_timestamp(self, timestamp: str) -> str:
        """タイムスタンプを正規化"""
        # 簡易的な実装
        try:
            if "T" in timestamp and "Z" in timestamp:
                return timestamp
            elif "T" in timestamp:
                return timestamp + "Z"
            else:
                return timestamp
        except Exception:
            return datetime.now().isoformat()

    async def _fetch_confluence_pages(self) -> list[dict[str, Any]]:
        """Confluenceページを取得（テスト用のメソッド）"""
        async with ConfluenceConnector(self.config) as connector:
            return await connector.fetch_documents()

    async def _fetch_jira_issues(self) -> list[dict[str, Any]]:
        """JIRA課題を取得（テスト用のメソッド）"""
        async with JiraConnector(self.config) as connector:
            return await connector.fetch_documents()
