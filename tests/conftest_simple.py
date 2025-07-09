"""Pytest設定とフィクスチャ"""

import os
import secrets
from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

# テスト環境用の環境変数設定
os.environ["TESTING"] = "true"
test_secret_key = os.getenv("TEST_SECRET_KEY")
if not test_secret_key:
    test_secret_key = secrets.token_urlsafe(32)
os.environ["SECRET_KEY"] = test_secret_key
os.environ["APERTUREDB_HOST"] = "localhost"
os.environ["APERTUREDB_PORT"] = "55555"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def test_app() -> FastAPI:
    """テスト用アプリケーション"""
    # 認証をバイパスしたテスト用アプリを作成
    with (
        patch("app.core.middleware.CombinedAuthenticationMiddleware") as mock_auth,
        patch("app.models.aperturedb.Client"),
    ):
        # 認証ミドルウェアをバイパス
        mock_auth_instance = mock_auth.return_value
        mock_auth_instance.authenticate.return_value = {
            "sub": "test@example.com",
            "role": "admin",
            "permissions": ["read", "write", "admin"],
        }

        from app.main import create_app

        app = create_app()
        return app


@pytest.fixture
async def async_client(test_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """非同期テストクライアント"""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture(autouse=True)
def mock_external_services() -> Generator[dict[str, Any], None, None]:
    """外部サービスのモック"""
    patches = []
    mocks = {}

    # ApertureDBのモック
    aperturedb_patch = patch("app.models.aperturedb.Client")
    patches.append(aperturedb_patch)
    mock_client = aperturedb_patch.start()
    
    # ApertureDBクライアントのモック設定
    mock_client_instance = AsyncMock()
    mock_client_instance.query.return_value = ([{"FindDescriptorSet": {"count": 0}}], [])
    mock_client.return_value = mock_client_instance

    mocks["aperturedb"] = mock_client_instance

    try:
        yield mocks
    finally:
        for patch_obj in patches:
            patch_obj.stop()


@pytest.fixture
def mock_database() -> Generator[AsyncMock, None, None]:
    """データベースのモック"""
    with patch("app.repositories.document_repository.DocumentRepository") as mock_repo:
        mock_instance = AsyncMock()
        mock_repo.return_value = mock_instance
        yield mock_instance
