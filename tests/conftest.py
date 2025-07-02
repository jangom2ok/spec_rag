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
os.environ["MILVUS_HOST"] = "localhost"
os.environ["MILVUS_PORT"] = "19530"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def test_app() -> FastAPI:
    """テスト用アプリケーション"""
    from app.main import create_app

    app = create_app()

    # テスト用の認証情報を返すモック関数
    def mock_current_user():
        return {
            "sub": "test@example.com",
            "role": "admin",
            "permissions": ["read", "write", "admin"],
        }

    # 認証依存関係をオーバーライド
    from app.api.documents import get_current_user_or_api_key
    from app.api.auth import get_current_user
    app.dependency_overrides[get_current_user_or_api_key] = mock_current_user
    app.dependency_overrides[get_current_user] = mock_current_user

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

    # Milvusのモック
    milvus_patches = [
        patch("app.models.milvus.connections"),
        patch("app.models.milvus.utility"),
        patch("app.models.milvus.Collection"),
        patch("app.models.milvus.FieldSchema"),
        patch("app.models.milvus.CollectionSchema"),
        patch("app.models.milvus.DataType"),
    ]

    for milvus_patch in milvus_patches:
        patches.append(milvus_patch)
        mock_obj = milvus_patch.start()

        # 基本的なモック設定
        if hasattr(mock_obj, "connect"):
            mock_obj.connect.return_value = None
        if hasattr(mock_obj, "has_collection"):
            mock_obj.has_collection.return_value = False
        if hasattr(mock_obj, "create_collection"):
            mock_obj.create_collection.return_value = None

    mocks["milvus"] = mock_obj

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


# テストで使用するアプリケーション用の設定
from app.main import app
