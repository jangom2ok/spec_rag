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

# Import extended fixtures
try:
    from fixtures_extended import *  # noqa: F403, F401
except ImportError:
    pass  # Extended fixtures not available


@pytest.fixture(scope="session")
def test_app() -> FastAPI:
    """テスト用アプリケーション"""
    from app.main import create_app

    app = create_app()
    return app


@pytest.fixture(autouse=True)
def setup_auth_overrides(request, test_app):
    """認証のオーバーライドを設定"""
    # no_auth_middlewareマーカーがある場合、認証をバイパス
    if "no_auth_middleware" in request.keywords:
        # テスト用の認証情報を返すモック関数
        def mock_current_user():
            return {
                "sub": "test@example.com",
                "role": "admin",
                "permissions": ["read", "write", "admin"],
            }

        # 認証依存関係をオーバーライド
        from app.api.auth import get_current_user
        from app.api.documents import get_current_user_or_api_key

        test_app.dependency_overrides[get_current_user_or_api_key] = mock_current_user
        test_app.dependency_overrides[get_current_user] = mock_current_user

        yield

        # クリーンアップ
        test_app.dependency_overrides.clear()
    else:
        yield


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
    mock_client_instance.query.return_value = (
        [{"FindDescriptorSet": {"count": 0}}],
        [],
    )
    mock_client.return_value = mock_client_instance

    mocks["aperturedb"] = mock_client_instance

    # FlagModelのモック
    import numpy as np

    class MockFlagModel:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, sentences, **kwargs):
            if isinstance(sentences, str):
                sentences = [sentences]
            
            # 返り値の形式を指定
            return_dense = kwargs.get("return_dense", True)
            return_sparse = kwargs.get("return_sparse", False)
            return_colbert_vecs = kwargs.get("return_colbert_vecs", False)
            
            results = {}
            
            if return_dense:
                results["dense_vecs"] = np.random.rand(len(sentences), 1024).astype(np.float32)
            
            if return_sparse:
                results["lexical_weights"] = [
                    {i: np.random.rand() for i in range(0, 1000, 100)} for _ in sentences
                ]
            
            if return_colbert_vecs:
                results["colbert_vecs"] = [
                    np.random.rand(10, 1024).astype(np.float32) for _ in sentences
                ]
            
            return results

    flag_model_patch = patch("app.services.embedding_service.FlagModel", MockFlagModel)
    patches.append(flag_model_patch)
    flag_model_patch.start()

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
