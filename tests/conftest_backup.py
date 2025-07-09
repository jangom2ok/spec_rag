"""Pytest設定とフィクスチャ"""

import os
import secrets
from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, Request, Response
from httpx import ASGITransport, AsyncClient

# テスト環境用の環境変数設定
os.environ["TESTING"] = "true"
# テスト環境用のランダムシークレットキーを生成（セキュリティ警告を回避）
test_secret_key = os.getenv("TEST_SECRET_KEY")
if not test_secret_key:
    # テスト用の安全なランダムキーを生成
    test_secret_key = secrets.token_urlsafe(32)
os.environ["SECRET_KEY"] = test_secret_key
os.environ["APERTUREDB_HOST"] = "localhost"
os.environ["APERTUREDB_PORT"] = "55555"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def test_app() -> FastAPI:
    """テスト用アプリケーションのフィクスチャ"""
    from app.main import create_app

    app = create_app()
    return app


@pytest.fixture
async def async_client(test_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """非同期テストクライアントのフィクスチャ"""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture(autouse=True)
def mock_external_services(request: Any) -> Generator[dict[str, Any], None, None]:
    """外部サービスのモック（全テストで自動適用、選択的無効化可能）"""

    # モック無効化の設定
    disable_jwt_mock = hasattr(request, "node") and "no_jwt_mock" in [
        marker.name for marker in request.node.iter_markers()
    ]
    disable_apikey_mock = hasattr(request, "node") and "no_apikey_mock" in [
        marker.name for marker in request.node.iter_markers()
    ]
    disable_middleware = hasattr(request, "node") and "no_auth_middleware" in [
        marker.name for marker in request.node.iter_markers()
    ]

    patches: list[Any] = []
    mocks: dict[str, Any] = {}

    # ApertureDB（常にモック）
    aperturedb_patch = patch("app.models.aperturedb.Client")
    patches.append(aperturedb_patch)
    mock_client = aperturedb_patch.start()
    
    # ApertureDBクライアントのモック設定
    mock_client_instance = AsyncMock()
    mock_client_instance.query.return_value = ([{"FindDescriptorSet": {"count": 0}}], [])
    mock_client.return_value = mock_client_instance

    # 主要なApertureDBオブジェクトをmocksに保存
    mocks["aperturedb"] = {
    }

    # 認証ミドルウェア（条件付きモック）
    if disable_middleware:
        # 認証ミドルウェアを無効化
        middleware_patch = patch("app.main.app.middleware")
        patches.append(middleware_patch)
        mock_middleware = middleware_patch.start()
        mock_middleware.return_value = None
        mocks["auth_middleware"] = mock_middleware
    else:
        # デフォルトでは認証をバイパス（テスト用）
        # CombinedAuthenticationMiddlewareクラスをモック
        auth_patch = patch("app.core.middleware.CombinedAuthenticationMiddleware")
        patches.append(auth_patch)
        mock_auth_class = auth_patch.start()

        # モックインスタンスを作成
        mock_auth_instance = mock_auth_class.return_value
        mock_auth_instance.authenticate.return_value = {
            "sub": "test@example.com",
            "role": "admin",
            "permissions": ["read", "write", "admin"],
        }
        mocks["auth_middleware"] = mock_auth_instance

    # JWT認証（条件付きモック）
    if not disable_jwt_mock:
        jwt_patch = patch("app.core.auth.verify_token")
        patches.append(jwt_patch)
        mock_verify = jwt_patch.start()
        mock_verify.return_value = {
            "sub": "test@example.com",
            "role": "admin",
            "permissions": ["read", "write", "admin"],
        }
        mocks["verify_token"] = mock_verify

    # API Key認証（条件付きモック）
    if not disable_apikey_mock:
        apikey_patch = patch("app.core.auth.validate_api_key")
        patches.append(apikey_patch)
        mock_api_key = apikey_patch.start()

        def mock_validate_api_key(key: str) -> Any:
            if key and key.startswith("ak_"):
                return {
                    "user_id": "test-user",
                    "permissions": ["read", "write"],
                    "key_id": "test-key-id",
                }
            return None

        mock_api_key.side_effect = mock_validate_api_key
        mocks["api_key"] = mock_api_key

    try:
        yield mocks
    finally:
        for patch_obj in patches:
            patch_obj.stop()


@pytest.fixture
def no_auth_mock():
    """認証モックを無効化するフィクスチャ（JWTテスト用）"""
    # このフィクスチャが使われた場合、認証関連のモックを停止

    yield


@pytest.fixture
def disable_auth() -> Generator[None, None, None]:
    """認証を無効化するフィクスチャ"""
    with patch("app.core.middleware.auth_middleware") as mock_middleware:

        async def mock_auth_middleware(request: Request, call_next: Any) -> Response:
            return await call_next(request)

        mock_middleware.side_effect = mock_auth_middleware
        yield


@pytest.fixture
def mock_database() -> Generator[AsyncMock, None, None]:
    """データベースのモック"""
    with patch("app.repositories.document_repository.DocumentRepository") as mock_repo:
        mock_instance = AsyncMock()
        mock_repo.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_aperturedb() -> Generator[dict[str, AsyncMock], None, None]:
    """ApertureDBのモック"""
    with (
        patch("app.models.aperturedb.DenseVectorCollection") as mock_dense,
        patch("app.models.aperturedb.SparseVectorCollection") as mock_sparse,
    ):
        mock_dense_instance = AsyncMock()
        mock_sparse_instance = AsyncMock()

        mock_dense.return_value = mock_dense_instance
        mock_sparse.return_value = mock_sparse_instance

        yield {"dense": mock_dense_instance, "sparse": mock_sparse_instance}
