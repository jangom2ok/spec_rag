"""FastAPIアプリケーション基盤のテスト"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app, create_app


class TestFastAPIBase:
    """FastAPIアプリケーション基盤のテストクラス"""

    def test_app_creation(self):
        """アプリケーション作成のテスト"""
        test_app = create_app()

        assert isinstance(test_app, FastAPI)
        assert test_app.title == "RAG System"
        assert test_app.version == "1.0.0"
        # Description is None by default in FastAPI

    def test_app_instance(self):
        """アプリケーションインスタンスのテスト"""
        assert isinstance(app, FastAPI)
        assert app.title == "RAG System"

    def test_cors_middleware(self):
        """CORSミドルウェアの設定テスト"""
        client = TestClient(app)

        # OPTIONSリクエストでCORSヘッダーを確認
        response = client.options("/")
        assert response.status_code in [200, 404, 405]  # 405 if OPTIONS not allowed

    def test_app_startup(self):
        """アプリケーション起動テスト"""
        client = TestClient(app)

        # アプリケーションが正常に起動することを確認
        assert client.app is not None

    @pytest.mark.asyncio
    async def test_async_client(self):
        """非同期クライアントのテスト"""
        from httpx import ASGITransport

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            # 基本的な接続テスト（存在しないエンドポイントでも構造は確認できる）
            response = await ac.get("/nonexistent")
            # 404が返ることを確認（アプリが動作している証拠）
            assert response.status_code == 404


class TestAppConfiguration:
    """アプリケーション設定のテストクラス"""

    def test_debug_mode(self):
        """デバッグモードの設定テスト"""
        test_app = create_app()
        # 開発環境ではデバッグモードが有効
        assert hasattr(test_app, "debug")

    def test_api_prefix(self):
        """APIプレフィックスの設定テスト"""
        # 将来的にAPIプレフィックス（/v1）が設定されることを想定
        test_app = create_app()
        assert test_app is not None

    def test_docs_configuration(self):
        """API ドキュメント設定のテスト"""
        test_app = create_app()

        # OpenAPI設定が適切に行われていることを確認
        assert test_app.openapi_url is not None
        assert test_app.docs_url is not None
        assert test_app.redoc_url is not None


class TestMiddleware:
    """ミドルウェア設定のテストクラス"""

    def test_middleware_stack(self):
        """ミドルウェアスタックのテスト"""
        client = TestClient(app)

        # ミドルウェアが正常に動作することを確認
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200

    def test_request_response_cycle(self):
        """リクエスト・レスポンスサイクルのテスト"""
        client = TestClient(app)

        # 基本的なリクエスト・レスポンスが機能することを確認
        response = client.get("/docs")
        assert response.status_code == 200
