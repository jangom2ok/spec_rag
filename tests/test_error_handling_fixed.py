"""エラーハンドリングのテスト"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from pymilvus.exceptions import MilvusException
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    DatabaseError,
    RAGSystemError,
    ValidationError,
    VectorDatabaseError,
)
from app.main import app


class TestCustomExceptions:
    """カスタム例外のテストクラス"""

    def test_rag_system_exception(self):
        """RAGシステム基底例外のテスト"""
        exception = RAGSystemError("Test error", error_code="TEST_001")

        assert str(exception) == "Test error"
        assert exception.error_code == "TEST_001"
        assert exception.status_code == 500

    def test_database_exception(self):
        """データベース例外のテスト"""
        exception = DatabaseError("Database connection failed")

        assert str(exception) == "Database connection failed"
        assert exception.status_code == 500
        assert "database" in exception.error_code.lower()

    def test_vector_database_exception(self):
        """ベクトルデータベース例外のテスト"""
        exception = VectorDatabaseError("Milvus connection failed")

        assert str(exception) == "Milvus connection failed"
        assert exception.status_code == 500

    def test_validation_exception(self):
        """バリデーション例外のテスト"""
        exception = ValidationError("Invalid input data")

        assert str(exception) == "Invalid input data"
        assert exception.status_code == 422

    def test_authentication_exception(self):
        """認証例外のテスト"""
        exception = AuthenticationError("Invalid credentials")

        assert str(exception) == "Invalid credentials"
        assert exception.status_code == 401

    def test_authorization_exception(self):
        """認可例外のテスト"""
        exception = AuthorizationError("Access denied")

        assert str(exception) == "Access denied"
        assert exception.status_code == 403


@pytest.mark.no_auth_middleware
class TestErrorHandlers:
    """エラーハンドラーのテストクラス"""

    def test_422_validation_error_handler(self):
        """422バリデーションエラーハンドラーのテスト"""
        client = TestClient(app)

        # 無効なJSONデータを送信
        response = client.post(
            "/v1/documents",
            json={"invalid": "data"},
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422
        assert "detail" in response.json()

    def test_500_internal_server_error_handler(self):
        """500内部サーバーエラーハンドラーのテスト"""
        with patch("app.api.health.get_health_status") as mock_health:
            mock_health.side_effect = Exception("Internal server error")

            client = TestClient(app)
            response = client.get("/v1/health")

            assert response.status_code == 500
            assert "error" in response.json()


@pytest.mark.no_auth_middleware
class TestDatabaseErrorHandling:
    """データベースエラーハンドリングのテストクラス"""

    @patch("app.repositories.document_repository.DocumentRepository.get_by_id")
    def test_database_connection_error(self, mock_get):
        """データベース接続エラーのテスト"""
        mock_get.side_effect = SQLAlchemyError("Connection lost")

        client = TestClient(app)
        response = client.get("/v1/documents/test-id")

        assert response.status_code == 500
        assert "error" in response.json()

    @patch("app.models.milvus.DenseVectorCollection.search")
    def test_vector_database_error(self, mock_search):
        """ベクトルデータベースエラーのテスト"""
        # MilvusExceptionの正しい使用方法
        mock_search.side_effect = MilvusException(code=1, message="Milvus connection failed")

        client = TestClient(app)
        response = client.post("/v1/search", json={"query": "test query", "top_k": 10})

        assert response.status_code == 500
        assert "error" in response.json()

    @patch("app.repositories.document_repository.DocumentRepository.create")
    def test_database_constraint_violation(self, mock_create):
        """データベース制約違反エラーのテスト"""
        # IntegrityErrorの正しい使用方法
        mock_create.side_effect = IntegrityError(
            "UNIQUE constraint failed: documents.title",
            params=None,
            orig=Exception("UNIQUE constraint failed")
        )

        client = TestClient(app)
        response = client.post(
            "/v1/documents",
            json={
                "title": "Test Document",
                "content": "Test content",
                "source_type": "test",
            },
        )

        assert response.status_code == 409
        assert "error" in response.json()


@pytest.mark.no_auth_middleware
class TestValidationErrorHandling:
    """バリデーションエラーハンドリングのテストクラス"""

    def test_missing_required_fields(self):
        """必須フィールド不足のテスト"""
        client = TestClient(app)

        response = client.post("/v1/documents", json={})

        assert response.status_code == 422
        assert "detail" in response.json()
        # フィールドエラーの詳細を確認
        assert len(response.json()["detail"]) > 0

    def test_invalid_field_types(self):
        """無効なフィールドタイプのテスト"""
        client = TestClient(app)

        response = client.post(
            "/v1/documents",
            json={
                "title": 123,  # 文字列が期待されるが数値
                "content": "Test content",
                "source_type": "test",
            },
        )

        assert response.status_code == 422
        assert "detail" in response.json()

    def test_invalid_enum_values(self):
        """無効なEnum値のテスト"""
        client = TestClient(app)

        response = client.post(
            "/v1/documents",
            json={
                "title": "Test Document",
                "content": "Test content",
                "source_type": "invalid_type",  # 無効なsource_type
            },
        )

        assert response.status_code == 422
        assert "detail" in response.json()


@pytest.mark.no_auth_middleware
class TestAsyncErrorHandling:
    """非同期エラーハンドリングのテストクラス"""

    @pytest.mark.asyncio
    async def test_async_database_error(self):
        """非同期データベースエラーのテスト"""
        with patch(
            "app.repositories.document_repository.DocumentRepository.get_by_id"
        ) as mock_get:
            mock_get.side_effect = SQLAlchemyError("Async connection error")

            from httpx import ASGITransport

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                response = await ac.get("/v1/documents/test-id")

                assert response.status_code == 500
                assert "error" in response.json()

    @pytest.mark.asyncio
    async def test_async_timeout_error(self):
        """非同期タイムアウトエラーのテスト"""

        with patch("app.api.health.check_postgresql_connection") as mock_check:
            mock_check.side_effect = TimeoutError("Connection timeout")

            from httpx import ASGITransport

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                response = await ac.get("/v1/health/detailed")

                assert response.status_code == 500
                assert "error" in response.json()
