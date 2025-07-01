"""エラーハンドリングのテスト"""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock
from sqlalchemy.exc import SQLAlchemyError
from pymilvus.exceptions import MilvusException

from app.main import app
from app.core.exceptions import (
    RAGSystemException,
    DatabaseException,
    VectorDatabaseException,
    ValidationException,
    AuthenticationException,
    AuthorizationException
)


class TestCustomExceptions:
    """カスタム例外のテストクラス"""

    def test_rag_system_exception(self):
        """RAGシステム基底例外のテスト"""
        exception = RAGSystemException("Test error", error_code="TEST_001")

        assert str(exception) == "Test error"
        assert exception.error_code == "TEST_001"
        assert exception.status_code == 500

    def test_database_exception(self):
        """データベース例外のテスト"""
        exception = DatabaseException("Database connection failed")

        assert str(exception) == "Database connection failed"
        assert exception.status_code == 500
        assert "database" in exception.error_code.lower()

    def test_vector_database_exception(self):
        """ベクトルデータベース例外のテスト"""
        exception = VectorDatabaseException("Milvus connection failed")

        assert str(exception) == "Milvus connection failed"
        assert exception.status_code == 500

    def test_validation_exception(self):
        """バリデーション例外のテスト"""
        exception = ValidationException("Invalid input data")

        assert str(exception) == "Invalid input data"
        assert exception.status_code == 422

    def test_authentication_exception(self):
        """認証例外のテスト"""
        exception = AuthenticationException("Invalid credentials")

        assert str(exception) == "Invalid credentials"
        assert exception.status_code == 401

    def test_authorization_exception(self):
        """認可例外のテスト"""
        exception = AuthorizationException("Access denied")

        assert str(exception) == "Access denied"
        assert exception.status_code == 403


class TestErrorHandlers:
    """エラーハンドラーのテストクラス"""

    def test_404_error_handler(self):
        """404エラーハンドラーのテスト"""
        client = TestClient(app)

        response = client.get("/nonexistent-endpoint")

        assert response.status_code == 404
        data = response.json()

        assert "error" in data
        assert data["error"]["code"] == "NOT_FOUND"
        assert "message" in data["error"]
        assert "timestamp" in data

    def test_405_error_handler(self):
        """405エラーハンドラーのテスト"""
        client = TestClient(app)

        # 存在するエンドポイントに対して無効なHTTPメソッドを使用
        response = client.post("/v1/health")  # GETのみ許可されているエンドポイント

        assert response.status_code == 405
        data = response.json()

        assert "error" in data
        assert data["error"]["code"] == "METHOD_NOT_ALLOWED"

    def test_422_validation_error_handler(self):
        """422バリデーションエラーハンドラーのテスト"""
        client = TestClient(app)

        # 無効なJSONデータを送信
        response = client.post(
            "/v1/documents",
            json={"invalid": "data"},
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422
        data = response.json()

        assert "error" in data
        assert data["error"]["code"] == "VALIDATION_ERROR"
        assert "details" in data["error"]

    def test_500_internal_server_error_handler(self):
        """500内部サーバーエラーハンドラーのテスト"""
        with patch('app.api.health.get_health_status') as mock_health:
            mock_health.side_effect = Exception("Internal server error")

            client = TestClient(app)
            response = client.get("/v1/health")

            assert response.status_code == 500
            data = response.json()

            assert "error" in data
            assert data["error"]["code"] == "INTERNAL_SERVER_ERROR"
            assert "request_id" in data


class TestDatabaseErrorHandling:
    """データベースエラーハンドリングのテストクラス"""

    @patch('app.repositories.document_repository.DocumentRepository.get_by_id')
    def test_database_connection_error(self, mock_get):
        """データベース接続エラーのテスト"""
        mock_get.side_effect = SQLAlchemyError("Connection lost")

        client = TestClient(app)
        response = client.get("/v1/documents/test-id")

        assert response.status_code == 500
        data = response.json()

        assert data["error"]["code"] == "DATABASE_ERROR"
        assert "database" in data["error"]["message"].lower()

    @patch('app.models.milvus.DenseVectorCollection.search')
    def test_vector_database_error(self, mock_search):
        """ベクトルデータベースエラーのテスト"""
        mock_search.side_effect = MilvusException("Milvus connection failed")

        client = TestClient(app)
        response = client.post(
            "/v1/search",
            json={"query": "test query", "top_k": 10}
        )

        assert response.status_code == 500
        data = response.json()

        assert data["error"]["code"] == "VECTOR_DATABASE_ERROR"

    @patch('app.repositories.document_repository.DocumentRepository.create')
    def test_database_constraint_violation(self, mock_create):
        """データベース制約違反エラーのテスト"""
        from sqlalchemy.exc import IntegrityError
        mock_create.side_effect = IntegrityError("", "", "")

        client = TestClient(app)
        response = client.post(
            "/v1/documents",
            json={
                "title": "Test Document",
                "content": "Test content",
                "source_type": "test"
            }
        )

        assert response.status_code == 409
        data = response.json()

        assert data["error"]["code"] == "CONSTRAINT_VIOLATION"


class TestValidationErrorHandling:
    """バリデーションエラーハンドリングのテストクラス"""

    def test_missing_required_fields(self):
        """必須フィールド不足のテスト"""
        client = TestClient(app)

        response = client.post("/v1/documents", json={})

        assert response.status_code == 422
        data = response.json()

        assert data["error"]["code"] == "VALIDATION_ERROR"
        assert "details" in data["error"]

    def test_invalid_field_types(self):
        """無効なフィールドタイプのテスト"""
        client = TestClient(app)

        response = client.post(
            "/v1/documents",
            json={
                "title": 123,  # 文字列が期待されるが数値
                "content": "Test content",
                "source_type": "test"
            }
        )

        assert response.status_code == 422
        data = response.json()

        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_invalid_enum_values(self):
        """無効なEnum値のテスト"""
        client = TestClient(app)

        response = client.post(
            "/v1/documents",
            json={
                "title": "Test Document",
                "content": "Test content",
                "source_type": "invalid_type"  # 無効なsource_type
            }
        )

        assert response.status_code == 422
        data = response.json()

        assert data["error"]["code"] == "VALIDATION_ERROR"


class TestAsyncErrorHandling:
    """非同期エラーハンドリングのテストクラス"""

    @pytest.mark.asyncio
    async def test_async_database_error(self):
        """非同期データベースエラーのテスト"""
        with patch('app.repositories.document_repository.DocumentRepository.get_by_id') as mock_get:
            mock_get.side_effect = SQLAlchemyError("Async connection error")

            async with AsyncClient(app=app, base_url="http://test") as ac:
                response = await ac.get("/v1/documents/test-id")

                assert response.status_code == 500
                data = response.json()
                assert data["error"]["code"] == "DATABASE_ERROR"

    @pytest.mark.asyncio
    async def test_async_timeout_error(self):
        """非同期タイムアウトエラーのテスト"""
        import asyncio

        with patch('app.api.health.check_postgresql_connection') as mock_check:
            mock_check.side_effect = asyncio.TimeoutError("Connection timeout")

            async with AsyncClient(app=app, base_url="http://test") as ac:
                response = await ac.get("/v1/health/detailed")

                assert response.status_code == 500
                data = response.json()
                assert "timeout" in data["error"]["message"].lower()


class TestErrorResponseFormat:
    """エラーレスポンス形式のテストクラス"""

    def test_error_response_structure(self):
        """エラーレスポンス構造のテスト"""
        client = TestClient(app)

        response = client.get("/nonexistent")
        data = response.json()

        # 必須フィールドの確認
        assert "error" in data
        assert "timestamp" in data
        assert "request_id" in data

        # errorオブジェクトの構造確認
        error = data["error"]
        assert "code" in error
        assert "message" in error
        assert "type" in error

    def test_error_response_consistency(self):
        """エラーレスポンスの一貫性テスト"""
        client = TestClient(app)

        # 異なるエラータイプで同じ構造が返されることを確認
        responses = [
            client.get("/nonexistent"),  # 404
            client.post("/v1/health"),   # 405
        ]

        for response in responses:
            data = response.json()

            # 共通フィールドの存在確認
            assert "error" in data
            assert "timestamp" in data
            assert "request_id" in data

            # エラーオブジェクトの一貫性確認
            error = data["error"]
            assert isinstance(error["code"], str)
            assert isinstance(error["message"], str)
            assert isinstance(error["type"], str)
