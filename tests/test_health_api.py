"""ヘルスチェックAPIのテスト"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app


class TestHealthAPI:
    """ヘルスチェックAPIのテストクラス"""

    def test_basic_health_check(self):
        """基本的なヘルスチェックのテスト"""
        client = TestClient(app)

        response = client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_health_check_response_format(self):
        """ヘルスチェックレスポンス形式のテスト"""
        client = TestClient(app)

        response = client.get("/v1/health")
        data = response.json()

        # 必須フィールドの存在確認
        required_fields = ["status", "timestamp", "version", "environment"]
        for field in required_fields:
            assert field in data

        # ステータスの値確認
        assert data["status"] in ["healthy", "unhealthy", "degraded"]

    @pytest.mark.asyncio
    async def test_async_health_check(self):
        """非同期ヘルスチェックのテスト"""
        from httpx import ASGITransport

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/v1/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    def test_health_check_headers(self):
        """ヘルスチェックのHTTPヘッダーテスト"""
        client = TestClient(app)

        response = client.get("/v1/health")

        assert response.status_code == 200
        assert "content-type" in response.headers
        assert response.headers["content-type"] == "application/json"


class TestDetailedHealthCheck:
    """詳細ヘルスチェックのテストクラス"""

    def test_detailed_health_check(self):
        """詳細ヘルスチェックのテスト"""
        client = TestClient(app)

        response = client.get("/v1/health/detailed")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "services" in data
        assert "system" in data

    def test_database_health_check(self):
        """データベースヘルスチェックのテスト"""
        client = TestClient(app)

        response = client.get("/v1/health/detailed")
        data = response.json()

        # データベース関連のヘルスチェック
        services = data.get("services", {})
        assert "postgresql" in services
        assert "aperturedb" in services

    @patch("app.api.health.check_postgresql_connection")
    async def test_postgresql_connection_check(self, mock_pg_check):
        """PostgreSQL接続チェックのテスト"""
        mock_pg_check.return_value = {"status": "healthy", "response_time": 0.05}

        client = TestClient(app)
        response = client.get("/v1/health/detailed")

        assert response.status_code == 200
        data = response.json()

        postgresql_status = data["services"]["postgresql"]
        assert postgresql_status["status"] == "healthy"
        assert "response_time" in postgresql_status

    @patch("app.api.health.check_aperturedb_connection")
    async def test_aperturedb_connection_check(self, mock_aperturedb_check):
        """ApertureDB接続チェックのテスト"""
        mock_aperturedb_check.return_value = {"status": "healthy", "descriptor_sets": 2}

        client = TestClient(app)
        response = client.get("/v1/health/detailed")

        assert response.status_code == 200
        data = response.json()

        aperturedb_status = data["services"]["aperturedb"]
        assert aperturedb_status["status"] == "healthy"
        assert "descriptor_sets" in aperturedb_status


class TestHealthCheckErrorScenarios:
    """ヘルスチェックエラーシナリオのテストクラス"""

    @patch("app.api.health.check_postgresql_connection")
    def test_postgresql_connection_failure(self, mock_pg_check):
        """PostgreSQL接続失敗時のテスト"""
        mock_pg_check.side_effect = Exception("Connection failed")

        client = TestClient(app)
        response = client.get("/v1/health/detailed")

        assert (
            response.status_code == 200
        )  # ヘルスチェックAPIは200を返すが、ステータスで異常を示す
        data = response.json()

        assert data["status"] in ["unhealthy", "degraded"]
        postgresql_status = data["services"]["postgresql"]
        assert postgresql_status["status"] == "unhealthy"

    @patch("app.api.health.check_aperturedb_connection")
    def test_aperturedb_connection_failure(self, mock_aperturedb_check):
        """ApertureDB接続失敗時のテスト"""
        mock_aperturedb_check.side_effect = Exception("ApertureDB connection failed")

        client = TestClient(app)
        response = client.get("/v1/health/detailed")

        assert response.status_code == 200
        data = response.json()

        aperturedb_status = data["services"]["aperturedb"]
        assert aperturedb_status["status"] == "unhealthy"
        assert "error" in aperturedb_status

    def test_invalid_health_endpoint(self):
        """無効なヘルスチェックエンドポイントのテスト"""
        client = TestClient(app)

        response = client.get("/v1/health/invalid")
        assert response.status_code == 404


class TestSystemHealth:
    """システムヘルスチェックのテストクラス"""

    def test_system_metrics(self):
        """システムメトリクスのテスト"""
        client = TestClient(app)

        response = client.get("/v1/health/detailed")
        data = response.json()

        system = data.get("system", {})
        expected_metrics = ["cpu_usage", "memory_usage", "disk_usage", "uptime"]

        for metric in expected_metrics:
            assert metric in system

    def test_readiness_probe(self):
        """Readiness Probeのテスト"""
        client = TestClient(app)

        response = client.get("/v1/health/ready")

        assert response.status_code in [200, 503]  # 準備完了または未完了

        if response.status_code == 200:
            data = response.json()
            assert data["ready"] is True

    def test_liveness_probe(self):
        """Liveness Probeのテスト"""
        client = TestClient(app)

        response = client.get("/v1/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["alive"] is True
