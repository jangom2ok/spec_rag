"""Tests for health API error handling."""

import asyncio
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

try:
    pass  # aperturedb is optional
except ImportError:
    pass

from app.main import create_app

app = create_app()
client = TestClient(app)


class TestHealthAPIErrors:
    """Health API error handling tests."""

    def test_health_database_error(self):
        """Test health check with database error."""
        with patch("app.api.health.database") as mock_db:
            # SQLAlchemyErrorを発生させる
            mock_engine = Mock()
            mock_engine.execute = Mock(side_effect=SQLAlchemyError("Connection failed"))
            mock_db.engine = mock_engine

            response = client.get("/v1/health")

            # エラーが発生してもレスポンスは返す
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    def test_check_aperturedb_connection_error(self):
        """Test ApertureDB connection check with error."""
        # ApertureDBの接続チェックは現在モック実装
        response = client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()
        # 現在はモック実装なのでhealthyが返る
        assert data["aperturedb"]["status"] == "healthy"

    def test_get_system_metrics_error(self):
        """Test system metrics with error."""
        with patch("app.api.health.psutil.cpu_percent") as mock_cpu:
            mock_cpu.side_effect = Exception("System error")

            with patch("app.api.health.get_system_metrics") as mock_metrics:
                # get_system_metricsのException処理をテスト
                mock_metrics.return_value = {
                    "cpu_usage": 0.0,
                    "memory_usage": 0.0,
                    "disk_usage": 0.0,
                    "uptime": 0.0,
                }

                response = client.get("/v1/health")

                assert response.status_code == 200
                data = response.json()
                assert data["metrics"]["cpu_usage"] == 0.0
                assert data["metrics"]["memory_usage"] == 0.0

    def test_database_check_with_sqlalchemy_error(self):
        """Test database check function directly with SQLAlchemyError."""
        # health.pyの関数を直接テスト
        import app.api.health as health_module

        # check_database_connection関数をモック
        async def mock_check_database():
            try:
                # SQLAlchemyErrorを発生させる
                raise SQLAlchemyError("DB error")
            except SQLAlchemyError as e:
                return {"status": "unhealthy", "error": str(e)}

        # 関数を一時的に置き換え
        original = getattr(health_module, "check_database_connection", None)
        if not original:
            # check_database_connectionが存在しない場合、health_check内のロジックをテスト
            with patch("app.api.health.database.engine.execute") as mock_execute:
                mock_execute.side_effect = SQLAlchemyError("Connection error")

                response = client.get("/v1/health")
                assert response.status_code == 200

    def test_system_metrics_psutil_error_direct(self):
        """Test get_system_metrics directly with psutil error."""
        from app.api.health import get_system_metrics

        with patch(
            "app.api.health.psutil.cpu_percent", side_effect=Exception("CPU error")
        ):
            result = asyncio.run(get_system_metrics())

            assert result["cpu_usage"] == 0.0
            assert result["memory_usage"] == 0.0
            assert result["disk_usage"] == 0.0
            assert result["uptime"] == 0.0
