"""Tests for ApertureDB Mock

ApertureDBモッククラスの包括的なテスト。
カバレッジの向上を目的として、すべてのメソッドとエラーケースをテスト。
"""

import os
from unittest.mock import patch

import pytest

from app.models.aperturedb_mock import Client, DBError, MockClient, Utils


class TestMockClient:
    """MockClientクラスのテスト"""

    def test_init_with_defaults(self):
        """デフォルト値での初期化テスト"""
        client = MockClient()
        assert client.host == "localhost"
        assert client.port == 55555
        assert client.username == "admin"
        assert client.password == "admin"
        assert client._connected is True

    def test_init_with_params(self):
        """パラメータ指定での初期化テスト"""
        client = MockClient(
            host="testhost", port=12345, username="testuser", password="testpass"
        )
        assert client.host == "testhost"
        assert client.port == 12345
        assert client.username == "testuser"
        assert client.password == "testpass"

    def test_init_with_env_vars(self):
        """環境変数からの初期化テスト"""
        with patch.dict(
            os.environ,
            {"APERTUREDB_USERNAME": "env_user", "APERTUREDB_PASSWORD": "env_pass"},
        ):
            client = MockClient()
            assert client.username == "env_user"
            assert client.password == "env_pass"

    def test_query_empty(self):
        """空のクエリのテスト"""
        client = MockClient()
        result, data = client.query([])
        assert result == []
        assert data == []

    def test_query_find_descriptor_set(self):
        """FindDescriptorSetクエリのテスト"""
        client = MockClient()
        query = [{"FindDescriptorSet": {"name": "test_set"}}]
        result, data = client.query(query)

        assert len(result) == 1
        assert "FindDescriptorSet" in result[0]
        assert result[0]["FindDescriptorSet"]["count"] == 0

    def test_query_add_descriptor_set(self):
        """AddDescriptorSetクエリのテスト"""
        client = MockClient()
        query = [{"AddDescriptorSet": {"name": "test_set", "dimensions": 128}}]
        result, data = client.query(query)

        assert len(result) == 1
        assert "AddDescriptorSet" in result[0]
        assert result[0]["AddDescriptorSet"]["status"] == 0

    def test_query_add_descriptor(self):
        """AddDescriptorクエリのテスト"""
        client = MockClient()
        query = [{"AddDescriptor": {"set": "test_set", "label": "test"}}]
        result, data = client.query(query)

        assert len(result) == 1
        assert "AddDescriptor" in result[0]
        assert result[0]["AddDescriptor"]["status"] == 0

    def test_query_find_descriptor(self):
        """FindDescriptorクエリのテスト"""
        client = MockClient()
        query = [{"FindDescriptor": {"set": "test_set", "k_neighbors": 10}}]
        result, data = client.query(query)

        assert len(result) == 1
        assert "FindDescriptor" in result[0]
        assert result[0]["FindDescriptor"]["returned"] == 0
        assert result[0]["FindDescriptor"]["entities"] == []

    def test_query_delete_descriptor(self):
        """DeleteDescriptorクエリのテスト"""
        client = MockClient()
        query = [{"DeleteDescriptor": {"set": "test_set", "label": "test"}}]
        result, data = client.query(query)

        assert len(result) == 1
        assert "DeleteDescriptor" in result[0]
        assert result[0]["DeleteDescriptor"]["status"] == 0

    def test_query_add_entity(self):
        """AddEntityクエリのテスト"""
        client = MockClient()
        query = [{"AddEntity": {"class": "TestEntity", "properties": {"name": "test"}}}]
        result, data = client.query(query)

        assert len(result) == 1
        assert "AddEntity" in result[0]
        assert result[0]["AddEntity"]["status"] == 0

    def test_query_get_status(self):
        """GetStatusクエリのテスト"""
        client = MockClient()
        query = [{"GetStatus": {}}]
        result, data = client.query(query)

        assert len(result) == 1
        assert "GetStatus" in result[0]
        assert result[0]["GetStatus"]["status"] == "ready"

    def test_query_unknown_type(self):
        """未知のクエリタイプのテスト"""
        client = MockClient()
        query = [{"UnknownQuery": {"param": "value"}}]
        result, data = client.query(query)

        assert len(result) == 1
        assert result[0] == {}

    def test_query_multiple_queries(self):
        """複数クエリのテスト（最初のクエリのみ処理される）"""
        client = MockClient()
        query = [
            {"FindDescriptorSet": {"name": "test_set"}},
            {"AddDescriptor": {"set": "test_set", "label": "test"}},
        ]
        result, data = client.query(query)

        # 最初のクエリの結果のみが返される
        assert len(result) == 1
        assert "FindDescriptorSet" in result[0]


class TestDBError:
    """DBErrorクラスのテスト"""

    def test_db_error_creation(self):
        """DBError例外の作成テスト"""
        error = DBError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_db_error_raise(self):
        """DBError例外の発生テスト"""
        with pytest.raises(DBError, match="Test error"):
            raise DBError("Test error")


class TestUtils:
    """Utilsクラスのテスト"""

    def test_utils_instantiation(self):
        """Utilsクラスのインスタンス化テスト"""
        utils = Utils()
        assert utils is not None
        assert isinstance(utils, Utils)


class TestModuleExports:
    """モジュールエクスポートのテスト"""

    def test_client_export(self):
        """Client エクスポートのテスト"""
        assert Client is MockClient

    def test_db_exception_export(self):
        """DBException エクスポートのテスト"""
        from app.models.aperturedb_mock import DBException

        assert DBException is DBError

    def test_creating_client_via_export(self):
        """エクスポートされたClient経由でのインスタンス作成テスト"""
        client = Client(host="test", port=9999)
        assert isinstance(client, MockClient)
        assert client.host == "test"
        assert client.port == 9999
