"""API Key認証のテスト"""

from fastapi.testclient import TestClient

from app.main import app


class TestAPIKeyAuthentication:
    """API Key認証システムのテストクラス"""

    def test_api_key_generation(self):
        """API Key生成のテスト"""
        from app.core.auth import generate_api_key

        api_key = generate_api_key()

        assert isinstance(api_key, str)
        assert len(api_key) >= 32  # 最低32文字
        assert api_key.isalnum() or "-" in api_key or "_" in api_key

    def test_api_key_validation(self):
        """API Key検証のテスト"""
        from app.core.auth import validate_api_key

        # 有効なAPI Key
        valid_key = "ak_" + "test_1234567890abcdef"
        result = validate_api_key(valid_key)

        assert result is not None
        assert "user_id" in result
        assert "permissions" in result

    def test_api_key_invalid(self):
        """無効なAPI Key検証のテスト"""
        from app.core.auth import validate_api_key

        invalid_key = "invalid_key"
        result = validate_api_key(invalid_key)

        assert result is None

    def test_api_key_format_validation(self):
        """API Keyフォーマット検証のテスト"""
        from app.core.auth import is_valid_api_key_format

        # 有効なフォーマット
        valid_formats = [
            "ak_" + "test_1234567890abcdef",
            "ak_prod_abcdef1234567890",
            "ak_dev_0123456789abcdef",
        ]

        for key in valid_formats:
            assert is_valid_api_key_format(key) is True

        # 無効なフォーマット
        invalid_formats = ["invalid_key", "ak_", "test_1234", "ak_" + "test_short"]

        for key in invalid_formats:
            assert is_valid_api_key_format(key) is False


class TestAPIKeyAuthenticationAPI:
    """API Key認証APIのテストクラス"""

    def test_api_key_create_endpoint(self):
        """API Key作成エンドポイントのテスト"""
        client = TestClient(app)

        # JWT認証でログイン
        login_data = {"username": "test@example.com", "password": "testpassword"}
        login_response = client.post("/v1/auth/login", data=login_data)
        access_token = login_response.json()["access_token"]

        # API Key作成
        headers = {"Authorization": f"Bearer {access_token}"}
        api_key_data = {"name": "Test API Key", "permissions": ["read", "write"]}
        response = client.post("/v1/auth/api-keys", json=api_key_data, headers=headers)

        assert response.status_code == 201
        data = response.json()
        assert "api_key" in data
        assert "name" in data
        assert "permissions" in data
        assert data["name"] == "Test API Key"

    def test_api_key_list_endpoint(self):
        """API Key一覧取得エンドポイントのテスト"""
        client = TestClient(app)

        # JWT認証でログイン
        login_data = {"username": "test@example.com", "password": "testpassword"}
        login_response = client.post("/v1/auth/login", data=login_data)
        access_token = login_response.json()["access_token"]

        # API Key一覧取得
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.get("/v1/auth/api-keys", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert "api_keys" in data
        assert isinstance(data["api_keys"], list)

    def test_api_key_revoke_endpoint(self):
        """API Key無効化エンドポイントのテスト"""
        client = TestClient(app)

        # JWT認証でログイン
        login_data = {"username": "test@example.com", "password": "testpassword"}
        login_response = client.post("/v1/auth/login", data=login_data)
        access_token = login_response.json()["access_token"]

        # API Key作成
        headers = {"Authorization": f"Bearer {access_token}"}
        api_key_data = {"name": "Test API Key", "permissions": ["read"]}
        create_response = client.post(
            "/v1/auth/api-keys", json=api_key_data, headers=headers
        )
        api_key_id = create_response.json()["id"]

        # API Key無効化
        response = client.delete(f"/v1/auth/api-keys/{api_key_id}", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "API key revoked successfully"

    def test_protected_endpoint_with_api_key(self):
        """API Keyでの保護されたエンドポイントアクセステスト"""
        client = TestClient(app)

        # 有効なAPI Keyでアクセス
        headers = {"X-API-Key": "ak_" + "test_1234567890abcdef"}
        response = client.get("/v1/documents", headers=headers)

        assert response.status_code == 200

    def test_protected_endpoint_with_invalid_api_key(self):
        """無効なAPI Keyでの保護されたエンドポイントアクセステスト"""
        client = TestClient(app)

        headers = {"X-API-Key": "invalid_api_key"}
        response = client.get("/v1/documents", headers=headers)

        assert response.status_code == 401
        data = response.json()
        assert data["error"]["code"] == "AUTHENTICATION_ERROR"

    def test_api_key_permissions_check(self):
        """API Key権限チェックのテスト"""
        client = TestClient(app)

        # 読み取り専用API Key
        headers = {"X-API-Key": "ak_" + "readonly_1234567890abcdef"}

        # 読み取り操作（成功）
        response = client.get("/v1/documents", headers=headers)
        assert response.status_code == 200

        # 書き込み操作（失敗）
        document_data = {
            "title": "Test Document",
            "content": "Test content",
            "source_type": "test",
        }
        response = client.post("/v1/documents", json=document_data, headers=headers)
        assert response.status_code == 403
        data = response.json()
        assert data["error"]["code"] == "AUTHORIZATION_ERROR"


class TestAPIKeyManagement:
    """API Key管理のテストクラス"""

    def test_api_key_storage(self):
        """API Key保存のテスト"""
        from app.core.auth import get_api_key_info, store_api_key

        api_key_data = {
            "key": "ak_" + "test_1234567890abcdef",
            "user_id": "user123",
            "name": "Test Key",
            "permissions": ["read", "write"],
        }

        store_api_key(api_key_data)

        # 保存されたAPI Key情報を取得
        stored_data = get_api_key_info("ak_" + "test_1234567890abcdef")
        assert stored_data is not None
        assert stored_data["user_id"] == "user123"
        assert stored_data["name"] == "Test Key"
        assert stored_data["permissions"] == ["read", "write"]

    def test_api_key_expiration(self):
        """API Key有効期限のテスト"""
        from datetime import datetime, timedelta

        from app.core.auth import create_api_key_with_expiration, is_api_key_expired

        # 1時間後に期限切れのAPI Key
        expiration = datetime.utcnow() + timedelta(hours=1)
        api_key = create_api_key_with_expiration("user123", expiration)

        assert is_api_key_expired(api_key) is False

        # 過去の時刻で期限切れのAPI Key
        past_expiration = datetime.utcnow() - timedelta(hours=1)
        expired_key = create_api_key_with_expiration("user123", past_expiration)

        assert is_api_key_expired(expired_key) is True

    def test_api_key_usage_tracking(self):
        """API Key使用状況追跡のテスト"""
        from app.core.auth import get_api_key_usage_stats, track_api_key_usage

        api_key = "ak_" + "test_1234567890abcdef"

        # 使用状況を記録
        track_api_key_usage(api_key, "/v1/documents", "GET")
        track_api_key_usage(api_key, "/v1/search", "POST")

        # 使用統計を取得
        stats = get_api_key_usage_stats(api_key)
        assert stats["total_requests"] >= 2
        assert "/v1/documents" in stats["endpoints"]
        assert "/v1/search" in stats["endpoints"]
