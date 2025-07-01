"""認証ミドルウェアのテスト"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app


class TestAuthenticationMiddleware:
    """認証ミドルウェアのテストクラス"""

    def test_jwt_middleware_valid_token(self):
        """有効なJWTトークンでのミドルウェアテスト"""
        from app.core.middleware import JWTAuthenticationMiddleware

        # モックリクエスト
        request = MagicMock()
        request.headers = {"Authorization": "Bearer valid_jwt_token"}

        middleware = JWTAuthenticationMiddleware()

        with (
            patch("app.core.auth.verify_token") as mock_verify,
            patch("app.core.auth.is_token_blacklisted") as mock_blacklist,
        ):
            mock_verify.return_value = {"sub": "test@example.com", "role": "user"}
            mock_blacklist.return_value = False

            result = middleware.authenticate(request)

            assert result is not None
            assert result["sub"] == "test@example.com"
            assert result["role"] == "user"

    def test_jwt_middleware_invalid_token(self):
        """無効なJWTトークンでのミドルウェアテスト"""
        from jwt import InvalidTokenError

        from app.core.middleware import JWTAuthenticationMiddleware

        request = MagicMock()
        request.headers = {"Authorization": "Bearer invalid_jwt_token"}

        middleware = JWTAuthenticationMiddleware()

        with patch("app.core.auth.verify_token") as mock_verify:
            mock_verify.side_effect = InvalidTokenError("Invalid token")

            with pytest.raises(HTTPException) as exc_info:
                middleware.authenticate(request)

            assert exc_info.value.status_code == 401

    def test_jwt_middleware_missing_token(self):
        """トークンなしでのミドルウェアテスト"""
        from app.core.middleware import JWTAuthenticationMiddleware

        request = MagicMock()
        request.headers = {}

        middleware = JWTAuthenticationMiddleware()

        with pytest.raises(HTTPException) as exc_info:
            middleware.authenticate(request)

        assert exc_info.value.status_code == 401

    def test_api_key_middleware_valid_key(self):
        """有効なAPI Keyでのミドルウェアテスト"""
        from app.core.middleware import APIKeyAuthenticationMiddleware

        request = MagicMock()
        request.headers = {"X-API-Key": "ak_test_1234567890abcdef"}

        middleware = APIKeyAuthenticationMiddleware()

        with patch("app.core.auth.validate_api_key") as mock_validate:
            mock_validate.return_value = {
                "user_id": "user123",
                "permissions": ["read", "write"],
            }

            result = middleware.authenticate(request)

            assert result is not None
            assert result["user_id"] == "user123"
            assert "read" in result["permissions"]

    def test_api_key_middleware_invalid_key(self):
        """無効なAPI Keyでのミドルウェアテスト"""
        from app.core.middleware import APIKeyAuthenticationMiddleware

        request = MagicMock()
        request.headers = {"X-API-Key": "invalid_api_key"}

        middleware = APIKeyAuthenticationMiddleware()

        with patch("app.core.auth.validate_api_key") as mock_validate:
            mock_validate.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                middleware.authenticate(request)

            assert exc_info.value.status_code == 401

    def test_combined_auth_middleware(self):
        """複合認証ミドルウェアのテスト"""
        from app.core.middleware import CombinedAuthenticationMiddleware

        # JWT認証を試行
        request = MagicMock()
        request.headers = {"Authorization": "Bearer valid_jwt_token"}

        middleware = CombinedAuthenticationMiddleware()

        with (
            patch("app.core.auth.verify_token") as mock_verify,
            patch("app.core.auth.is_token_blacklisted") as mock_blacklist,
        ):
            mock_verify.return_value = {"sub": "test@example.com", "role": "user"}
            mock_blacklist.return_value = False

            result = middleware.authenticate(request)

            assert result is not None
            assert result["sub"] == "test@example.com"

    def test_combined_auth_fallback_to_api_key(self):
        """複合認証でAPI Keyフォールバックのテスト"""
        from jwt import InvalidTokenError

        from app.core.middleware import CombinedAuthenticationMiddleware

        # JWT無効、API Key有効
        request = MagicMock()
        request.headers = {
            "Authorization": "Bearer invalid_jwt_token",
            "X-API-Key": "ak_test_1234567890abcdef",
        }

        middleware = CombinedAuthenticationMiddleware()

        with (
            patch("app.core.auth.verify_token") as mock_verify,
            patch("app.core.auth.validate_api_key") as mock_validate,
        ):

            mock_verify.side_effect = InvalidTokenError("Invalid token")
            mock_validate.return_value = {"user_id": "user123", "permissions": ["read"]}

            result = middleware.authenticate(request)

            assert result is not None
            assert result["user_id"] == "user123"


class TestAuthorizationMiddleware:
    """認可ミドルウェアのテストクラス"""

    def test_permission_middleware_success(self):
        """権限チェック成功のテスト"""
        from app.core.middleware import PermissionMiddleware

        request = MagicMock()
        request.state.user = {"permissions": ["read", "write"], "role": "editor"}

        middleware = PermissionMiddleware(required_permission="read")

        # 権限チェック成功
        result = middleware.check_permission(request)
        assert result is True

    def test_permission_middleware_failure(self):
        """権限チェック失敗のテスト"""
        from app.core.middleware import PermissionMiddleware

        request = MagicMock()
        request.state.user = {"permissions": ["read"], "role": "user"}

        middleware = PermissionMiddleware(required_permission="delete")

        # 権限チェック失敗
        with pytest.raises(HTTPException) as exc_info:
            middleware.check_permission(request)

        assert exc_info.value.status_code == 403

    def test_role_middleware_success(self):
        """ロールチェック成功のテスト"""
        from app.core.middleware import RoleMiddleware

        request = MagicMock()
        request.state.user = {
            "role": "admin",
            "permissions": ["read", "write", "delete", "admin"],
        }

        middleware = RoleMiddleware(required_role="admin")

        # ロールチェック成功
        result = middleware.check_role(request)
        assert result is True

    def test_role_middleware_failure(self):
        """ロールチェック失敗のテスト"""
        from app.core.middleware import RoleMiddleware

        request = MagicMock()
        request.state.user = {"role": "user", "permissions": ["read"]}

        middleware = RoleMiddleware(required_role="admin")

        # ロールチェック失敗
        with pytest.raises(HTTPException) as exc_info:
            middleware.check_role(request)

        assert exc_info.value.status_code == 403

    def test_resource_permission_middleware(self):
        """リソース権限ミドルウェアのテスト"""
        from app.core.middleware import ResourcePermissionMiddleware

        request = MagicMock()
        request.state.user = {"user_id": "user123"}
        request.path_params = {"document_id": "doc123"}

        middleware = ResourcePermissionMiddleware(
            resource_type="document", required_permission="read"
        )

        with patch("app.core.auth.check_user_resource_permission") as mock_check:
            mock_check.return_value = True

            result = middleware.check_resource_permission(request)
            assert result is True

            mock_check.assert_called_once_with("user123", "document", "doc123", "read")


class TestMiddlewareIntegration:
    """ミドルウェア統合テスト"""

    def test_authentication_flow_integration(self):
        """認証フロー統合テスト"""
        client = TestClient(app)

        # 1. 認証なしでアクセス（拒否）
        response = client.get("/v1/documents")
        assert response.status_code == 401

        # 2. 有効なJWTトークンでアクセス（成功）
        with patch("app.core.auth.verify_token") as mock_verify:
            mock_verify.return_value = {
                "sub": "test@example.com",
                "role": "user",
                "permissions": ["read"],
            }

            headers = {"Authorization": "Bearer valid_jwt_token"}
            response = client.get("/v1/documents", headers=headers)
            assert response.status_code == 200

    def test_authorization_flow_integration(self):
        """認可フロー統合テスト"""
        client = TestClient(app)

        # 読み取り権限のみのユーザー
        with patch("app.core.auth.verify_token") as mock_verify:
            mock_verify.return_value = {
                "sub": "readonly@example.com",
                "role": "user",
                "permissions": ["read"],
            }

            headers = {"Authorization": "Bearer readonly_token"}

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

    def test_api_key_rate_limiting(self):
        """API Keyレート制限のテスト"""
        from app.core.middleware import RateLimitMiddleware

        request = MagicMock()
        request.headers = {"X-API-Key": "ak_test_1234567890abcdef"}
        request.client.host = "127.0.0.1"

        middleware = RateLimitMiddleware(max_requests=5, window_seconds=60)

        # 制限内のリクエスト
        for i in range(5):
            result = middleware.check_rate_limit(request)
            assert result is True

        # 制限超過
        with pytest.raises(HTTPException) as exc_info:
            middleware.check_rate_limit(request)

        assert exc_info.value.status_code == 429

    def test_token_blacklist_middleware(self):
        """トークンブラックリストミドルウェアのテスト"""
        from app.core.middleware import TokenBlacklistMiddleware

        request = MagicMock()
        request.headers = {"Authorization": "Bearer blacklisted_token"}

        middleware = TokenBlacklistMiddleware()

        with patch("app.core.auth.is_token_blacklisted") as mock_blacklist:
            mock_blacklist.return_value = True

            result = middleware.check_blacklist(request)
            assert result is True  # ブラックリストに登録されている場合はTrueを返す


class TestMiddlewareConfiguration:
    """ミドルウェア設定のテスト"""

    def test_middleware_order(self):
        """ミドルウェア実行順序のテスト"""
        from app.core.middleware import MiddlewareChain

        chain = MiddlewareChain()

        # ミドルウェアを順序付きで追加
        chain.add_middleware("authentication", priority=1)
        chain.add_middleware("authorization", priority=2)
        chain.add_middleware("rate_limiting", priority=3)

        ordered_middleware = chain.get_ordered_middleware()

        assert ordered_middleware[0] == "authentication"
        assert ordered_middleware[1] == "authorization"
        assert ordered_middleware[2] == "rate_limiting"

    def test_conditional_middleware(self):
        """条件付きミドルウェアのテスト"""
        from app.core.middleware import ConditionalMiddleware

        request = MagicMock()
        request.url.path = "/v1/auth/login"

        # ログインエンドポイントでは認証不要
        middleware = ConditionalMiddleware(skip_paths=["/v1/auth/login", "/v1/health"])

        should_skip = middleware.should_skip_authentication(request)
        assert should_skip is True

        # 他のエンドポイントでは認証必要
        request.url.path = "/v1/documents"
        should_skip = middleware.should_skip_authentication(request)
        assert should_skip is False

    def test_middleware_error_handling(self):
        """ミドルウェアエラーハンドリングのテスト"""
        from app.core.middleware import ErrorHandlingMiddleware

        request = MagicMock()
        middleware = ErrorHandlingMiddleware()

        # 認証エラー
        auth_error = HTTPException(status_code=401, detail="Authentication failed")
        response = middleware.handle_auth_error(request, auth_error)

        assert response.status_code == 401
        assert "AUTHENTICATION_ERROR" in response.body.decode()

        # 認可エラー
        authz_error = HTTPException(status_code=403, detail="Authorization failed")
        response = middleware.handle_authz_error(request, authz_error)

        assert response.status_code == 403
        assert "AUTHORIZATION_ERROR" in response.body.decode()
