"""Middleware tests."""

import time
from unittest.mock import Mock, patch

import jwt
import pytest
from fastapi import HTTPException, Request

try:
    from app.core.middleware import (  # type: ignore
        APIKeyAuthenticationMiddleware,  # type: ignore
        CombinedAuthenticationMiddleware,  # type: ignore
        JWTAuthenticationMiddleware,  # type: ignore
        RateLimitMiddleware,  # type: ignore
    )
except ImportError:
    # Create mock middleware classes for testing
    from starlette.middleware.base import BaseHTTPMiddleware

    class JWTAuthenticationMiddleware(BaseHTTPMiddleware):  # type: ignore[no-redef]
        """Mock JWT Authentication middleware."""

        def __init__(self, app, secret_key: str):
            super().__init__(app)
            self.secret_key = secret_key

        async def dispatch(self, request, call_next):
            return await call_next(request)

    class APIKeyAuthenticationMiddleware(BaseHTTPMiddleware):  # type: ignore[no-redef]
        """Mock API Key Authentication middleware."""

        def __init__(self, app):
            super().__init__(app)

        async def dispatch(self, request, call_next):
            return await call_next(request)

    class CombinedAuthenticationMiddleware(BaseHTTPMiddleware):  # type: ignore[no-redef]
        """Mock Combined Authentication middleware."""

        def __init__(self, app, jwt_secret: str):
            super().__init__(app)
            self.jwt_secret = jwt_secret

        async def dispatch(self, request, call_next):
            return await call_next(request)

    class RateLimitMiddleware(BaseHTTPMiddleware):  # type: ignore[no-redef]
        """Mock Rate Limit middleware."""

        def __init__(self, app):
            super().__init__(app)
            self.request_counts = {}

        async def dispatch(self, request, call_next):
            return await call_next(request)


class TestJWTAuthenticationMiddleware:
    """JWT Authentication middleware tests."""

    def test_authenticate_invalid_scheme(self):
        """Test authentication with invalid scheme."""
        middleware = JWTAuthenticationMiddleware()
        mock_request = Mock(spec=Request)
        mock_request.headers = {"Authorization": "Basic dGVzdDp0ZXN0"}  # Basicスキーム

        with pytest.raises(HTTPException) as exc_info:
            middleware.authenticate(mock_request)

        assert exc_info.value.status_code == 401
        assert "Invalid authentication scheme" in str(exc_info.value.detail)

    def test_authenticate_invalid_format(self):
        """Test authentication with invalid authorization header format."""
        middleware = JWTAuthenticationMiddleware()
        mock_request = Mock(spec=Request)
        mock_request.headers = {"Authorization": "InvalidFormat"}  # スペースなし

        with pytest.raises(HTTPException) as exc_info:
            middleware.authenticate(mock_request)

        assert exc_info.value.status_code == 401
        assert "Invalid authorization header format" in str(exc_info.value.detail)

    def test_authenticate_missing_auth(self):
        """Test authentication with missing authorization header."""
        middleware = JWTAuthenticationMiddleware()
        mock_request = Mock(spec=Request)
        mock_request.headers = {}  # Authorizationヘッダーなし

        with pytest.raises(HTTPException) as exc_info:
            middleware.authenticate(mock_request)

        assert exc_info.value.status_code == 401
        assert "Authorization header missing" in str(exc_info.value.detail)

    def test_authenticate_blacklisted_token(self):
        """Test authentication with blacklisted token."""
        middleware = JWTAuthenticationMiddleware()
        mock_request = Mock(spec=Request)
        mock_request.headers = {"Authorization": "Bearer blacklisted_token"}

        with patch("app.core.auth.is_token_blacklisted", return_value=True):
            with pytest.raises(HTTPException) as exc_info:
                middleware.authenticate(mock_request)

            assert exc_info.value.status_code == 401
            assert "Token has been revoked" in str(exc_info.value.detail)

    def test_authenticate_expired_token(self):
        """Test authentication with expired token."""
        middleware = JWTAuthenticationMiddleware()
        mock_request = Mock(spec=Request)
        mock_request.headers = {"Authorization": "Bearer expired_token"}

        with patch("app.core.auth.is_token_blacklisted", return_value=False):
            with patch(
                "app.core.auth.verify_token",
                side_effect=jwt.ExpiredSignatureError("Token expired"),
            ):
                with pytest.raises(HTTPException) as exc_info:
                    middleware.authenticate(mock_request)

                assert exc_info.value.status_code == 401
                assert "Token has expired" in str(exc_info.value.detail)

    def test_authenticate_invalid_token(self):
        """Test authentication with invalid token."""
        middleware = JWTAuthenticationMiddleware()
        mock_request = Mock(spec=Request)
        mock_request.headers = {"Authorization": "Bearer invalid_token"}

        with patch("app.core.auth.is_token_blacklisted", return_value=False):
            with patch(
                "app.core.auth.verify_token",
                side_effect=jwt.InvalidTokenError("Invalid token"),
            ):
                with pytest.raises(HTTPException) as exc_info:
                    middleware.authenticate(mock_request)

                assert exc_info.value.status_code == 401
                assert "Invalid token" in str(exc_info.value.detail)


class TestAPIKeyAuthenticationMiddleware:
    """API Key Authentication middleware tests."""

    def test_authenticate_missing_api_key(self):
        """Test authentication with missing API key."""
        middleware = APIKeyAuthenticationMiddleware()
        mock_request = Mock(spec=Request)
        mock_request.headers = {}  # X-API-Key headerなし

        with pytest.raises(HTTPException) as exc_info:
            middleware.authenticate(mock_request)

        assert exc_info.value.status_code == 401
        assert "API key missing" in str(exc_info.value.detail)

    def test_authenticate_invalid_api_key(self):
        """Test authentication with invalid API key."""
        middleware = APIKeyAuthenticationMiddleware()
        mock_request = Mock(spec=Request)
        mock_request.headers = {"X-API-Key": "invalid_key"}

        with patch("app.core.auth.validate_api_key", return_value=None):
            with pytest.raises(HTTPException) as exc_info:
                middleware.authenticate(mock_request)

            assert exc_info.value.status_code == 401
            assert "Invalid API key" in str(exc_info.value.detail)

    def test_authenticate_valid_api_key(self):
        """Test authentication with valid API key."""
        middleware = APIKeyAuthenticationMiddleware()
        mock_request = Mock(spec=Request)
        mock_request.headers = {"X-API-Key": "valid_key"}

        api_key_info = {"key_id": "test_key", "permissions": ["read"]}
        with patch("app.core.auth.validate_api_key", return_value=api_key_info):
            result = middleware.authenticate(mock_request)

        assert result == api_key_info


class TestCombinedAuthenticationMiddleware:
    """Combined Authentication middleware tests."""

    def test_authenticate_with_jwt(self):
        """Test authentication with JWT token."""
        middleware = CombinedAuthenticationMiddleware()
        mock_request = Mock(spec=Request)
        mock_request.headers = {"Authorization": "Bearer valid_token"}

        jwt_payload = {"user_id": "test_user", "permissions": ["read"]}
        with patch("app.core.auth.is_token_blacklisted", return_value=False):
            with patch("app.core.auth.verify_token", return_value=jwt_payload):
                result = middleware.authenticate(mock_request)

        assert result == jwt_payload

    def test_authenticate_with_api_key_fallback(self):
        """Test authentication with API key when JWT not present."""
        middleware = CombinedAuthenticationMiddleware()
        mock_request = Mock(spec=Request)
        mock_request.headers = {"X-API-Key": "valid_key"}

        api_key_info = {"key_id": "test_key", "permissions": ["read"]}
        with patch("app.core.auth.validate_api_key", return_value=api_key_info):
            result = middleware.authenticate(mock_request)

        assert result == api_key_info

    def test_authenticate_no_credentials(self):
        """Test authentication with no credentials."""
        middleware = CombinedAuthenticationMiddleware()
        mock_request = Mock(spec=Request)
        mock_request.headers = {}

        with pytest.raises(HTTPException) as exc_info:
            middleware.authenticate(mock_request)

        assert exc_info.value.status_code == 401
        assert "Authentication failed" in str(exc_info.value.detail)

    def test_authenticate_invalid_jwt_fallback_to_api_key(self):
        """Test authentication falls back to API key when JWT is invalid."""
        middleware = CombinedAuthenticationMiddleware()
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            "Authorization": "Bearer invalid_token",
            "X-API-Key": "valid_key",
        }

        api_key_info = {"key_id": "test_key", "permissions": ["read"]}
        with patch("app.core.auth.is_token_blacklisted", return_value=False):
            with patch(
                "app.core.auth.verify_token",
                side_effect=jwt.InvalidTokenError("Invalid token"),
            ):
                with patch("app.core.auth.validate_api_key", return_value=api_key_info):
                    result = middleware.authenticate(mock_request)

        assert result == api_key_info


class TestRateLimitMiddleware:
    """Rate limiting middleware tests."""

    def test_check_rate_limit_exceeded(self):
        """Test rate limiting when limit is exceeded."""
        # RateLimitMiddlewareのインスタンスを作成
        middleware = RateLimitMiddleware(max_requests=1, window_seconds=60)

        # モックのリクエストを作成
        mock_request = Mock(spec=Request)
        mock_request.headers = {}
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"

        # レート制限を超えるように設定
        client_id = "127.0.0.1"
        current_time = time.time()
        middleware.request_counts[client_id] = [current_time]  # 既に1リクエスト記録

        # 2回目のリクエストでレート制限を超える
        with pytest.raises(HTTPException) as exc_info:
            middleware.check_rate_limit(mock_request)

        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in str(exc_info.value.detail)

    def test_check_rate_limit_within_limit(self):
        """Test rate limiting within the limit."""
        middleware = RateLimitMiddleware(max_requests=10, window_seconds=60)

        # モックのリクエストを作成
        mock_request = Mock(spec=Request)
        mock_request.headers = {}
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"

        # 制限内のリクエスト
        result = middleware.check_rate_limit(mock_request)
        assert result is True
        assert len(middleware.request_counts["127.0.0.1"]) == 1

    def test_check_rate_limit_with_api_key(self):
        """Test rate limiting with API key."""
        middleware = RateLimitMiddleware(max_requests=5, window_seconds=60)

        # API Keyを持つリクエスト
        mock_request = Mock(spec=Request)
        mock_request.headers = {"X-API-Key": "test-api-key"}
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"

        # API Keyがclient_idとして使用される
        result = middleware.check_rate_limit(mock_request)
        assert result is True
        assert len(middleware.request_counts["test-api-key"]) == 1
        assert len(middleware.request_counts["127.0.0.1"]) == 0

    def test_check_rate_limit_window_expiry(self):
        """Test rate limit window expiry."""
        middleware = RateLimitMiddleware(max_requests=1, window_seconds=1)

        # モックのリクエストを作成
        mock_request = Mock(spec=Request)
        mock_request.headers = {}
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"

        # 古いタイムスタンプを追加（2秒前）
        old_time = time.time() - 2
        middleware.request_counts["127.0.0.1"] = [old_time]

        # ウィンドウが期限切れなので、新しいリクエストは許可される
        result = middleware.check_rate_limit(mock_request)
        assert result is True
        # 古いタイムスタンプは削除され、新しいものだけが残る
        assert len(middleware.request_counts["127.0.0.1"]) == 1
