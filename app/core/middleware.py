"""認証・認可ミドルウェア"""

import time
from typing import Optional, List, Dict, Any
from collections import defaultdict
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import jwt

from app.core.auth import Permission


class JWTAuthenticationMiddleware:
    """JWT認証ミドルウェア"""

    def authenticate(self, request: Request) -> Optional[dict]:
        """JWT認証を実行"""
        authorization = request.headers.get("Authorization")
        if not authorization:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header missing"
            )

        try:
            scheme, token = authorization.split()
            if scheme.lower() != "bearer":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication scheme"
                )
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format"
            )

        # トークンブラックリストチェック
        from app.core.auth import is_token_blacklisted, verify_token
        if is_token_blacklisted(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked"
            )

        try:
            payload = verify_token(token)
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )


class APIKeyAuthenticationMiddleware:
    """API Key認証ミドルウェア"""

    def authenticate(self, request: Request) -> Optional[dict]:
        """API Key認証を実行"""
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key missing"
            )

        from app.core.auth import validate_api_key
        api_key_info = validate_api_key(api_key)
        if not api_key_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )

        return api_key_info


class CombinedAuthenticationMiddleware:
    """複合認証ミドルウェア（JWT + API Key）"""

    def __init__(self):
        self.jwt_middleware = JWTAuthenticationMiddleware()
        self.api_key_middleware = APIKeyAuthenticationMiddleware()

    def authenticate(self, request: Request) -> Optional[dict]:
        """複合認証を実行（JWT優先、API Keyフォールバック）"""
        # まずJWT認証を試行
        try:
            return self.jwt_middleware.authenticate(request)
        except HTTPException:
            pass

        # JWT認証が失敗した場合、API Key認証を試行
        try:
            return self.api_key_middleware.authenticate(request)
        except HTTPException:
            pass

        # どちらも失敗した場合
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


class PermissionMiddleware:
    """権限チェックミドルウェア"""

    def __init__(self, required_permission: str):
        self.required_permission = required_permission

    def check_permission(self, request: Request) -> bool:
        """権限をチェック"""
        user = getattr(request.state, 'user', None)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not authenticated"
            )

        user_permissions = user.get('permissions', [])

        # 文字列をPermissionに変換
        try:
            required_perm = Permission(self.required_permission)
            user_perms = [Permission(p) for p in user_permissions]
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid permission format"
            )

        from app.core.auth import has_permission
        if not has_permission(user_perms, required_perm):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )

        return True


class RoleMiddleware:
    """ロールチェックミドルウェア"""

    def __init__(self, required_role: str):
        self.required_role = required_role

    def check_role(self, request: Request) -> bool:
        """ロールをチェック"""
        user = getattr(request.state, 'user', None)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not authenticated"
            )

        user_role = user.get('role')
        if user_role != self.required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient role"
            )

        return True


class ResourcePermissionMiddleware:
    """リソース権限チェックミドルウェア"""

    def __init__(self, resource_type: str, required_permission: str):
        self.resource_type = resource_type
        self.required_permission = required_permission

    def check_resource_permission(self, request: Request) -> bool:
        """リソース権限をチェック"""
        user = getattr(request.state, 'user', None)
        if not user:
            return False

        user_id = user.get("user_id")
        resource_id = request.path_params.get(f"{self.resource_type}_id")

        # check_user_resource_permission関数を呼び出し
        from app.core.auth import check_user_resource_permission
        return check_user_resource_permission(
            user_id, self.resource_type, resource_id, self.required_permission
        )


class RateLimitMiddleware:
    """レート制限ミドルウェア"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_counts = defaultdict(list)

    def check_rate_limit(self, request: Request) -> bool:
        """レート制限をチェック"""
        # クライアントIPまたはAPI Keyを識別子として使用
        api_key = request.headers.get("X-API-Key")
        client_id = api_key or request.client.host

        current_time = time.time()

        # 古いリクエスト記録を削除
        self.request_counts[client_id] = [
            timestamp for timestamp in self.request_counts[client_id]
            if current_time - timestamp < self.window_seconds
        ]

        # 現在のリクエスト数をチェック
        if len(self.request_counts[client_id]) >= self.max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )

        # 現在のリクエストを記録
        self.request_counts[client_id].append(current_time)
        return True


class TokenBlacklistMiddleware:
    """トークンブラックリストミドルウェア"""

    def check_token_blacklist(self, request: Request) -> bool:
        """トークンがブラックリストに登録されているかチェック"""
        authorization = request.headers.get("Authorization")
        if not authorization:
            return False

        try:
            scheme, token = authorization.split()
            if scheme.lower() != "bearer":
                return False
        except ValueError:
            return False

        return is_token_blacklisted(token)

    def check_blacklist(self, request: Request) -> bool:
        """トークンブラックリストをチェック（テスト用の別名）"""
        return self.check_token_blacklist(request)


class MiddlewareChain:
    """ミドルウェアチェーン管理"""

    def __init__(self):
        self.middleware = []

    def add_middleware(self, name: str, priority: int = 0):
        """ミドルウェアを追加"""
        self.middleware.append((name, priority))

    def get_ordered_middleware(self) -> List[str]:
        """優先度順にミドルウェアを取得"""
        sorted_middleware = sorted(self.middleware, key=lambda x: x[1])
        return [name for name, _ in sorted_middleware]


class ConditionalMiddleware:
    """条件付きミドルウェア"""

    def __init__(self, skip_paths: List[str] = None):
        self.skip_paths = skip_paths or []

    def should_skip_authentication(self, request: Request) -> bool:
        """認証をスキップするかどうか判定"""
        path = request.url.path
        return path in self.skip_paths


class ErrorHandlingMiddleware:
    """エラーハンドリングミドルウェア"""

    def handle_auth_error(self, request: Request, exc: HTTPException) -> JSONResponse:
        """認証エラーをハンドリング"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "AUTHENTICATION_ERROR",
                    "message": str(exc.detail),
                    "type": "authentication",
                    "timestamp": time.time()
                }
            }
        )

    def handle_authz_error(self, request: Request, exc: HTTPException) -> JSONResponse:
        """認可エラーをハンドリング"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "AUTHORIZATION_ERROR",
                    "message": str(exc.detail),
                    "type": "authorization",
                    "timestamp": time.time()
                }
            }
        )
