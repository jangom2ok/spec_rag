# Step06: 認証・認可システム

## 🎯 この章の目標
JWT・API Key認証の実装詳細、RBAC（Role-Based Access Control）、セキュリティベストプラクティスを理解する

---

## 📋 概要

RAGシステムでは、企業の機密文書を扱うため、堅牢な認証・認可システムが不可欠です。JWT Token認証とAPI Key認証のハイブリッド方式により、Webアプリケーションと外部システム連携の両方に対応します。

### 🏗️ 認証システム構成

```
認証・認可フロー
├── 認証方式
│   ├── JWT Token認証    # Webアプリケーション用
│   ├── API Key認証      # 外部システム連携用
│   └── フォールバック   # 複数方式の自動選択
├── 認可システム
│   ├── RBAC            # ロールベースアクセス制御
│   ├── リソース権限    # エンドポイント別権限
│   └── データフィルタ  # ユーザー別データ制限
└── セキュリティ機能
    ├── レート制限      # API制限・DoS対策
    ├── セッション管理  # トークン管理・無効化
    └── 監査ログ        # セキュリティイベント記録
```

---

## 🔐 JWT Token認証システム

### 1. JWT Token構造とクレーム

#### JWT Payload設計
```python
@dataclass
class JWTClaims:
    """JWTクレーム（ペイロード）構造"""
    
    # 標準クレーム
    sub: str            # Subject (ユーザーID/email)
    iss: str            # Issuer (発行者)
    aud: str            # Audience (対象システム)
    exp: int            # Expiration (有効期限)
    iat: int            # Issued At (発行時刻)
    jti: str            # JWT ID (トークンID)
    
    # カスタムクレーム
    user_id: str        # ユーザーUUID
    username: str       # ユーザー名
    email: str          # メールアドレス
    role: str           # ロール
    permissions: list[str]  # 権限リスト
    session_id: str     # セッションID
    
    # メタデータ
    login_method: str   # ログイン方式
    ip_address: str     # ログイン元IP
    user_agent: str     # ユーザーエージェント
    
    def to_dict(self) -> dict:
        return asdict(self)

# JWT作成例
sample_jwt_payload = {
    "sub": "user@example.com",
    "iss": "rag-system",
    "aud": "rag-api",
    "exp": 1735689600,  # 2025-01-01 00:00:00
    "iat": 1704067200,  # 2024-01-01 00:00:00
    "jti": "uuid-jwt-token-id",
    
    "user_id": "uuid-user-id",
    "username": "john_doe",
    "email": "user@example.com",
    "role": "editor",
    "permissions": ["read", "write"],
    "session_id": "uuid-session-id",
    
    "login_method": "password",
    "ip_address": "192.168.1.100",
    "user_agent": "Mozilla/5.0..."
}
```

### 2. JWT トークン管理サービス

```python
import jwt
import secrets
from datetime import datetime, timedelta
from typing import Any, Optional

class JWTService:
    """JWT トークン管理サービス"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire = timedelta(minutes=30)
        self.refresh_token_expire = timedelta(days=7)
    
    async def create_access_token(
        self, 
        user: dict, 
        request_info: dict
    ) -> str:
        """アクセストークンの生成"""
        
        now = datetime.utcnow()
        
        claims = JWTClaims(
            sub=user["email"],
            iss="rag-system",
            aud="rag-api",
            exp=int((now + self.access_token_expire).timestamp()),
            iat=int(now.timestamp()),
            jti=secrets.token_urlsafe(32),
            
            user_id=user["id"],
            username=user["username"],
            email=user["email"],
            role=user["role"],
            permissions=user["permissions"],
            session_id=secrets.token_urlsafe(16),
            
            login_method="password",
            ip_address=request_info.get("ip_address"),
            user_agent=request_info.get("user_agent")
        )
        
        # セッション情報をRedisに保存
        await self._store_session(claims.session_id, claims.to_dict())
        
        # JWT生成
        token = jwt.encode(
            claims.to_dict(),
            self.secret_key,
            algorithm=self.algorithm
        )
        
        return token
    
    async def create_refresh_token(self, user_id: str) -> str:
        """リフレッシュトークンの生成"""
        
        now = datetime.utcnow()
        
        refresh_claims = {
            "sub": user_id,
            "iss": "rag-system",
            "aud": "rag-refresh",
            "exp": int((now + self.refresh_token_expire).timestamp()),
            "iat": int(now.timestamp()),
            "jti": secrets.token_urlsafe(32),
            "token_type": "refresh"
        }
        
        # リフレッシュトークンをデータベースに保存
        await self._store_refresh_token(refresh_claims)
        
        return jwt.encode(
            refresh_claims,
            self.secret_key,
            algorithm=self.algorithm
        )
    
    async def verify_token(self, token: str) -> dict[str, Any]:
        """トークンの検証とデコード"""
        
        try:
            # JWT デコード
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # トークンタイプチェック
            if payload.get("token_type") == "refresh":
                raise ValueError("Access token required, refresh token provided")
            
            # セッション確認
            session_id = payload.get("session_id")
            if session_id:
                session_valid = await self._verify_session(session_id)
                if not session_valid:
                    raise ValueError("Session has been invalidated")
            
            # ブラックリスト確認
            jti = payload.get("jti")
            if await self._is_token_blacklisted(jti):
                raise ValueError("Token has been revoked")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {e}")
    
    async def revoke_token(self, token: str) -> None:
        """トークンの無効化"""
        
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False}  # 期限切れでも処理
            )
            
            jti = payload.get("jti")
            exp = payload.get("exp")
            
            if jti and exp:
                # 期限まで黒リストに追加
                expire_time = datetime.fromtimestamp(exp)
                await self._blacklist_token(jti, expire_time)
                
                # セッション無効化
                session_id = payload.get("session_id")
                if session_id:
                    await self._invalidate_session(session_id)
                    
        except jwt.InvalidTokenError:
            # 無効なトークンは無視
            pass
    
    async def refresh_access_token(self, refresh_token: str) -> tuple[str, str]:
        """アクセストークンの更新"""
        
        try:
            payload = jwt.decode(
                refresh_token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            if payload.get("token_type") != "refresh":
                raise ValueError("Refresh token required")
            
            user_id = payload.get("sub")
            
            # リフレッシュトークンの有効性確認
            if not await self._verify_refresh_token(payload["jti"]):
                raise ValueError("Refresh token has been revoked")
            
            # ユーザー情報取得
            user = await self._get_user_by_id(user_id)
            if not user or not user["is_active"]:
                raise ValueError("User not found or inactive")
            
            # 新しいトークンペア生成
            new_access_token = await self.create_access_token(
                user, {"ip_address": None, "user_agent": None}
            )
            new_refresh_token = await self.create_refresh_token(user_id)
            
            # 古いリフレッシュトークンを無効化
            await self._revoke_refresh_token(payload["jti"])
            
            return new_access_token, new_refresh_token
            
        except jwt.ExpiredSignatureError:
            raise ValueError("Refresh token has expired")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid refresh token: {e}")
```

### 3. セッション管理

```python
import aioredis
from typing import Optional

class SessionManager:
    """セッション管理サービス"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.session_prefix = "session:"
        self.blacklist_prefix = "blacklist:"
        self.session_timeout = 3600 * 24 * 7  # 7日間
    
    async def store_session(self, session_id: str, session_data: dict) -> None:
        """セッション情報の保存"""
        
        session_key = f"{self.session_prefix}{session_id}"
        
        # セッションデータをJSON形式で保存
        await self.redis.setex(
            session_key,
            self.session_timeout,
            json.dumps(session_data)
        )
        
        # ユーザー別セッション一覧に追加
        user_sessions_key = f"user_sessions:{session_data['user_id']}"
        await self.redis.sadd(user_sessions_key, session_id)
        await self.redis.expire(user_sessions_key, self.session_timeout)
    
    async def get_session(self, session_id: str) -> Optional[dict]:
        """セッション情報の取得"""
        
        session_key = f"{self.session_prefix}{session_id}"
        session_data = await self.redis.get(session_key)
        
        if session_data:
            return json.loads(session_data)
        return None
    
    async def invalidate_session(self, session_id: str) -> None:
        """特定セッションの無効化"""
        
        session_key = f"{self.session_prefix}{session_id}"
        
        # セッションデータ取得
        session_data = await self.get_session(session_id)
        
        if session_data:
            # ユーザーセッション一覧から削除
            user_sessions_key = f"user_sessions:{session_data['user_id']}"
            await self.redis.srem(user_sessions_key, session_id)
        
        # セッション削除
        await self.redis.delete(session_key)
    
    async def invalidate_all_user_sessions(self, user_id: str) -> None:
        """ユーザーの全セッション無効化"""
        
        user_sessions_key = f"user_sessions:{user_id}"
        session_ids = await self.redis.smembers(user_sessions_key)
        
        # 各セッションを無効化
        for session_id in session_ids:
            await self.invalidate_session(session_id.decode())
        
        # ユーザーセッション一覧も削除
        await self.redis.delete(user_sessions_key)
    
    async def blacklist_token(self, jti: str, expire_time: datetime) -> None:
        """トークンの黒リスト追加"""
        
        blacklist_key = f"{self.blacklist_prefix}{jti}"
        
        # 有効期限まで黒リストに保持
        ttl = int((expire_time - datetime.utcnow()).total_seconds())
        if ttl > 0:
            await self.redis.setex(blacklist_key, ttl, "revoked")
    
    async def is_token_blacklisted(self, jti: str) -> bool:
        """トークン黒リスト確認"""
        
        blacklist_key = f"{self.blacklist_prefix}{jti}"
        return await self.redis.exists(blacklist_key) > 0
```

---

## 🔑 API Key認証システム

### 1. API Key生成と管理

```python
import hashlib
import secrets
from typing import Optional

class APIKeyService:
    """API Key管理サービス"""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.key_prefix = "rag_"
        self.key_length = 32
    
    async def generate_api_key(
        self,
        user_id: str,
        name: str,
        permissions: list[str],
        expires_at: Optional[datetime] = None,
        rate_limit: int = 100
    ) -> tuple[str, str]:
        """API Key生成"""
        
        # ランダムキー生成
        raw_key = secrets.token_urlsafe(self.key_length)
        full_key = f"{self.key_prefix}{raw_key}"
        
        # キーハッシュ化（保存用）
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()
        
        # プレフィックス（表示用）
        key_prefix_display = full_key[:12] + "..."
        
        # データベースに保存
        api_key_id = await self._store_api_key(
            user_id=user_id,
            key_hash=key_hash,
            key_prefix=key_prefix_display,
            name=name,
            permissions=permissions,
            expires_at=expires_at,
            rate_limit=rate_limit
        )
        
        return api_key_id, full_key
    
    async def _store_api_key(
        self,
        user_id: str,
        key_hash: str,
        key_prefix: str,
        name: str,
        permissions: list[str],
        expires_at: Optional[datetime],
        rate_limit: int
    ) -> str:
        """API Keyのデータベース保存"""
        
        async with self.db_pool.acquire() as conn:
            api_key_id = await conn.fetchval("""
                INSERT INTO api_keys (
                    user_id, key_hash, key_prefix, name, 
                    permissions, expires_at, rate_limit_per_minute
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
            """, user_id, key_hash, key_prefix, name, 
                permissions, expires_at, rate_limit)
            
            return str(api_key_id)
    
    async def validate_api_key(self, api_key: str) -> Optional[dict]:
        """API Key検証"""
        
        if not api_key.startswith(self.key_prefix):
            return None
        
        # キーハッシュ化
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchrow("""
                SELECT 
                    ak.id, ak.user_id, ak.permissions, 
                    ak.rate_limit_per_minute, ak.usage_count,
                    ak.expires_at, ak.is_active,
                    u.username, u.role, u.is_active as user_active
                FROM api_keys ak
                JOIN users u ON ak.user_id = u.id
                WHERE ak.key_hash = $1
                  AND ak.is_active = true
                  AND u.is_active = true
                  AND (ak.expires_at IS NULL OR ak.expires_at > NOW())
            """, key_hash)
            
            if not result:
                return None
            
            # 使用回数更新
            await self._update_api_key_usage(result["id"])
            
            return {
                "api_key_id": str(result["id"]),
                "user_id": str(result["user_id"]),
                "username": result["username"],
                "role": result["role"],
                "permissions": result["permissions"],
                "rate_limit": result["rate_limit_per_minute"],
                "usage_count": result["usage_count"]
            }
    
    async def _update_api_key_usage(self, api_key_id: str) -> None:
        """API Key使用統計更新"""
        
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE api_keys 
                SET usage_count = usage_count + 1,
                    last_used_at = NOW()
                WHERE id = $1
            """, api_key_id)
    
    async def revoke_api_key(self, api_key_id: str, user_id: str) -> bool:
        """API Key無効化"""
        
        async with self.db_pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE api_keys 
                SET is_active = false, updated_at = NOW()
                WHERE id = $1 AND user_id = $2
            """, api_key_id, user_id)
            
            return result.split()[-1] == "1"  # 1行更新されたかチェック
    
    async def list_user_api_keys(self, user_id: str) -> list[dict]:
        """ユーザーのAPI Key一覧"""
        
        async with self.db_pool.acquire() as conn:
            results = await conn.fetch("""
                SELECT 
                    id, name, key_prefix, permissions,
                    rate_limit_per_minute, usage_count,
                    is_active, expires_at, created_at, last_used_at
                FROM api_keys
                WHERE user_id = $1
                ORDER BY created_at DESC
            """, user_id)
            
            return [
                {
                    "id": str(row["id"]),
                    "name": row["name"],
                    "key_preview": row["key_prefix"],
                    "permissions": row["permissions"],
                    "rate_limit": row["rate_limit_per_minute"],
                    "usage_count": row["usage_count"],
                    "is_active": row["is_active"],
                    "expires_at": row["expires_at"].isoformat() if row["expires_at"] else None,
                    "created_at": row["created_at"].isoformat(),
                    "last_used_at": row["last_used_at"].isoformat() if row["last_used_at"] else None
                }
                for row in results
            ]
```

### 2. レート制限システム

```python
import time
from collections import defaultdict
from typing import Optional

class RateLimiter:
    """レート制限サービス"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.rate_limit_prefix = "rate_limit:"
        self.window_size = 60  # 1分間のウィンドウ
    
    async def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window: Optional[int] = None
    ) -> tuple[bool, dict]:
        """レート制限チェック"""
        
        window = window or self.window_size
        current_time = int(time.time())
        window_start = current_time - window
        
        rate_key = f"{self.rate_limit_prefix}{identifier}"
        
        # Sliding Window Log アルゴリズム
        pipe = self.redis.pipeline()
        
        # 古いエントリを削除
        pipe.zremrangebyscore(rate_key, 0, window_start)
        
        # 現在のリクエスト数を取得
        pipe.zcard(rate_key)
        
        # 現在の時刻を記録
        pipe.zadd(rate_key, {str(current_time): current_time})
        
        # TTL設定
        pipe.expire(rate_key, window)
        
        results = await pipe.execute()
        current_requests = results[1]
        
        # 制限判定
        is_allowed = current_requests < limit
        
        # 残り制限回数計算
        remaining = max(0, limit - current_requests - 1)
        
        # リセット時刻計算
        reset_time = current_time + window
        
        rate_limit_info = {
            "limit": limit,
            "remaining": remaining,
            "reset_time": reset_time,
            "window_size": window
        }
        
        return is_allowed, rate_limit_info
    
    async def get_rate_limit_info(
        self,
        identifier: str,
        limit: int,
        window: Optional[int] = None
    ) -> dict:
        """レート制限情報取得"""
        
        window = window or self.window_size
        current_time = int(time.time())
        window_start = current_time - window
        
        rate_key = f"{self.rate_limit_prefix}{identifier}"
        
        # 現在のウィンドウ内のリクエスト数
        current_requests = await self.redis.zcount(
            rate_key, window_start, current_time
        )
        
        remaining = max(0, limit - current_requests)
        reset_time = current_time + window
        
        return {
            "limit": limit,
            "used": current_requests,
            "remaining": remaining,
            "reset_time": reset_time,
            "window_size": window
        }
```

---

## 👤 RBAC (Role-Based Access Control)

### 1. ロール・権限定義

```python
from enum import Enum
from dataclasses import dataclass
from typing import Set

class Permission(str, Enum):
    """権限定義"""
    
    # ドキュメント関連
    DOCUMENT_READ = "document:read"
    DOCUMENT_WRITE = "document:write"
    DOCUMENT_DELETE = "document:delete"
    
    # 検索関連
    SEARCH_BASIC = "search:basic"
    SEARCH_ADVANCED = "search:advanced"
    SEARCH_ANALYTICS = "search:analytics"
    
    # システム管理
    SYSTEM_STATUS = "system:status"
    SYSTEM_METRICS = "system:metrics"
    SYSTEM_CONFIG = "system:config"
    
    # ユーザー管理
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_DELETE = "user:delete"
    
    # API Key管理
    APIKEY_READ = "apikey:read"
    APIKEY_WRITE = "apikey:write"
    APIKEY_DELETE = "apikey:delete"

@dataclass
class Role:
    """ロール定義"""
    
    name: str
    description: str
    permissions: Set[Permission]
    is_system_role: bool = False

# システムロール定義
SYSTEM_ROLES = {
    "viewer": Role(
        name="viewer",
        description="閲覧専用ユーザー",
        permissions={
            Permission.DOCUMENT_READ,
            Permission.SEARCH_BASIC,
            Permission.APIKEY_READ
        },
        is_system_role=True
    ),
    
    "editor": Role(
        name="editor", 
        description="編集権限ユーザー",
        permissions={
            Permission.DOCUMENT_READ,
            Permission.DOCUMENT_WRITE,
            Permission.SEARCH_BASIC,
            Permission.SEARCH_ADVANCED,
            Permission.APIKEY_READ,
            Permission.APIKEY_WRITE
        },
        is_system_role=True
    ),
    
    "admin": Role(
        name="admin",
        description="管理者",
        permissions={
            Permission.DOCUMENT_READ,
            Permission.DOCUMENT_WRITE,
            Permission.DOCUMENT_DELETE,
            Permission.SEARCH_BASIC,
            Permission.SEARCH_ADVANCED,
            Permission.SEARCH_ANALYTICS,
            Permission.SYSTEM_STATUS,
            Permission.SYSTEM_METRICS,
            Permission.USER_READ,
            Permission.USER_WRITE,
            Permission.APIKEY_READ,
            Permission.APIKEY_WRITE,
            Permission.APIKEY_DELETE
        },
        is_system_role=True
    ),
    
    "super_admin": Role(
        name="super_admin",
        description="スーパー管理者",
        permissions=set(Permission),  # 全権限
        is_system_role=True
    )
}
```

### 2. 権限チェック実装

```python
class AuthorizationService:
    """認可サービス"""
    
    def __init__(self):
        self.roles = SYSTEM_ROLES
    
    def check_permission(
        self,
        user_role: str,
        user_permissions: list[str],
        required_permission: str
    ) -> bool:
        """権限チェック"""
        
        # 個別権限での確認
        if required_permission in user_permissions:
            return True
        
        # ロールベースでの確認
        role = self.roles.get(user_role)
        if role and Permission(required_permission) in role.permissions:
            return True
        
        return False
    
    def get_user_permissions(self, user_role: str) -> Set[str]:
        """ユーザーの全権限取得"""
        
        role = self.roles.get(user_role)
        if role:
            return {perm.value for perm in role.permissions}
        return set()
    
    def can_access_resource(
        self,
        user: dict,
        resource_type: str,
        action: str,
        resource_owner: Optional[str] = None
    ) -> bool:
        """リソースアクセス権限チェック"""
        
        required_permission = f"{resource_type}:{action}"
        
        # 基本権限チェック
        has_permission = self.check_permission(
            user["role"],
            user.get("permissions", []),
            required_permission
        )
        
        if not has_permission:
            return False
        
        # オーナーシップチェック（必要に応じて）
        if resource_owner and action in ["write", "delete"]:
            # リソースの所有者または管理者のみ
            return (
                user["user_id"] == resource_owner or
                user["role"] in ["admin", "super_admin"]
            )
        
        return True

# FastAPI 依存性注入での使用
def require_permission(permission: str):
    """権限要求デコレーター"""
    
    def permission_checker(
        current_user: dict = Depends(get_current_user_or_api_key),
        auth_service: AuthorizationService = Depends(get_auth_service)
    ):
        if not auth_service.check_permission(
            current_user["role"],
            current_user.get("permissions", []),
            permission
        ):
            raise HTTPException(
                status_code=403,
                detail=f"Permission required: {permission}"
            )
        return current_user
    
    return permission_checker

# 使用例
@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    current_user: dict = Depends(require_permission("document:delete"))
):
    # 削除処理...
    pass
```

---

## 🛡️ セキュリティベストプラクティス

### 1. パスワードセキュリティ

```python
import bcrypt
import secrets
import re
from typing import tuple

class PasswordService:
    """パスワード管理サービス"""
    
    def __init__(self):
        self.min_length = 8
        self.max_length = 128
        self.salt_rounds = 12
    
    def validate_password_strength(self, password: str) -> tuple[bool, list[str]]:
        """パスワード強度チェック"""
        
        errors = []
        
        # 長さチェック
        if len(password) < self.min_length:
            errors.append(f"Password must be at least {self.min_length} characters")
        
        if len(password) > self.max_length:
            errors.append(f"Password must be no more than {self.max_length} characters")
        
        # 複雑さチェック
        if not re.search(r"[a-z]", password):
            errors.append("Password must contain lowercase letters")
        
        if not re.search(r"[A-Z]", password):
            errors.append("Password must contain uppercase letters")
        
        if not re.search(r"\d", password):
            errors.append("Password must contain numbers")
        
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            errors.append("Password must contain special characters")
        
        # 一般的なパスワードチェック
        common_passwords = [
            "password", "123456", "123456789", "12345678",
            "qwerty", "abc123", "password123"
        ]
        
        if password.lower() in common_passwords:
            errors.append("Password is too common")
        
        return len(errors) == 0, errors
    
    def hash_password(self, password: str) -> str:
        """パスワードハッシュ化"""
        
        salt = bcrypt.gensalt(rounds=self.salt_rounds)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """パスワード検証"""
        
        return bcrypt.checkpw(
            password.encode('utf-8'),
            hashed.encode('utf-8')
        )
    
    def generate_secure_password(self, length: int = 16) -> str:
        """安全なパスワード生成"""
        
        # 文字セット定義
        lowercase = "abcdefghijklmnopqrstuvwxyz"
        uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        digits = "0123456789"
        special = "!@#$%^&*(),.?\":{}|<>"
        
        # 各カテゴリから最低1文字
        password = [
            secrets.choice(lowercase),
            secrets.choice(uppercase),
            secrets.choice(digits),
            secrets.choice(special)
        ]
        
        # 残りの文字をランダム選択
        all_chars = lowercase + uppercase + digits + special
        for _ in range(length - 4):
            password.append(secrets.choice(all_chars))
        
        # シャッフル
        secrets.SystemRandom().shuffle(password)
        
        return "".join(password)
```

### 2. セキュリティミドルウェア

```python
import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityMiddleware(BaseHTTPMiddleware):
    """セキュリティミドルウェア"""
    
    def __init__(self, app, config: dict):
        super().__init__(app)
        self.config = config
        self.rate_limiter = RateLimiter(redis_client)
    
    async def dispatch(self, request: Request, call_next):
        """リクエスト前後処理"""
        
        start_time = time.time()
        
        # 1. セキュリティヘッダー設定
        response = await call_next(request)
        self._add_security_headers(response)
        
        # 2. レート制限チェック
        client_ip = self._get_client_ip(request)
        is_allowed, rate_info = await self.rate_limiter.check_rate_limit(
            f"ip:{client_ip}",
            self.config.get("global_rate_limit", 1000)
        )
        
        if not is_allowed:
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={
                    "X-RateLimit-Limit": str(rate_info["limit"]),
                    "X-RateLimit-Remaining": str(rate_info["remaining"]),
                    "X-RateLimit-Reset": str(rate_info["reset_time"])
                }
            )
        
        # 3. レスポンスにレート制限情報追加
        response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_info["reset_time"])
        
        # 4. 処理時間記録
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    
    def _add_security_headers(self, response: Response) -> None:
        """セキュリティヘッダー追加"""
        
        # HTTPS強制
        response.headers["Strict-Transport-Security"] = \
            "max-age=31536000; includeSubDomains"
        
        # XSS対策
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # CSP設定
        response.headers["Content-Security-Policy"] = \
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"
        
        # 参照元情報制限
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # 権限ポリシー
        response.headers["Permissions-Policy"] = \
            "geolocation=(), microphone=(), camera=()"
    
    def _get_client_ip(self, request: Request) -> str:
        """クライアントIP取得"""
        
        # プロキシ経由の場合
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # 直接接続の場合
        return request.client.host
```

---

## ❗ よくある落とし穴と対策

### 1. JWT セキュリティ問題

```python
# ❌ 問題: 秘密鍵の不適切な管理
SECRET_KEY = "my-secret-key"  # 固定値・短いキー

# ✅ 対策: 安全な秘密鍵管理
SECRET_KEY = os.getenv("JWT_SECRET_KEY")  # 環境変数から取得
if not SECRET_KEY or len(SECRET_KEY) < 32:
    raise ValueError("JWT_SECRET_KEY must be at least 32 characters")

# 秘密鍵のローテーション対応
class JWTKeyManager:
    def __init__(self):
        self.current_key = os.getenv("JWT_SECRET_KEY")
        self.previous_keys = os.getenv("JWT_PREVIOUS_KEYS", "").split(",")
    
    def get_keys(self) -> list[str]:
        """検証用キー一覧（現在+過去）"""
        keys = [self.current_key]
        keys.extend([k.strip() for k in self.previous_keys if k.strip()])
        return keys
```

### 2. API Key 漏洩対策

```python
# ❌ 問題: プレーンテキストでの保存
def store_api_key_unsafe(api_key: str):
    # DBにそのまま保存 → 漏洩リスク
    db.execute("INSERT INTO api_keys (key) VALUES (?)", api_key)

# ✅ 対策: ハッシュ化保存 + プレフィックス管理
def store_api_key_safe(api_key: str):
    # ハッシュ化して保存
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    key_prefix = api_key[:12] + "..."  # 表示用プレフィックス
    
    db.execute(
        "INSERT INTO api_keys (key_hash, key_prefix) VALUES (?, ?)",
        key_hash, key_prefix
    )

# 定期的なAPI Keyローテーション
async def rotate_api_keys():
    """期限切れAPI Keyの自動無効化"""
    await db.execute("""
        UPDATE api_keys 
        SET is_active = false 
        WHERE expires_at < NOW() AND is_active = true
    """)
```

### 3. 権限昇格脆弱性

```python
# ❌ 問題: 不適切な権限チェック
def update_user_role(user_id: str, new_role: str, current_user: dict):
    # 権限チェックなし → 権限昇格可能
    db.execute("UPDATE users SET role = ? WHERE id = ?", new_role, user_id)

# ✅ 対策: 厳密な権限チェック
def update_user_role_safe(user_id: str, new_role: str, current_user: dict):
    # 管理者権限チェック
    if current_user["role"] not in ["admin", "super_admin"]:
        raise PermissionError("Admin role required")
    
    # 自分より上位ロールへの変更禁止
    role_hierarchy = {"viewer": 1, "editor": 2, "admin": 3, "super_admin": 4}
    
    current_level = role_hierarchy[current_user["role"]]
    target_level = role_hierarchy[new_role]
    
    if target_level >= current_level:
        raise PermissionError("Cannot assign equal or higher role")
    
    # 自分自身のロール変更禁止
    if user_id == current_user["user_id"]:
        raise PermissionError("Cannot modify own role")
    
    db.execute("UPDATE users SET role = ? WHERE id = ?", new_role, user_id)
```

---

## 🎯 理解確認のための設問

### JWT理解
1. JWTクレームで`sub`、`iss`、`aud`各フィールドの役割を説明してください
2. アクセストークンとリフレッシュトークンを分ける理由とその利点を説明してください
3. JWT黒リスト機能が必要な理由と実装上の考慮点を説明してください

### API Key理解
1. API Keyをハッシュ化して保存する理由と、プレフィックス表示の目的を説明してください
2. レート制限でSliding Window Logアルゴリズムを使用する利点を説明してください
3. API Key権限とユーザー権限の違いと使い分けを説明してください

### RBAC理解
1. ロールベースアクセス制御の利点と、権限の粒度設計について説明してください
2. リソースオーナーシップチェックが必要な場面と実装方法を説明してください
3. 権限昇格脆弱性を防ぐための3つの対策を挙げてください

### セキュリティ理解
1. セキュリティミドルウェアで設定される5つのHTTPヘッダーの目的を説明してください
2. パスワード強度チェックで検証すべき6つの要素を挙げてください
3. 秘密鍵ローテーションが必要な理由と実装時の注意点を説明してください

---

## 📚 次のステップ

認証・認可システムを理解できたら、次の学習段階に進んでください：

- **Step07**: エラーハンドリングと監視 - 例外処理・ログ・メトリクス収集
- **Step08**: デプロイメントと運用 - Docker・Kubernetes・CI/CD・監視

堅牢な認証・認可システムは、企業システムの信頼性を決定する重要な要素です。次のステップでは、システムの安定性を支えるエラーハンドリングと監視について学習します。