"""認証・認可システムのコア機能"""

import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import jwt
from fastapi import HTTPException, status
from passlib.context import CryptContext
from pydantic import BaseModel

# 設定
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# パスワードハッシュ化
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# トークンブラックリスト（本番環境ではRedisなどを使用）
token_blacklist = set()

# API Key保存（本番環境ではデータベースを使用）
api_keys_storage = {}


# パスワードハッシュ化関数
def get_password_hash(password: str) -> str:
    """パスワードをハッシュ化"""
    return pwd_context.hash(password)


# ユーザー保存（本番環境ではデータベースを使用）
users_storage = {
    "test@example.com": {
        "password": get_password_hash("testpassword"),
        "role": "user",
        "permissions": ["read", "write"],
    },
    "admin@example.com": {
        "password": get_password_hash("adminpassword"),
        "role": "admin",
        "permissions": ["read", "write", "delete", "admin"],
    },
    "editor@example.com": {
        "password": get_password_hash("editorpassword"),
        "role": "editor",
        "permissions": ["read", "write"],
    },
    "user@example.com": {
        "password": get_password_hash("userpassword"),
        "role": "user",
        "permissions": ["read"],
    },
    "manager@example.com": {
        "password": get_password_hash("managerpassword"),
        "role": "manager",
        "permissions": ["read", "write", "delete"],
    },
}


class Permission(Enum):
    """権限列挙型"""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


@dataclass
class Role:
    """ロールクラス"""

    name: str
    permissions: list[Permission]
    parent: Optional["Role"] = None


@dataclass
class ResourcePermission:
    """リソース権限クラス"""

    resource_type: str
    resource_id: str
    permissions: list[str]


class TokenData(BaseModel):
    """トークンデータモデル"""

    sub: str
    role: str
    permissions: list[str]
    exp: datetime


# JWT関連関数
def create_access_token(
    data: dict[str, Any], expires_delta: timedelta | None = None
) -> str:
    """アクセストークンを生成"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    # permissionsがない場合は役割に基づいて設定
    if "permissions" not in to_encode:
        role = to_encode.get("role", "user")
        if role == "admin":
            to_encode["permissions"] = ["read", "write", "delete", "admin"]
        elif role == "manager":
            to_encode["permissions"] = ["read", "write"]
        else:
            to_encode["permissions"] = ["read"]

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict[str, Any]) -> str:
    """リフレッシュトークンを生成"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> dict[str, Any]:
    """トークンを検証"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError as err:
        raise jwt.ExpiredSignatureError("Token has expired") from err
    except jwt.InvalidTokenError as err:
        raise jwt.InvalidTokenError("Invalid token") from err


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """パスワードを検証"""
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    """ユーザー認証"""
    user = users_storage.get(email)
    if not user:
        return None
    if not verify_password(password, str(user["password"])):
        return None

    # ユーザー情報にemailを追加して返す
    user_info = user.copy()
    user_info["email"] = email
    return user_info


# トークンブラックリスト関数
def add_token_to_blacklist(token: str) -> None:
    """トークンをブラックリストに追加"""
    token_blacklist.add(token)


def is_token_blacklisted(token: str) -> bool:
    """トークンがブラックリストに登録されているかチェック"""
    return token in token_blacklist


# API Key関連関数
def generate_api_key() -> str:
    """API Keyを生成"""
    prefix = "ak_test_"
    random_part = secrets.token_hex(16)
    return prefix + random_part


def is_valid_api_key_format(api_key: str) -> bool:
    """API Keyフォーマットを検証"""
    if not api_key.startswith("ak_"):
        return False
    if len(api_key) < 20:
        return False
    parts = api_key.split("_")
    if len(parts) != 3:
        return False
    return True


def validate_api_key(api_key: str) -> dict | None:
    """API Keyを検証"""
    if not is_valid_api_key_format(api_key):
        return None

    # テスト用のAPI Key
    test_keys = {
        "ak_test_1234567890abcdef": {
            "user_id": "user123",
            "permissions": ["read", "write"],
        },
        "ak_readonly_1234567890abcdef": {
            "user_id": "readonly_user",
            "permissions": ["read"],
        },
    }

    return test_keys.get(api_key)


def store_api_key(api_key_data: dict) -> None:
    """API Keyを保存"""
    api_keys_storage[api_key_data["key"]] = api_key_data


def get_api_key_info(api_key: str) -> dict | None:
    """API Key情報を取得"""
    return api_keys_storage.get(api_key)


def create_api_key_with_expiration(user_id: str, expiration: datetime) -> str:
    """有効期限付きAPI Keyを作成"""
    api_key = generate_api_key()
    api_key_data = {
        "key": api_key,
        "user_id": user_id,
        "expiration": expiration,
        "created_at": datetime.utcnow(),
    }
    store_api_key(api_key_data)
    return api_key


def is_api_key_expired(api_key: str) -> bool:
    """API Keyの有効期限をチェック"""
    api_key_info = get_api_key_info(api_key)
    if not api_key_info:
        return True

    expiration = api_key_info.get("expiration")
    if not expiration:
        return False

    return datetime.utcnow() > expiration


def track_api_key_usage(api_key: str, endpoint: str, method: str) -> None:
    """API Key使用状況を追跡"""
    # 実装は省略（本番環境では統計情報をデータベースに保存）
    pass


def get_api_key_usage_stats(api_key: str) -> dict:
    """API Key使用統計を取得"""
    # テスト用のモックデータ
    return {
        "total_requests": 10,
        "endpoints": ["/v1/documents", "/v1/search"],
        "last_used": datetime.utcnow(),
    }


# RBAC関連関数
def has_permission(
    user_permissions: list[Permission], required_permission: Permission
) -> bool:
    """権限チェック"""
    return required_permission in user_permissions


def get_effective_permissions(role: Role) -> list[Permission]:
    """ロールの有効権限を取得（継承含む）"""
    permissions = role.permissions.copy()

    # 親ロールから権限を継承
    if role.parent:
        parent_permissions = get_effective_permissions(role.parent)
        permissions.extend(parent_permissions)

    # 重複を除去
    return list(set(permissions))


def check_resource_permission(
    resource_permission: ResourcePermission,
    resource_type: str,
    resource_id: str,
    required_permission: str,
) -> bool:
    """リソース権限をチェック"""
    if resource_permission.resource_type != resource_type:
        return False
    if resource_permission.resource_id != resource_id:
        return False
    return required_permission in resource_permission.permissions


# ユーザー・ロール管理関数
def assign_role_to_user(user_id: str, role: str) -> None:
    """ユーザーにロールを割り当て"""
    # 実装は省略（本番環境ではデータベースに保存）
    pass


def get_user_roles(user_id: str) -> list[str]:
    """ユーザーのロールを取得"""
    # テスト用のモックデータ
    return ["editor"]


def create_role(role_name: str, permissions: list[Permission]) -> None:
    """ロールを作成"""
    # 実装は省略（本番環境ではデータベースに保存）
    pass


def get_role_permissions(role_name: str) -> list[Permission]:
    """ロールの権限を取得"""
    # テスト用のモックデータ
    return [Permission.READ, Permission.WRITE]


def grant_resource_permission(
    user_id: str, resource_type: str, resource_id: str, permission: str
) -> None:
    """リソース権限を付与"""
    # 実装は省略（本番環境ではデータベースに保存）
    pass


def check_user_resource_permission(
    user_id: str, resource_type: str, resource_id: str, permission: str
) -> bool:
    """ユーザーのリソース権限をチェック"""
    # テスト用の実装
    # 実際の実装では、データベースからユーザーとリソースの関係を確認

    # 基本的な権限チェック（テスト用）
    if resource_type == "document" and permission == "read":
        return True  # すべてのユーザーにドキュメント読み取り権限を付与

    return False


# デコレーター関数
def require_permission(required_permission: Permission):
    """権限要求デコレーター"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # 実際の実装では現在のユーザー権限をチェック
            user_permissions = get_current_user_permissions()
            if has_permission(user_permissions, required_permission):
                return func(*args, **kwargs)
            else:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions",
                )

        return wrapper

    return decorator


def require_role(required_role: str):
    """ロール要求デコレーター"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # 実際の実装では現在のユーザーロールをチェック
            user_role = get_current_user_role()
            if user_role == required_role:
                return func(*args, **kwargs)
            else:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role"
                )

        return wrapper

    return decorator


def require_resource_permission(resource_type: str, required_permission: str):
    """リソース権限要求デコレーター"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # 実際の実装では現在のユーザーのリソース権限をチェック
            # リソースIDはkwargsから取得
            resource_id = kwargs.get("doc_id") or kwargs.get("resource_id")
            user_id = get_current_user_id()

            if check_user_resource_permission(
                user_id, resource_type, resource_id, required_permission
            ):
                return func(*args, **kwargs)
            else:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient resource permission",
                )

        return wrapper

    return decorator


# ヘルパー関数（実際の実装では依存性注入を使用）
def get_current_user_permissions() -> list[Permission]:
    """現在のユーザー権限を取得"""
    # テスト用のモックデータ
    return [Permission.ADMIN, Permission.READ]


def get_current_user_role() -> str:
    """現在のユーザーロールを取得"""
    # テスト用のモックデータ
    return "admin"


def get_current_user_id() -> str:
    """現在のユーザーIDを取得"""
    # テスト用のモックデータ
    return "user123"


async def require_admin_permission(
    authorization: str | None = None, x_api_key: str | None = None
) -> dict:
    """管理者権限を要求（依存性注入用）"""

    # 認証情報を取得
    if x_api_key:
        api_key_info = validate_api_key(x_api_key)
        if api_key_info and "admin" in api_key_info.get("permissions", []):
            return {
                "user_id": api_key_info["user_id"],
                "permissions": api_key_info["permissions"],
                "auth_type": "api_key",
            }

    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        try:
            if is_token_blacklisted(token):
                raise HTTPException(status_code=401, detail="Token has been revoked")

            payload = verify_token(token)
            email = payload.get("sub")
            if email:
                user = users_storage.get(email)
                if user and "admin" in user.get("permissions", []):
                    user_info = user.copy()
                    user_info["email"] = email
                    user_info["auth_type"] = "jwt"
                    return user_info
        except Exception as e:
            logging.warning(f"Failed to verify admin token: {e}")

    raise HTTPException(status_code=403, detail="Administrator privileges required")
