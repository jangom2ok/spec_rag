"""認証エンドポイント"""

from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

from app.core.auth import (
    add_token_to_blacklist,
    create_access_token,
    create_refresh_token,
    is_token_blacklisted,
    verify_token,
)
from app.services.auth_service import AuthService, get_auth_service

# パスワードハッシュ化
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Temporary in-memory storage (will be removed)
users_storage: dict[str, dict[str, Any]] = {}
api_keys_storage: dict[str, dict[str, Any]] = {}


def get_password_hash(password: str) -> str:
    """パスワードをハッシュ化"""
    hash_result: str = pwd_context.hash(password)
    return hash_result


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """パスワードを検証"""
    result: bool = pwd_context.verify(plain_password, hashed_password)
    return result


def authenticate_user(email: str, password: str) -> dict | None:
    """ユーザーを認証（一時的な実装）"""
    user = users_storage.get(email)
    if not user:
        return None
    if not verify_password(password, user["password"]):
        return None
    user_info = user.copy()
    user_info["email"] = email
    return user_info


def generate_api_key() -> str:
    """APIキーを生成"""
    import secrets

    return secrets.token_urlsafe(32)


def store_api_key(api_key_info: dict) -> None:
    """APIキー情報を保存"""
    api_keys_storage[api_key_info["key"]] = api_key_info


router = APIRouter(prefix="/v1/auth", tags=["authentication"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/login")


# Pydanticモデル定義
class UserRegister(BaseModel):
    """ユーザー登録モデル"""

    email: EmailStr
    password: str
    full_name: str | None = None
    role: str = "user"


class TokenResponse(BaseModel):
    """トークンレスポンスモデル"""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"  # noqa: S105
    expires_in: int = 1800  # 30分


class RefreshTokenRequest(BaseModel):
    """リフレッシュトークンリクエストモデル"""

    refresh_token: str


class UserProfile(BaseModel):
    """ユーザープロファイルモデル"""

    email: str
    role: str
    permissions: list[str]


class APIKeyCreate(BaseModel):
    """API Key作成リクエストモデル"""

    name: str
    permissions: list[str]


class APIKeyResponse(BaseModel):
    """API Keyレスポンスモデル"""

    id: str
    api_key: str
    name: str
    permissions: list[str]
    created_at: datetime


class APIKeyList(BaseModel):
    """API Key一覧レスポンスモデル"""

    api_keys: list[dict]


class MessageResponse(BaseModel):
    """メッセージレスポンスモデル"""

    message: str


# 依存性注入関数
class AuthHTTPError(HTTPException):
    """認証エラー例外"""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=message)
        self.error_code = "AUTHENTICATION_ERROR"
        self.error_type = "authentication"


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> dict:
    """現在のユーザーを取得"""
    try:
        # トークンブラックリストチェック
        if is_token_blacklisted(token):
            raise AuthHTTPError("Token has been revoked")

        payload = verify_token(token)
        email = payload.get("sub")
        if email is None:
            raise AuthHTTPError("Could not validate credentials")

        # データベースからユーザーを取得
        user = await auth_service.get_user_by_email(email)
        if user is None:
            raise AuthHTTPError("User not found")

        # ユーザー情報を辞書形式で返す
        return {
            "email": user.email,
            "role": user.role,
            "permissions": user.permissions,
            "id": user.id,
        }
    except AuthHTTPError:
        raise  # 認証エラーはそのまま再発生
    except Exception as err:
        raise AuthHTTPError("Could not validate credentials") from err


# 認証エンドポイント
@router.post(
    "/register", response_model=MessageResponse, status_code=status.HTTP_201_CREATED
)
async def register_user(
    user_data: UserRegister, auth_service: AuthService = Depends(get_auth_service)
) -> MessageResponse:
    """ユーザー登録"""
    try:
        # データベースにユーザーを作成
        await auth_service.create_user(
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
            role=user_data.role,
        )
        return MessageResponse(message="User registered successfully")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user",
        ) from e


@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends(get_auth_service),
):
    """ログイン"""
    # データベースでユーザーを認証
    user = await auth_service.authenticate_user(form_data.username, form_data.password)
    if not user:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": {
                    "code": "AUTHENTICATION_ERROR",
                    "message": "Incorrect email or password",
                    "type": "authentication",
                }
            },
        )

    # トークンを生成
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={
            "sub": user.email,
            "role": user.role,
            "permissions": user.permissions,
        },
        expires_delta=access_token_expires,
    )

    refresh_token = create_refresh_token(data={"sub": user.email})

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",  # noqa: S106
        expires_in=1800,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest, auth_service: AuthService = Depends(get_auth_service)
) -> TokenResponse:
    """トークンリフレッシュ"""
    try:
        payload = verify_token(request.refresh_token)
        email = payload.get("sub")

        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
            )

        # データベースからユーザーを取得
        user = await auth_service.get_user_by_email(email)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
            )

        # 新しいアクセストークンを生成
        access_token_expires = timedelta(minutes=30)
        access_token = create_access_token(
            data={
                "sub": user.email,
                "role": user.role,
                "permissions": user.permissions,
            },
            expires_delta=access_token_expires,
        )

        return TokenResponse(
            access_token=access_token,
            refresh_token=request.refresh_token,  # リフレッシュトークンは再利用
            token_type="bearer",  # noqa: S106
            expires_in=1800,
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        ) from err


@router.post("/logout", response_model=MessageResponse)
async def logout(
    current_user: dict = Depends(get_current_user),
    token: str = Depends(oauth2_scheme),
):
    """ログアウト"""
    # トークンをブラックリストに追加
    add_token_to_blacklist(token)

    return MessageResponse(message="Successfully logged out")


@router.get("/me", response_model=UserProfile)
async def get_user_profile(
    current_user: dict = Depends(get_current_user),
):
    """ユーザープロファイル取得"""
    return UserProfile(
        email=current_user["email"],
        role=current_user["role"],
        permissions=current_user["permissions"],
    )


# API Key管理エンドポイント
@router.post(
    "/api-keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED
)
async def create_api_key(
    api_key_data: APIKeyCreate,
    current_user: dict = Depends(get_current_user),
):
    """API Key作成"""
    # 管理者権限をチェック（API Key認証の場合は権限リストを直接チェック）
    user_permissions = list(current_user.get("permissions", []))
    if "admin" not in user_permissions and "write" not in user_permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or write permission required",
        )

    # API Keyを生成
    api_key = generate_api_key()

    # API Key情報を保存
    api_key_info = {
        "id": f"ak_{len(api_keys_storage) + 1}",
        "key": api_key,
        "name": api_key_data.name,
        "permissions": api_key_data.permissions,
        "user_id": current_user["email"],
        "created_at": datetime.utcnow(),
    }

    store_api_key(api_key_info)

    return APIKeyResponse(
        id=api_key_info["id"],
        api_key=api_key,
        name=api_key_info["name"],
        permissions=api_key_info["permissions"],
        created_at=api_key_info["created_at"],
    )


@router.get("/api-keys", response_model=APIKeyList)
async def list_api_keys(current_user: dict = Depends(get_current_user)):
    """API Key一覧取得"""
    # ユーザーのAPI Keyのみを返す
    user_api_keys = []
    for key, info in api_keys_storage.items():
        if info.get("user_id") == current_user["email"]:
            # セキュリティのため、実際のキーは隠す
            safe_info = info.copy()
            safe_info["key"] = f"{key[:10]}...{key[-4:]}"
            user_api_keys.append(safe_info)

    return APIKeyList(api_keys=user_api_keys)


@router.delete("/api-keys/{api_key_id}", response_model=MessageResponse)
async def revoke_api_key(
    api_key_id: str, current_user: dict = Depends(get_current_user)
):
    """API Key無効化"""
    # API Keyを検索
    api_key_to_revoke = None
    for key, info in api_keys_storage.items():
        if info.get("id") == api_key_id:
            api_key_to_revoke = key
            break

    if not api_key_to_revoke:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
        )

    # 所有権をチェック
    api_key_info = api_keys_storage[api_key_to_revoke]
    if api_key_info.get("user_id") != current_user[
        "email"
    ] and "admin" not in current_user.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Permission denied"
        )

    # API Keyを削除
    del api_keys_storage[api_key_to_revoke]

    return MessageResponse(message="API key revoked successfully")


# 管理者エンドポイント用の別ルーター
admin_router = APIRouter(prefix="/v1/admin", tags=["admin"])


@admin_router.get("/users")
async def list_users(current_user: dict = Depends(get_current_user)):
    """ユーザー一覧取得（管理者のみ）"""
    if "admin" not in current_user.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin permission required"
        )

    users = []
    for email, user_info in users_storage.items():
        users.append(
            {
                "email": email,
                "role": user_info["role"],
                "permissions": user_info["permissions"],
            }
        )

    return {"users": users}


@admin_router.post("/users/roles")
async def assign_user_role(
    role_data: dict, current_user: dict = Depends(get_current_user)
):
    """ユーザーロール割り当て（管理者のみ）"""
    if "admin" not in current_user.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin permission required"
        )

    # user_id = role_data.get("user_id")
    # role = role_data.get("role")

    # 実装は省略（実際にはデータベースを更新）

    return MessageResponse(message="Role assigned successfully")


@admin_router.put("/users/role")
async def change_user_role(
    role_change: dict, current_user: dict = Depends(get_current_user)
):
    """ユーザーロール変更（管理者のみ）"""
    if "admin" not in current_user.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin permission required"
        )

    user_email = role_change.get("user_email")
    new_role = role_change.get("role")

    if not user_email or not new_role:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both user_email and role are required",
        )

    if user_email not in users_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # ロールに応じた権限を設定
    if new_role == "admin":
        permissions = ["read", "write", "delete", "admin"]
    elif new_role == "editor":
        permissions = ["read", "write"]
    else:
        permissions = ["read"]

    # ユーザー情報を更新
    users_storage[user_email]["role"] = new_role
    users_storage[user_email]["permissions"] = list(permissions)  # 明示的にlistに変換

    return MessageResponse(message="User role updated successfully")


@admin_router.get("/team")
async def get_team_info(current_user: dict = Depends(get_current_user)):
    """チーム情報取得（マネージャー以上）"""
    user_role = current_user.get("role")
    if user_role not in ["manager", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Manager or admin permission required",
        )

    return {"team": "development", "members": 5}
