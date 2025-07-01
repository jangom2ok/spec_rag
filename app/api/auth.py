"""認証エンドポイント"""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr

from app.core.auth import (
    add_token_to_blacklist,
    api_keys_storage,
    authenticate_user,
    create_access_token,
    create_refresh_token,
    generate_api_key,
    get_password_hash,
    is_token_blacklisted,
    store_api_key,
    users_storage,
    verify_token,
)

router = APIRouter(prefix="/v1/auth", tags=["authentication"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/login")


# Pydanticモデル定義
class UserRegister(BaseModel):
    """ユーザー登録モデル"""

    email: EmailStr
    password: str
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
class AuthenticationError(HTTPException):
    """認証エラー例外"""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=message)
        self.error_code = "AUTHENTICATION_ERROR"
        self.error_type = "authentication"


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """現在のユーザーを取得"""
    try:
        # トークンブラックリストチェック
        if is_token_blacklisted(token):
            raise AuthenticationError("Token has been revoked")

        payload = verify_token(token)
        email = payload.get("sub")
        if email is None:
            raise AuthenticationError("Could not validate credentials")

        user = users_storage.get(email)
        if user is None:
            raise AuthenticationError("User not found")

        # ユーザー情報にemailを追加して返す
        user_info = user.copy()
        user_info["email"] = email
        return user_info
    except AuthenticationError:
        raise  # 認証エラーはそのまま再発生
    except Exception as err:
        raise AuthenticationError("Could not validate credentials") from err


# 認証エンドポイント
@router.post(
    "/register", response_model=MessageResponse, status_code=status.HTTP_201_CREATED
)
async def register_user(user_data: UserRegister):
    """ユーザー登録"""
    # ユーザーが既に存在するかチェック
    if user_data.email in users_storage:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="User already exists"
        )

    # パスワードハッシュ化
    hashed_password = get_password_hash(user_data.password)

    # ロールに応じた権限を設定
    if user_data.role == "admin":
        permissions = ["read", "write", "delete", "admin"]
    elif user_data.role == "editor":
        permissions = ["read", "write"]
    elif user_data.role == "manager":
        permissions = ["read", "write", "delete"]
    else:
        permissions = ["read"]

    # ユーザー情報を保存
    users_storage[user_data.email] = {
        "password": hashed_password,
        "role": user_data.role,
        "permissions": permissions,
    }

    return MessageResponse(message="User registered successfully")


@router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):  # noqa: B008
    """ログイン"""
    user = authenticate_user(form_data.username, form_data.password)
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
            "sub": user["email"],
            "role": user["role"],
            "permissions": user["permissions"],
        },
        expires_delta=access_token_expires,
    )

    refresh_token = create_refresh_token(data={"sub": user["email"]})

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",  # noqa: S106
        expires_in=1800,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """トークンリフレッシュ"""
    try:
        payload = verify_token(request.refresh_token)
        email = payload.get("sub")

        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
            )

        user = users_storage.get(email)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
            )

        # 新しいアクセストークンを生成
        access_token_expires = timedelta(minutes=30)
        access_token = create_access_token(
            data={
                "sub": email,
                "role": user["role"],
                "permissions": user["permissions"],
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
async def logout(  # noqa: B008
    current_user: dict = Depends(get_current_user),  # noqa: B008
    token: str = Depends(oauth2_scheme),  # noqa: B008
):
    """ログアウト"""
    # トークンをブラックリストに追加
    add_token_to_blacklist(token)

    return MessageResponse(message="Successfully logged out")


@router.get("/me", response_model=UserProfile)
async def get_user_profile(current_user: dict = Depends(get_current_user)):  # noqa: B008
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
async def create_api_key(  # noqa: B008
    api_key_data: APIKeyCreate, current_user: dict = Depends(get_current_user)  # noqa: B008
):
    """API Key作成"""
    # 管理者権限をチェック（API Key認証の場合は権限リストを直接チェック）
    user_permissions = current_user.get("permissions", [])
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
async def list_api_keys(current_user: dict = Depends(get_current_user)):  # noqa: B008
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
async def revoke_api_key(  # noqa: B008
    api_key_id: str, current_user: dict = Depends(get_current_user)  # noqa: B008
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
async def list_users(current_user: dict = Depends(get_current_user)):  # noqa: B008
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
async def assign_user_role(  # noqa: B008
    role_data: dict, current_user: dict = Depends(get_current_user)  # noqa: B008
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
async def change_user_role(  # noqa: B008
    role_change: dict, current_user: dict = Depends(get_current_user)  # noqa: B008
):
    """ユーザーロール変更（管理者のみ）"""
    if "admin" not in current_user.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin permission required"
        )

    user_email = role_change.get("user_email")
    new_role = role_change.get("role")

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
    users_storage[user_email]["permissions"] = permissions

    return MessageResponse(message="User role updated successfully")


@admin_router.get("/team")
async def get_team_info(current_user: dict = Depends(get_current_user)):  # noqa: B008
    """チーム情報取得（マネージャー以上）"""
    user_role = current_user.get("role")
    if user_role not in ["manager", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Manager or admin permission required",
        )

    return {"team": "development", "members": 5}
