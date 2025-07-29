"""認証サービス - データベースを使用したユーザー管理"""

import secrets
from datetime import datetime, timedelta
from typing import Optional, cast

from fastapi import Depends, HTTPException, status
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_async_db
from app.models.database import APIKey, User

# パスワードハッシュ化
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """認証サービス"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_user(
        self,
        email: str,
        password: str,
        full_name: str | None = None,
        role: str = "user",
    ) -> User:
        """新規ユーザーを作成"""
        # 既存ユーザーのチェック
        result = await self.db.execute(select(User).where(User.email == email))
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User already exists",
            )

        # パスワードをハッシュ化
        hashed_password = pwd_context.hash(password)

        # ユーザーを作成
        user = User(
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            role=role,
            permissions=self._get_default_permissions(role),
        )
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def authenticate_user(self, email: str, password: str) -> User | None:
        """ユーザーを認証"""
        result = await self.db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if not user or not pwd_context.verify(password, user.hashed_password):
            return None

        # 最終ログイン時刻を更新
        user.last_login_at = datetime.utcnow()
        await self.db.commit()

        return cast(User, user)

    async def get_user_by_email(self, email: str) -> User | None:
        """メールアドレスでユーザーを取得"""
        result = await self.db.execute(select(User).where(User.email == email))
        return cast(User | None, result.scalar_one_or_none())

    async def get_user_by_id(self, user_id: str) -> User | None:
        """IDでユーザーを取得"""
        result = await self.db.execute(select(User).where(User.id == user_id))
        return cast(User | None, result.scalar_one_or_none())

    async def create_api_key(
        self, user_id: str, name: str, expires_days: int | None = None
    ) -> tuple[str, APIKey]:
        """APIキーを作成"""
        # APIキーを生成
        api_key = secrets.token_urlsafe(32)
        key_hash = pwd_context.hash(api_key)

        # 有効期限を設定
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)

        # データベースに保存
        db_api_key = APIKey(
            user_id=user_id,
            key_hash=key_hash,
            name=name,
            expires_at=expires_at,
        )
        self.db.add(db_api_key)
        await self.db.commit()
        await self.db.refresh(db_api_key)

        # 実際のAPIキーとデータベースオブジェクトを返す
        return api_key, db_api_key

    async def verify_api_key(self, api_key: str) -> User | None:
        """APIキーを検証してユーザーを返す"""
        # すべてのアクティブなAPIキーを取得
        result = await self.db.execute(select(APIKey).where(APIKey.is_active.is_(True)))
        api_keys = result.scalars().all()

        # APIキーをハッシュと比較
        for db_key in api_keys:
            if pwd_context.verify(api_key, db_key.key_hash):
                # 有効期限をチェック
                if db_key.expires_at and db_key.expires_at < datetime.utcnow():
                    continue

                # 最終使用時刻を更新
                db_key.last_used_at = datetime.utcnow()
                await self.db.commit()

                # ユーザーを取得
                return await self.get_user_by_id(db_key.user_id)

        return None

    async def revoke_api_key(self, api_key_id: str, user_id: str) -> bool:
        """APIキーを無効化"""
        result = await self.db.execute(
            select(APIKey).where(
                APIKey.id == api_key_id,
                APIKey.user_id == user_id,
            )
        )
        api_key = result.scalar_one_or_none()

        if api_key:
            api_key.is_active = False
            await self.db.commit()
            return True
        return False

    async def list_api_keys(self, user_id: str) -> list[APIKey]:
        """ユーザーのAPIキー一覧を取得"""
        result = await self.db.execute(
            select(APIKey).where(
                APIKey.user_id == user_id,
                APIKey.is_active.is_(True),
            )
        )
        return list(result.scalars().all())

    def _get_default_permissions(self, role: str) -> list[str]:
        """ロールに基づくデフォルト権限を取得"""
        permissions_map = {
            "user": ["read", "write"],
            "admin": ["read", "write", "delete", "admin"],
            "viewer": ["read"],
        }
        return permissions_map.get(role, ["read"])


# 依存性注入用のヘルパー関数
async def get_auth_service(
    db: AsyncSession = Depends(get_async_db),
) -> AuthService:
    """AuthServiceのインスタンスを取得"""
    return AuthService(db)
