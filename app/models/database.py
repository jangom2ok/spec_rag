"""PostgreSQL用のデータベースモデル"""

import uuid
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy import JSON, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from sqlalchemy.types import String, TypeDecorator


class UUIDType(TypeDecorator[str]):
    """UUID型のカスタムタイプ"""

    impl = String
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return value
        elif isinstance(value, uuid.UUID):
            return str(value)
        elif isinstance(value, str):
            return value
        return str(value)

    def process_result_value(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return value
        return str(value)


class Base(DeclarativeBase):
    """SQLAlchemy 2.0の基底クラス"""

    pass


class Document(Base):
    """ドキュメントテーブル"""

    __tablename__ = "documents"

    # プライマリキー
    id: Mapped[str] = mapped_column(
        UUIDType(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # 基本情報
    source_type: Mapped[str] = mapped_column(sa.String(50), nullable=False)
    source_id: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    title: Mapped[str] = mapped_column(sa.Text, nullable=False)
    content: Mapped[str] = mapped_column(sa.Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(sa.String(64), nullable=False)

    # オプション情報
    file_type: Mapped[str | None] = mapped_column(sa.String(50), nullable=True)
    language: Mapped[str] = mapped_column(sa.String(10), nullable=False, default="ja")
    status: Mapped[str] = mapped_column(sa.String(20), nullable=False, default="active")

    # タイムスタンプ
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
    )

    # リレーション
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="select",
    )

    # 制約
    __table_args__ = (
        UniqueConstraint("source_type", "source_id", name="uq_documents_source"),
        Index("idx_documents_source", "source_type", "source_id"),
        Index("idx_documents_updated", "updated_at"),
        Index("idx_documents_hash", "content_hash"),
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, title={self.title}, source_type={self.source_type})>"


class DocumentChunk(Base):
    """ドキュメントチャンクテーブル"""

    __tablename__ = "document_chunks"

    # プライマリキー
    id: Mapped[str] = mapped_column(
        UUIDType(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # 外部キー
    document_id: Mapped[str] = mapped_column(
        UUIDType(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )

    # チャンク情報
    chunk_index: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    chunk_type: Mapped[str] = mapped_column(sa.String(20), nullable=False)
    title: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    content: Mapped[str] = mapped_column(sa.Text, nullable=False)
    content_length: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    token_count: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    hierarchy_path: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    chunk_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # タイムスタンプ
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # リレーション
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks",
        lazy="select",
    )

    # 制約
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
        Index("idx_chunks_document", "document_id"),
        Index("idx_chunks_type", "chunk_type"),
        Index("idx_chunks_metadata", "chunk_metadata", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"


class User(Base):
    """ユーザーテーブル"""

    __tablename__ = "users"

    # プライマリキー
    id: Mapped[str] = mapped_column(
        UUIDType(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # 基本情報
    email: Mapped[str] = mapped_column(sa.String(255), nullable=False, unique=True)
    hashed_password: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(sa.String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(sa.Boolean, default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(
        sa.Boolean, default=False, nullable=False
    )

    # 権限・ロール
    role: Mapped[str] = mapped_column(sa.String(50), default="user", nullable=False)
    permissions: Mapped[list[str]] = mapped_column(
        JSON, default=lambda: ["read"], nullable=False
    )

    # タイムスタンプ
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    last_login_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
    )

    # APIキー関連
    api_keys: Mapped[list["APIKey"]] = relationship(
        "APIKey",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select",
    )

    # インデックス
    __table_args__ = (
        Index("idx_users_email", "email"),
        Index("idx_users_role", "role"),
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"


class APIKey(Base):
    """APIキーテーブル"""

    __tablename__ = "api_keys"

    # プライマリキー
    id: Mapped[str] = mapped_column(
        UUIDType(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # 外部キー
    user_id: Mapped[str] = mapped_column(
        UUIDType(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    # APIキー情報
    key_hash: Mapped[str] = mapped_column(sa.String(255), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(sa.String(100), nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(sa.Boolean, default=True, nullable=False)

    # タイムスタンプ
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # リレーション
    user: Mapped["User"] = relationship(
        "User",
        back_populates="api_keys",
        lazy="select",
    )

    # インデックス
    __table_args__ = (
        Index("idx_api_keys_key_hash", "key_hash"),
        Index("idx_api_keys_user", "user_id"),
    )

    def __repr__(self) -> str:
        return f"<APIKey(id={self.id}, name={self.name}, user_id={self.user_id})>"
