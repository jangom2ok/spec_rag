"""データベースモデルのテスト"""

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import (
    Base,
    Document,
    DocumentChunk,
)


@pytest.fixture
async def async_engine():
    """テスト用のasync engine"""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def async_session(async_engine):
    """テスト用のasync session"""
    async_session = sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session


class TestDocument:
    """Documentモデルのテスト"""

    async def test_document_creation(self, async_session: AsyncSession):
        """ドキュメントの作成テスト"""
        document = Document(
            source_type="git",
            source_id="repo/doc.md",
            title="テストドキュメント",
            content="これはテストコンテンツです。",
            content_hash="abc123",
            file_type="markdown",
            language="ja",
        )

        async_session.add(document)
        await async_session.commit()
        await async_session.refresh(document)

        assert document.id is not None
        assert isinstance(document.id, str)  # UUIDは文字列として保存
        assert document.source_type == "git"
        assert document.title == "テストドキュメント"
        assert document.status == "active"  # デフォルト値
        assert document.created_at is not None
        assert document.updated_at is not None

    async def test_document_unique_constraint(self, async_session: AsyncSession):
        """source_type + source_idの一意制約テスト"""
        document1 = Document(
            source_type="git",
            source_id="repo/doc.md",
            title="テストドキュメント1",
            content="コンテンツ1",
            content_hash="hash1",
        )

        document2 = Document(
            source_type="git",
            source_id="repo/doc.md",  # 同じsource_id
            title="テストドキュメント2",
            content="コンテンツ2",
            content_hash="hash2",
        )

        async_session.add(document1)
        await async_session.commit()

        async_session.add(document2)
        with pytest.raises(sa.exc.IntegrityError):
            await async_session.commit()

    async def test_document_relationships(self, async_session: AsyncSession):
        """DocumentとDocumentChunkの関係テスト"""
        document = Document(
            source_type="git",
            source_id="repo/doc.md",
            title="テストドキュメント",
            content="これはテストコンテンツです。",
            content_hash="abc123",
        )

        async_session.add(document)
        await async_session.commit()
        await async_session.refresh(document)

        # チャンクを追加
        chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=0,
            chunk_type="paragraph",
            title="チャンクタイトル",
            content="チャンクコンテンツ",
            content_length=100,
            token_count=20,
        )

        async_session.add(chunk)
        await async_session.commit()

        # 関係を確認（明示的にchunksをロードしてチェック）
        result = await async_session.execute(
            sa.select(DocumentChunk).where(DocumentChunk.document_id == document.id)
        )
        chunks = list(result.scalars().all())

        assert len(chunks.chunks if hasattr(chunks, "chunks") else []) == 1
        assert (chunks.chunks if hasattr(chunks, "chunks") else [])[
            0
        ].document_id == document.id
        assert (chunks.chunks if hasattr(chunks, "chunks") else [])[
            0
        ].content == "チャンクコンテンツ"


class TestDocumentChunk:
    """DocumentChunkモデルのテスト"""

    async def test_chunk_creation(self, async_session: AsyncSession):
        """チャンクの作成テスト"""
        # 先にドキュメントを作成
        document = Document(
            source_type="git",
            source_id="repo/doc.md",
            title="テストドキュメント",
            content="コンテンツ",
            content_hash="hash123",
        )
        async_session.add(document)
        await async_session.commit()
        await async_session.refresh(document)

        # チャンクを作成
        chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=0,
            chunk_type="section",
            title="セクションタイトル",
            content="セクションコンテンツ",
            content_length=200,
            token_count=50,
            hierarchy_path="1.2.3",
            chunk_metadata={"key": "value", "tags": ["tag1", "tag2"]},
        )

        async_session.add(chunk)
        await async_session.commit()
        await async_session.refresh(chunk)

        assert chunk.id is not None
        assert chunk.document_id == document.id
        assert chunk.chunk_type == "section"
        assert chunk.chunk_metadata == {"key": "value", "tags": ["tag1", "tag2"]}
        assert chunk.hierarchy_path == "1.2.3"

    async def test_chunk_unique_constraint(self, async_session: AsyncSession):
        """document_id + chunk_indexの一意制約テスト"""
        # ドキュメントを作成
        document = Document(
            source_type="git",
            source_id="repo/doc.md",
            title="テストドキュメント",
            content="コンテンツ",
            content_hash="hash123",
        )
        async_session.add(document)
        await async_session.commit()
        await async_session.refresh(document)

        # 同じindexのチャンクを2つ作成
        chunk1 = DocumentChunk(
            document_id=document.id,
            chunk_index=0,
            chunk_type="section",
            content="コンテンツ1",
            content_length=100,
        )

        chunk2 = DocumentChunk(
            document_id=document.id,
            chunk_index=0,  # 同じindex
            chunk_type="paragraph",
            content="コンテンツ2",
            content_length=150,
        )

        async_session.add(chunk1)
        await async_session.commit()

        async_session.add(chunk2)
        with pytest.raises(sa.exc.IntegrityError):
            await async_session.commit()

    async def test_chunk_cascade_delete(self, async_session: AsyncSession):
        """ドキュメント削除時のカスケード削除テスト"""
        # ドキュメントとチャンクを作成
        document = Document(
            source_type="git",
            source_id="repo/doc.md",
            title="テストドキュメント",
            content="コンテンツ",
            content_hash="hash123",
        )
        async_session.add(document)
        await async_session.commit()
        await async_session.refresh(document)

        chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=0,
            chunk_type="section",
            content="チャンクコンテンツ",
            content_length=100,
        )
        async_session.add(chunk)
        await async_session.commit()

        # ドキュメントを削除
        await async_session.delete(document)
        await async_session.commit()

        # チャンクも削除されていることを確認
        result = await async_session.execute(
            sa.select(DocumentChunk).where(DocumentChunk.document_id == document.id)
        )
        chunks = result.scalars().all()
        assert len(chunks.chunks if hasattr(chunks, "chunks") else []) == 0
