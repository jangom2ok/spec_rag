"""リポジトリパターンによるCRUD操作のテスト"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.database import Base, Document
from app.repositories.document_repository import DocumentRepository


@pytest.fixture
async def async_engine():
    """テスト用のasync engine"""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def async_session(async_engine):
    """テスト用のasync session"""
    async_session = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session


class TestDocumentRepository:
    """ドキュメントリポジトリのテスト"""

    @pytest.fixture
    def document_repo(self, async_session):
        """ドキュメントリポジトリ"""
        return DocumentRepository(async_session)

    async def test_create_document(self, document_repo: DocumentRepository):
        """ドキュメント作成のテスト"""
        document = Document(
            source_type="git",
            source_id="repo/test.md",
            title="テストドキュメント",
            content="これはテストコンテンツです。",
            content_hash="abc123hash",
            file_type="markdown",
            language="ja",
        )

        created_doc = await document_repo.create(document)

        assert created_doc.id is not None
        assert created_doc.source_type == "git"
        assert created_doc.title == "テストドキュメント"
        assert created_doc.status == "active"
        assert created_doc.created_at is not None

    async def test_get_document_by_id(self, document_repo: DocumentRepository):
        """ID指定でのドキュメント取得テスト"""
        document = Document(
            source_type="git",
            source_id="repo/test.md",
            title="テストドキュメント",
            content="コンテンツ",
            content_hash="hash123",
        )
        created_doc = await document_repo.create(document)

        retrieved_doc = await document_repo.get_by_id(created_doc.id)

        assert retrieved_doc is not None
        assert retrieved_doc.id == created_doc.id
        assert retrieved_doc.title == "テストドキュメント"

    async def test_update_document(self, document_repo: DocumentRepository):
        """ドキュメント更新のテスト"""
        document = Document(
            source_type="git",
            source_id="repo/test.md",
            title="元のタイトル",
            content="元のコンテンツ",
            content_hash="original_hash",
        )
        created_doc = await document_repo.create(document)

        created_doc.title = "更新されたタイトル"
        created_doc.content = "更新されたコンテンツ"

        updated_doc = await document_repo.update(created_doc)

        assert updated_doc.title == "更新されたタイトル"
        assert updated_doc.content == "更新されたコンテンツ"

    async def test_delete_document(self, document_repo: DocumentRepository):
        """ドキュメント削除のテスト"""
        document = Document(
            source_type="git",
            source_id="repo/delete_test.md",
            title="削除テスト",
            content="削除予定",
            content_hash="delete_hash",
        )
        created_doc = await document_repo.create(document)

        deleted = await document_repo.delete(created_doc.id)
        assert deleted is True

        retrieved_doc = await document_repo.get_by_id(created_doc.id)
        assert retrieved_doc is None
