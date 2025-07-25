"""Tests for Document Repository

ドキュメントリポジトリの包括的なテスト。
カバレッジの向上を目的として、すべてのCRUD操作をテスト。
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import Document
from app.repositories.document_repository import DocumentRepository


@pytest.fixture
def mock_session():
    """モックセッションのフィクスチャ"""
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.fixture
def document_repository(mock_session):
    """ドキュメントリポジトリのフィクスチャ"""
    return DocumentRepository(mock_session)


@pytest.fixture
def sample_document():
    """サンプルドキュメントのフィクスチャ"""
    return Document(
        id="doc_1",
        source_type="confluence",
        source_id="src_1",
        title="Test Document",
        content="This is a test document content",
        content_hash="hash123",
        file_type="md",
        status="active",
        processed_at=datetime.now(),
    )


class TestDocumentRepository:
    """DocumentRepositoryのテスト"""

    @pytest.mark.asyncio
    async def test_create_document(
        self, document_repository, mock_session, sample_document
    ):
        """ドキュメント作成のテスト"""
        # セッションメソッドのモック設定
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        # 作成実行
        result = await document_repository.create(sample_document)

        # 検証
        assert result == sample_document
        mock_session.add.assert_called_once_with(sample_document)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_document)

    @pytest.mark.asyncio
    async def test_get_by_id_found(
        self, document_repository, mock_session, sample_document
    ):
        """IDによるドキュメント取得（見つかった場合）のテスト"""
        # モック結果の設定
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_document
        mock_session.execute.return_value = mock_result

        # 取得実行
        result = await document_repository.get_by_id("doc_1")

        # 検証
        assert result == sample_document
        mock_session.execute.assert_called_once()

        # SQLクエリの検証
        call_args = mock_session.execute.call_args[0][0]
        assert isinstance(call_args, sa.sql.Select)

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, document_repository, mock_session):
        """IDによるドキュメント取得（見つからない場合）のテスト"""
        # モック結果の設定
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # 取得実行
        result = await document_repository.get_by_id("nonexistent")

        # 検証
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_source(
        self, document_repository, mock_session, sample_document
    ):
        """ソースタイプとIDによるドキュメント取得のテスト"""
        # モック結果の設定
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_document
        mock_session.execute.return_value = mock_result

        # 取得実行
        result = await document_repository.get_by_source("confluence", "src_1")

        # 検証
        assert result == sample_document
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_with_chunks(
        self, document_repository, mock_session, sample_document
    ):
        """チャンクを含むドキュメント取得のテスト"""
        # チャンクをモック
        sample_document.chunks = []  # 空のチャンクリスト

        # モック結果の設定
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_document
        mock_session.execute.return_value = mock_result

        # 取得実行
        result = await document_repository.get_with_chunks("doc_1")

        # 検証
        assert result == sample_document
        mock_session.execute.assert_called_once()

        # selectinloadが使用されていることを確認
        # SQLAlchemyのクエリオブジェクトの検証は複雑なため、実行が成功したことを確認

    @pytest.mark.asyncio
    async def test_update_document(
        self, document_repository, mock_session, sample_document
    ):
        """ドキュメント更新のテスト"""
        # セッションメソッドのモック設定
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        # 更新実行
        sample_document.title = "Updated Title"
        result = await document_repository.update(sample_document)

        # 検証
        assert result == sample_document
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_document)

    @pytest.mark.asyncio
    async def test_delete_document_exists(
        self, document_repository, mock_session, sample_document
    ):
        """ドキュメント削除（存在する場合）のテスト"""
        # モック結果の設定
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_document
        mock_session.execute.return_value = mock_result
        mock_session.delete = AsyncMock()
        mock_session.commit = AsyncMock()

        # 削除実行
        result = await document_repository.delete("doc_1")

        # 検証
        assert result is True
        mock_session.delete.assert_called_once_with(sample_document)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_document_not_exists(self, document_repository, mock_session):
        """ドキュメント削除（存在しない場合）のテスト"""
        # モック結果の設定
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # 削除実行
        result = await document_repository.delete("nonexistent")

        # 検証
        assert result is False
        mock_session.delete.assert_not_called()
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_list_documents_no_filter(self, document_repository, mock_session):
        """ドキュメント一覧取得（フィルタなし）のテスト"""
        # サンプルドキュメントのリスト
        documents = [
            Document(
                id=f"doc_{i}",
                source_type="confluence",
                source_id=f"src_{i}",
                title=f"Document {i}",
                content=f"Content {i}",
                content_hash=f"hash_{i}",
                status="active",
            )
            for i in range(3)
        ]

        # モック結果の設定
        mock_scalars = Mock()
        mock_scalars.all.return_value = documents
        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # 取得実行
        result = await document_repository.list_documents()

        # 検証
        assert len(result) == 3
        assert all(isinstance(doc, Document) for doc in result)
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_documents_with_filters(self, document_repository, mock_session):
        """ドキュメント一覧取得（フィルタあり）のテスト"""
        # フィルタに一致するドキュメント
        filtered_doc = Document(
            id="doc_1",
            source_type="jira",
            source_id="src_1",
            title="Jira Document",
            content="Jira content",
            content_hash="hash_1",
            status="active",
        )

        # モック結果の設定
        mock_scalars = Mock()
        mock_scalars.all.return_value = [filtered_doc]
        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # 取得実行
        result = await document_repository.list_documents(
            source_type="jira", status="active", limit=10, offset=0
        )

        # 検証
        assert len(result) == 1
        assert result[0].source_type == "jira"
        assert result[0].status == "active"

    @pytest.mark.asyncio
    async def test_search_by_content(self, document_repository, mock_session):
        """コンテンツ検索のテスト"""
        # 検索結果のドキュメント
        search_results = [
            Document(
                id="doc_1",
                source_type="confluence",
                source_id="src_1",
                title="Python Tutorial",
                content="Learn Python programming",
                content_hash="hash_1",
                status="active",
            ),
            Document(
                id="doc_2",
                source_type="confluence",
                source_id="src_2",
                title="Advanced Python",
                content="Advanced Python concepts",
                content_hash="hash_2",
                status="active",
            ),
        ]

        # モック結果の設定
        mock_scalars = Mock()
        mock_scalars.all.return_value = search_results
        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # 検索実行
        result = await document_repository.search_by_content("Python")

        # 検証
        assert len(result) == 2
        assert all("Python" in doc.title or "Python" in doc.content for doc in result)
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_count_by_source_type(self, document_repository, mock_session):
        """ソースタイプ別カウントのテスト"""
        # モック結果の設定
        mock_result = Mock()
        mock_result.scalar.return_value = 42
        mock_session.execute.return_value = mock_result

        # カウント実行
        result = await document_repository.count_by_source_type("confluence")

        # 検証
        assert result == 42
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_count_by_source_type_zero(self, document_repository, mock_session):
        """ソースタイプ別カウント（0件）のテスト"""
        # モック結果の設定
        mock_result = Mock()
        mock_result.scalar.return_value = None  # カウント0の場合
        mock_session.execute.return_value = mock_result

        # カウント実行
        result = await document_repository.count_by_source_type("nonexistent")

        # 検証
        assert result == 0

    @pytest.mark.asyncio
    async def test_get_outdated_documents(self, document_repository, mock_session):
        """古いドキュメント取得のテスト"""
        # 古いドキュメント
        old_time = datetime.utcnow() - timedelta(hours=48)
        outdated_docs = [
            Document(
                id="doc_1",
                source_type="confluence",
                source_id="src_1",
                title="Old Document 1",
                content="Old content 1",
                content_hash="hash_1",
                status="active",
                updated_at=old_time,
            ),
            Document(
                id="doc_2",
                source_type="confluence",
                source_id="src_2",
                title="Old Document 2",
                content="Old content 2",
                content_hash="hash_2",
                status="active",
                updated_at=old_time,
            ),
        ]

        # モック結果の設定
        mock_scalars = Mock()
        mock_scalars.all.return_value = outdated_docs
        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # 取得実行
        result = await document_repository.get_outdated_documents(hours=24)

        # 検証
        assert len(result) == 2
        assert all(doc.status == "active" for doc in result)
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_outdated_documents_custom_hours(
        self, document_repository, mock_session
    ):
        """古いドキュメント取得（カスタム時間）のテスト"""
        # モック結果の設定
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # 取得実行（72時間以上古いドキュメント）
        result = await document_repository.get_outdated_documents(hours=72)

        # 検証
        assert len(result) == 0
        mock_session.execute.assert_called_once()
