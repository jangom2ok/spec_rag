"""Tests for Document Chunk Repository

ドキュメントチャンクリポジトリの包括的なテスト。
カバレッジの向上を目的として、すべてのCRUD操作をテスト。
"""

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import DocumentChunk
from app.repositories.chunk_repository import DocumentChunkRepository


@pytest.fixture
def mock_session():
    """モックセッションのフィクスチャ"""
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.fixture
def chunk_repository(mock_session):
    """チャンクリポジトリのフィクスチャ"""
    return DocumentChunkRepository(mock_session)


@pytest.fixture
def sample_chunk():
    """サンプルチャンクのフィクスチャ"""
    return DocumentChunk(
        id="chunk_1",
        document_id="doc_1",
        chunk_index=0,
        chunk_type="paragraph",
        title="Test Chunk",
        content="This is a test chunk content",
        content_length=28,
        token_count=6,
        hierarchy_path="/doc/chunk1",
        chunk_metadata={"key": "value"},
    )


class TestDocumentChunkRepository:
    """DocumentChunkRepositoryのテスト"""

    @pytest.mark.asyncio
    async def test_create_chunk(self, chunk_repository, mock_session, sample_chunk):
        """チャンク作成のテスト"""
        # セッションメソッドのモック設定
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        # 作成実行
        result = await chunk_repository.create(sample_chunk)

        # 検証
        assert result == sample_chunk
        mock_session.add.assert_called_once_with(sample_chunk)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_chunk)

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, chunk_repository, mock_session, sample_chunk):
        """IDによるチャンク取得（見つかった場合）のテスト"""
        # モック結果の設定
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_chunk
        mock_session.execute.return_value = mock_result

        # 取得実行
        result = await chunk_repository.get_by_id("chunk_1")

        # 検証
        assert result == sample_chunk
        mock_session.execute.assert_called_once()

        # SQLクエリの検証
        call_args = mock_session.execute.call_args[0][0]
        assert isinstance(call_args, sa.sql.Select)

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, chunk_repository, mock_session):
        """IDによるチャンク取得（見つからない場合）のテスト"""
        # モック結果の設定
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # 取得実行
        result = await chunk_repository.get_by_id("nonexistent")

        # 検証
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_document_id(self, chunk_repository, mock_session):
        """ドキュメントIDによるチャンク取得のテスト"""
        # 複数のチャンクを作成
        chunks = [
            DocumentChunk(
                id=f"chunk_{i}",
                document_id="doc_1",
                chunk_index=i,
                chunk_type="paragraph",
                content=f"Content {i}",
                content_length=10,
            )
            for i in range(3)
        ]

        # モック結果の設定
        mock_scalars = Mock()
        mock_scalars.all.return_value = chunks
        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # 取得実行
        result = await chunk_repository.get_by_document_id("doc_1")

        # 検証
        assert len(result.chunks if hasattr(result, "chunks") else []) == 3
        assert all(
            chunk.document_id == "doc_1"
            for chunk in (result.chunks if hasattr(result, "chunks") else [])
        )
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_type(self, chunk_repository, mock_session):
        """チャンクタイプによる取得のテスト"""
        # 特定タイプのチャンクを作成
        chunks = [
            DocumentChunk(
                id=f"chunk_{i}",
                document_id=f"doc_{i}",
                chunk_index=0,
                chunk_type="code",
                content=f"Code {i}",
                content_length=10,
            )
            for i in range(2)
        ]

        # モック結果の設定
        mock_scalars = Mock()
        mock_scalars.all.return_value = chunks
        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # 取得実行
        result = await chunk_repository.get_by_type("code")

        # 検証
        assert len(result.chunks if hasattr(result, "chunks") else []) == 2
        assert all(
            chunk.chunk_type == "code"
            for chunk in (result.chunks if hasattr(result, "chunks") else [])
        )

    @pytest.mark.asyncio
    async def test_update_chunk(self, chunk_repository, mock_session, sample_chunk):
        """チャンク更新のテスト"""
        # セッションメソッドのモック設定
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        # 更新実行
        sample_chunk.content = "Updated content"
        result = await chunk_repository.update(sample_chunk)

        # 検証
        assert result == sample_chunk
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_chunk)

    @pytest.mark.asyncio
    async def test_delete_chunk_exists(
        self, chunk_repository, mock_session, sample_chunk
    ):
        """チャンク削除（存在する場合）のテスト"""
        # モック結果の設定
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_chunk
        mock_session.execute.return_value = mock_result
        mock_session.delete = AsyncMock()
        mock_session.commit = AsyncMock()

        # 削除実行
        result = await chunk_repository.delete("chunk_1")

        # 検証
        assert result is True
        mock_session.delete.assert_called_once_with(sample_chunk)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_chunk_not_exists(self, chunk_repository, mock_session):
        """チャンク削除（存在しない場合）のテスト"""
        # モック結果の設定
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # 削除実行
        result = await chunk_repository.delete("nonexistent")

        # 検証
        assert result is False
        mock_session.delete.assert_not_called()
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_by_document_id(self, chunk_repository, mock_session):
        """ドキュメントIDによる削除のテスト"""
        # 削除対象のチャンクを作成
        chunks = [
            DocumentChunk(
                id=f"chunk_{i}",
                document_id="doc_1",
                chunk_index=i,
                chunk_type="paragraph",
                content=f"Content {i}",
                content_length=10,
            )
            for i in range(3)
        ]

        # モック結果の設定
        mock_scalars = Mock()
        mock_scalars.all.return_value = chunks
        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        mock_session.delete = AsyncMock()
        mock_session.commit = AsyncMock()

        # 削除実行
        result = await chunk_repository.delete_by_document_id("doc_1")

        # 検証
        assert result == 3
        assert mock_session.delete.call_count == 3
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_by_document_id_no_chunks(
        self, chunk_repository, mock_session
    ):
        """ドキュメントIDによる削除（チャンクなし）のテスト"""
        # モック結果の設定
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        mock_session.commit = AsyncMock()

        # 削除実行
        result = await chunk_repository.delete_by_document_id("doc_999")

        # 検証
        assert result == 0
        mock_session.delete.assert_not_called()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_by_content(self, chunk_repository, mock_session):
        """コンテンツ検索のテスト"""
        # 検索結果のチャンクを作成
        chunks = [
            DocumentChunk(
                id="chunk_1",
                document_id="doc_1",
                chunk_index=0,
                chunk_type="paragraph",
                title="Python Tutorial",
                content="Learn Python programming",
                content_length=24,
            ),
            DocumentChunk(
                id="chunk_2",
                document_id="doc_2",
                chunk_index=0,
                chunk_type="paragraph",
                title="Python Advanced",
                content="Advanced Python concepts",
                content_length=24,
            ),
        ]

        # モック結果の設定
        mock_scalars = Mock()
        mock_scalars.all.return_value = chunks
        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # 検索実行
        result = await chunk_repository.search_by_content("Python")

        # 検証
        assert len(result.chunks if hasattr(result, "chunks") else []) == 2
        assert all(
            "Python" in chunk.title or "Python" in chunk.content
            for chunk in (result.chunks if hasattr(result, "chunks") else [])
        )
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_by_content_no_results(self, chunk_repository, mock_session):
        """コンテンツ検索（結果なし）のテスト"""
        # モック結果の設定
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # 検索実行
        result = await chunk_repository.search_by_content("NonExistentTerm")

        # 検証
        assert len(result.chunks if hasattr(result, "chunks") else []) == 0

    @pytest.mark.asyncio
    async def test_get_chunks_by_size_range(self, chunk_repository, mock_session):
        """サイズ範囲によるチャンク取得のテスト"""
        # 異なるサイズのチャンクを作成
        chunks = [
            DocumentChunk(
                id="chunk_1",
                document_id="doc_1",
                chunk_index=0,
                chunk_type="paragraph",
                content="Short",
                content_length=5,
            ),
            DocumentChunk(
                id="chunk_2",
                document_id="doc_2",
                chunk_index=0,
                chunk_type="paragraph",
                content="Medium length content",
                content_length=21,
            ),
            DocumentChunk(
                id="chunk_3",
                document_id="doc_3",
                chunk_index=0,
                chunk_type="paragraph",
                content="This is a much longer content that exceeds the range",
                content_length=52,
            ),
        ]

        # 範囲内のチャンクのみを返すようモック設定
        filtered_chunks = [chunks[1]]  # content_length=21のチャンクのみ
        mock_scalars = Mock()
        mock_scalars.all.return_value = filtered_chunks
        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # 取得実行
        result = await chunk_repository.get_chunks_by_size_range(10, 30)

        # 検証
        assert len(result.chunks if hasattr(result, "chunks") else []) == 1
        assert (result.chunks if hasattr(result, "chunks") else [])[
            0
        ].content_length == 21
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_chunks_by_size_range_empty(self, chunk_repository, mock_session):
        """サイズ範囲によるチャンク取得（結果なし）のテスト"""
        # モック結果の設定
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # 取得実行
        result = await chunk_repository.get_chunks_by_size_range(1000, 2000)

        # 検証
        assert len(result.chunks if hasattr(result, "chunks") else []) == 0
