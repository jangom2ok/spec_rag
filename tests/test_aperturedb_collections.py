"""ApertureDBコレクションのテスト"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from app.models.aperturedb import (
    DenseVectorCollection,
    SparseVectorCollection,
    VectorData,
)


class TestApertureDBCollection:
    """ApertureDBコレクションの基底クラステスト"""

    def test_collection_name_and_dimension(self):
        """コレクション名と次元数の検証テスト"""
        # Mock ApertureDB connection
        with patch("app.models.aperturedb.Client") as mock_client:
            mock_client.return_value = Mock()
            mock_client.return_value.query.return_value = (
                [{"FindDescriptorSet": {"count": 0}}],
                [],
            )

            dense_collection = DenseVectorCollection()
            sparse_collection = SparseVectorCollection()

            # コレクション名の確認
            assert dense_collection.get_collection_name() == "document_vectors_dense"
            assert sparse_collection.get_collection_name() == "document_vectors_sparse"

            # ベクトル次元数の確認
            assert dense_collection.get_vector_dimension() == 1024
            assert (
                sparse_collection.get_vector_dimension() == 1
            )  # Sparse vectorはダミー値

    def test_connection_initialization(self):
        """接続初期化のテスト"""
        with patch("app.models.aperturedb.Client") as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.query.return_value = (
                [{"FindDescriptorSet": {"count": 0}}],
                [],
            )
            mock_client.return_value = mock_client_instance

            collection = DenseVectorCollection()

            # クライアントが作成されていることを確認
            assert collection.client is not None
            mock_client.assert_called_once_with(
                host="localhost",
                port=55555,
                username="admin",
                password="admin",
            )


class TestDenseVectorCollection:
    """Dense Vectorコレクションのテスト"""

    @pytest.fixture
    def mock_aperturedb_client(self):
        """ApertureDBのクライアントをモック"""
        with patch("app.models.aperturedb.Client") as mock_client:
            mock_client_instance = Mock()
            # デスクリプターセットの初期化
            mock_client_instance.query.return_value = (
                [{"FindDescriptorSet": {"count": 0}}],
                [],
            )
            mock_client.return_value = mock_client_instance

            yield mock_client_instance

    async def test_dense_vector_insertion(self, mock_aperturedb_client):
        """Dense Vectorの挿入テスト"""
        collection = DenseVectorCollection()
        collection.client = mock_aperturedb_client

        # テストデータ
        vector_data = VectorData(
            id="test-id-1",
            document_id="doc-123",
            chunk_id="chunk-456",
            vector=np.random.random(1024).tolist(),
            chunk_type="paragraph",
            source_type="git",
            language="ja",
        )

        # 挿入成功をモック
        mock_aperturedb_client.query.return_value = (
            [{"AddDescriptor": {"status": 0}}],
            [],
        )

        # 挿入の実行
        result = await collection.insert([vector_data])

        # 結果の確認
        assert result["primary_keys"] == ["test-id-1"]

        # ApertureDBクライアントの呼び出し確認
        # 初期化時の呼び出しを除いた2回目の呼び出しを確認
        assert mock_aperturedb_client.query.call_count >= 2
        last_call = mock_aperturedb_client.query.call_args_list[-1]
        query = last_call[0][0][0]

        assert "AddDescriptor" in query
        assert query["AddDescriptor"]["set"] == "document_vectors_dense"
        assert len(query["AddDescriptor"]["descriptor"]) == 1024

    async def test_dense_vector_search(self, mock_aperturedb_client):
        """Dense Vectorの検索テスト"""
        collection = DenseVectorCollection()
        collection.client = mock_aperturedb_client

        # 検索結果をモック
        mock_aperturedb_client.query.return_value = (
            [
                {
                    "FindDescriptor": {
                        "returned": 2,
                        "entities": [
                            {
                                "id": "doc-1",
                                "_distance": 0.1,
                                "document_id": "doc-1",
                                "chunk_id": "chunk-1",
                                "chunk_type": "paragraph",
                            },
                            {
                                "id": "doc-2",
                                "_distance": 0.3,
                                "document_id": "doc-2",
                                "chunk_id": "chunk-2",
                                "chunk_type": "section",
                            },
                        ],
                    }
                }
            ],
            [],
        )

        # 検索実行
        query_vector = np.random.random(1024).tolist()
        results = await collection.search(
            query_vectors=[query_vector], top_k=10, filters={"source_type": "git"}
        )

        # 結果確認
        assert len(results) == 1
        assert results[0]["ids"] == ["doc-1", "doc-2"]
        assert results[0]["distances"] == [0.1, 0.3]
        assert len(results[0]["entities"]) == 2

    async def test_dense_vector_delete(self, mock_aperturedb_client):
        """Dense Vectorの削除テスト"""
        collection = DenseVectorCollection()
        collection.client = mock_aperturedb_client

        # 削除対象の検索結果をモック
        mock_aperturedb_client.query.side_effect = [
            # 削除対象の検索
            ([{"FindDescriptor": {"count": 3}}], []),
            # 削除実行
            ([{"DeleteDescriptor": {"status": 0}}], []),
        ]

        # 削除実行
        result = await collection.delete_by_document_id("doc-123")

        # 削除結果確認
        assert result["delete_count"] == 3


class TestSparseVectorCollection:
    """Sparse Vectorコレクションのテスト"""

    @pytest.fixture
    def mock_aperturedb_client(self):
        """ApertureDBのクライアントをモック"""
        with patch("app.models.aperturedb.Client") as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.query.return_value = (
                [{"FindDescriptorSet": {"count": 0}}],
                [],
            )
            mock_client.return_value = mock_client_instance

            yield mock_client_instance

    def test_sparse_vector_collection_name(self):
        """Sparse Vectorコレクション名のテスト"""
        with patch("app.models.aperturedb.Client") as mock_client:
            mock_client.return_value = Mock()
            mock_client.return_value.query.return_value = (
                [{"FindDescriptorSet": {"count": 0}}],
                [],
            )

            collection = SparseVectorCollection()
            assert collection.get_collection_name() == "document_vectors_sparse"

    async def test_sparse_vector_insertion(self, mock_aperturedb_client):
        """Sparse Vectorの挿入テスト"""
        collection = SparseVectorCollection()
        collection.client = mock_aperturedb_client

        # Sparse vectorのテストデータ（辞書形式）
        sparse_vector = {0: 0.5, 10: 0.3, 100: 0.8}

        vector_data = VectorData(
            id="sparse-test-1",
            document_id="doc-456",
            chunk_id="chunk-789",
            sparse_vector=sparse_vector,
            vocabulary_size=1000,
        )

        # 挿入成功をモック
        mock_aperturedb_client.query.return_value = ([{"AddEntity": {"status": 0}}], [])

        # 挿入実行
        result = await collection.insert([vector_data])

        # 結果確認
        assert result["primary_keys"] == ["sparse-test-1"]

        # クライアントの呼び出し確認
        assert mock_aperturedb_client.query.call_count >= 2
        last_call = mock_aperturedb_client.query.call_args_list[-1]
        query = last_call[0][0][0]

        assert "AddEntity" in query
        assert query["AddEntity"]["class"] == "SparseVector"
        assert query["AddEntity"]["properties"]["vocabulary_size"] == 1000

    async def test_hybrid_search_preparation(self, mock_aperturedb_client):
        """ハイブリッド検索の準備テスト"""
        dense_collection = DenseVectorCollection()
        sparse_collection = SparseVectorCollection()

        dense_collection.client = mock_aperturedb_client
        sparse_collection.client = mock_aperturedb_client

        # 両方のコレクションでの検索結果をモック
        with (
            patch.object(dense_collection, "search") as mock_dense_search,
            patch.object(sparse_collection, "search") as mock_sparse_search,
        ):
            mock_dense_search.return_value = [
                {"ids": ["doc-1", "doc-2"], "distances": [0.1, 0.3], "entities": []}
            ]
            mock_sparse_search.return_value = [
                {"ids": ["doc-2", "doc-3"], "distances": [0.2, 0.4], "entities": []}
            ]

            # Dense検索
            dense_results = await dense_collection.search(
                query_vectors=[np.random.random(1024).tolist()], top_k=10
            )

            # Sparse検索
            sparse_query = {0: 0.5, 10: 0.3}
            sparse_results = await sparse_collection.search(
                query_vectors=[sparse_query], top_k=10
            )

            # 結果確認（ハイブリッド検索の準備）
            assert len(dense_results) == 1
            assert len(sparse_results) == 1

            # RRF（Reciprocal Rank Fusion）のための準備
            dense_docs = dense_results[0]["ids"]
            sparse_docs = sparse_results[0]["ids"]

            # 共通ドキュメントの確認
            common_docs = set(dense_docs) & set(sparse_docs)
            assert "doc-2" in common_docs


class TestVectorData:
    """VectorDataクラスのテスト"""

    def test_vector_data_creation(self):
        """VectorDataの作成テスト"""
        vector_data = VectorData(
            id="test-123",
            document_id="doc-456",
            chunk_id="chunk-789",
            vector=[0.1, 0.2, 0.3],
            chunk_type="section",
            source_type="confluence",
            language="en",
        )

        assert vector_data.id == "test-123"
        assert vector_data.document_id == "doc-456"
        assert vector_data.vector == [0.1, 0.2, 0.3]
        assert vector_data.chunk_type == "section"

    def test_sparse_vector_data_creation(self):
        """Sparse VectorDataの作成テスト"""
        sparse_vector = {0: 0.5, 100: 0.8, 500: 0.3}

        vector_data = VectorData(
            id="sparse-123",
            document_id="doc-456",
            chunk_id="chunk-789",
            sparse_vector=sparse_vector,
            vocabulary_size=1000,
        )

        assert vector_data.sparse_vector == sparse_vector
        assert vector_data.vocabulary_size == 1000

    def test_vector_data_validation(self):
        """VectorDataのバリデーションテスト"""
        # Dense vectorの場合、vectorが必須
        with pytest.raises((ValueError, TypeError)):
            VectorData(
                id="test-123",
                document_id="doc-456",
                chunk_id="chunk-789",
                # vectorが不足
                chunk_type="section",
            )

        # Sparse vectorの場合、sparse_vectorとvocabulary_sizeが必須
        with pytest.raises((ValueError, TypeError)):
            VectorData(
                id="test-123",
                document_id="doc-456",
                chunk_id="chunk-789",
                sparse_vector={0: 0.5},
                # vocabulary_sizeが不足
            )
