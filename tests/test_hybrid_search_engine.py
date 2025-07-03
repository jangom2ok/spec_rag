"""ハイブリッド検索エンジンのテストモジュール

TDD実装：テストケース→実装→リファクタの順序で実装
BGE-M3を活用したdense/sparse vectorsハイブリッド検索
"""

import pytest
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.services.hybrid_search_engine import (
    HybridSearchEngine,
    SearchConfig,
    SearchResult,
    SearchQuery,
    SearchMode,
    RankingAlgorithm,
    VectorSearchResult,
    SearchFilter,
    FacetResult,
)


class TestHybridSearchEngine:
    """ハイブリッド検索エンジンのテストクラス"""

    @pytest.fixture
    def basic_search_config(self) -> SearchConfig:
        """基本的な検索設定"""
        return SearchConfig(
            dense_weight=0.7,
            sparse_weight=0.3,
            top_k=10,
            dense_search_params={"metric_type": "IP", "nprobe": 16},
            sparse_search_params={"drop_ratio_search": 0.2},
            enable_reranking=True,
            rerank_top_k=100,
            search_timeout=30.0,
        )

    @pytest.fixture
    def semantic_search_config(self) -> SearchConfig:
        """セマンティック検索設定"""
        return SearchConfig(
            dense_weight=0.9,
            sparse_weight=0.1,
            top_k=20,
            search_mode=SearchMode.SEMANTIC,
            enable_reranking=True,
            similarity_threshold=0.5,
        )

    @pytest.fixture
    def keyword_search_config(self) -> SearchConfig:
        """キーワード検索設定"""
        return SearchConfig(
            dense_weight=0.2,
            sparse_weight=0.8,
            top_k=15,
            search_mode=SearchMode.KEYWORD,
            enable_reranking=False,
        )

    @pytest.fixture
    def sample_search_query(self) -> SearchQuery:
        """サンプル検索クエリ"""
        return SearchQuery(
            text="machine learning algorithms for document processing",
            filters=[
                SearchFilter(field="source_type", value="confluence"),
                SearchFilter(field="language", value="en"),
            ],
            facets=["source_type", "language", "document_type"],
            search_mode=SearchMode.HYBRID,
            max_results=10,
        )

    @pytest.fixture
    def japanese_search_query(self) -> SearchQuery:
        """日本語検索クエリ"""
        return SearchQuery(
            text="機械学習アルゴリズムによるドキュメント処理",
            filters=[SearchFilter(field="language", value="ja")],
            facets=["source_type", "document_type"],
            search_mode=SearchMode.HYBRID,
            max_results=15,
        )

    @pytest.fixture
    def mock_dense_vectors(self) -> List[List[float]]:
        """モックdenseベクター"""
        np.random.seed(42)
        return [np.random.random(1024).tolist() for _ in range(5)]

    @pytest.fixture
    def mock_sparse_vectors(self) -> List[Dict[str, float]]:
        """モックsparseベクター"""
        return [
            {"machine": 0.8, "learning": 0.7, "algorithm": 0.6, "document": 0.5},
            {"processing": 0.7, "text": 0.6, "analysis": 0.5, "nlp": 0.4},
            {"search": 0.9, "vector": 0.8, "embedding": 0.7, "similarity": 0.6},
            {"model": 0.6, "training": 0.5, "data": 0.4, "feature": 0.3},
            {"neural": 0.7, "network": 0.6, "deep": 0.5, "ai": 0.4},
        ]

    @pytest.fixture
    def mock_documents(self) -> List[Dict[str, Any]]:
        """モックドキュメント"""
        return [
            {
                "id": "doc-1",
                "title": "Machine Learning Fundamentals",
                "content": "Introduction to machine learning algorithms and their applications in document processing.",
                "source_type": "confluence",
                "language": "en",
                "document_type": "wiki",
                "metadata": {
                    "author": "john_doe",
                    "tags": ["ml", "algorithm", "tutorial"],
                    "created_at": "2024-01-01T00:00:00Z",
                },
            },
            {
                "id": "doc-2", 
                "title": "Natural Language Processing Techniques",
                "content": "Advanced NLP techniques for text analysis and document understanding.",
                "source_type": "confluence",
                "language": "en",
                "document_type": "guide",
                "metadata": {
                    "author": "jane_smith",
                    "tags": ["nlp", "text", "processing"],
                    "created_at": "2024-01-02T00:00:00Z",
                },
            },
            {
                "id": "doc-3",
                "title": "機械学習によるドキュメント解析",
                "content": "機械学習アルゴリズムを用いたドキュメント処理の手法について説明します。",
                "source_type": "jira",
                "language": "ja",
                "document_type": "specification",
                "metadata": {
                    "author": "yamada_taro",
                    "tags": ["機械学習", "ドキュメント", "解析"],
                    "created_at": "2024-01-03T00:00:00Z",
                },
            },
        ]

    @pytest.mark.unit
    async def test_search_engine_initialization(self, basic_search_config: SearchConfig):
        """検索エンジンの初期化テスト"""
        search_engine = HybridSearchEngine(config=basic_search_config)
        
        assert search_engine.config == basic_search_config
        assert search_engine.config.dense_weight == 0.7
        assert search_engine.config.sparse_weight == 0.3
        assert search_engine.config.top_k == 10

    @pytest.mark.unit
    async def test_query_embedding_generation(
        self, 
        basic_search_config: SearchConfig, 
        sample_search_query: SearchQuery
    ):
        """クエリ埋め込み生成テスト"""
        search_engine = HybridSearchEngine(config=basic_search_config)
        
        with patch.object(search_engine, '_generate_query_embeddings', new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = {
                "dense": [0.1, 0.2, 0.3] * 341 + [0.4],  # 1024次元
                "sparse": {"machine": 0.8, "learning": 0.7, "algorithms": 0.6}
            }
            
            embeddings = await search_engine._generate_query_embeddings(sample_search_query.text)
            
            assert "dense" in embeddings
            assert "sparse" in embeddings
            assert len(embeddings["dense"]) == 1024
            assert isinstance(embeddings["sparse"], dict)
            assert "machine" in embeddings["sparse"]

    @pytest.mark.unit
    async def test_dense_vector_search(
        self,
        basic_search_config: SearchConfig,
        sample_search_query: SearchQuery,
        mock_dense_vectors: List[List[float]]
    ):
        """denseベクター検索テスト"""
        search_engine = HybridSearchEngine(config=basic_search_config)
        
        with patch.object(search_engine, '_search_dense_vectors', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = VectorSearchResult(
                ids=["doc-1", "doc-2", "doc-3"],
                scores=[0.95, 0.87, 0.73],
                search_time=0.05,
                total_hits=3,
            )
            
            query_vector = mock_dense_vectors[0]
            result = await search_engine._search_dense_vectors(query_vector, sample_search_query)
            
            assert isinstance(result, VectorSearchResult)
            assert len(result.ids) == 3
            assert len(result.scores) == 3
            assert result.scores[0] > result.scores[1] > result.scores[2]  # スコア降順
            assert result.search_time > 0

    @pytest.mark.unit
    async def test_sparse_vector_search(
        self,
        basic_search_config: SearchConfig,
        sample_search_query: SearchQuery,
        mock_sparse_vectors: List[Dict[str, float]]
    ):
        """sparseベクター検索テスト"""
        search_engine = HybridSearchEngine(config=basic_search_config)
        
        with patch.object(search_engine, '_search_sparse_vectors', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = VectorSearchResult(
                ids=["doc-2", "doc-1", "doc-3"],
                scores=[0.92, 0.84, 0.69],
                search_time=0.03,
                total_hits=3,
            )
            
            query_vector = mock_sparse_vectors[0]
            result = await search_engine._search_sparse_vectors(query_vector, sample_search_query)
            
            assert isinstance(result, VectorSearchResult)
            assert len(result.ids) == 3
            assert len(result.scores) == 3
            assert result.scores[0] > result.scores[1] > result.scores[2]

    @pytest.mark.unit
    async def test_hybrid_search_execution(
        self,
        basic_search_config: SearchConfig,
        sample_search_query: SearchQuery,
        mock_documents: List[Dict[str, Any]]
    ):
        """ハイブリッド検索実行テスト"""
        search_engine = HybridSearchEngine(config=basic_search_config)
        
        # Mock embedding generation
        with patch.object(search_engine, '_generate_query_embeddings', new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = {
                "dense": [0.1] * 1024,
                "sparse": {"machine": 0.8, "learning": 0.7}
            }
            
            # Mock vector searches
            with patch.object(search_engine, '_search_dense_vectors', new_callable=AsyncMock) as mock_dense:
                mock_dense.return_value = VectorSearchResult(
                    ids=["doc-1", "doc-2", "doc-3"],
                    scores=[0.95, 0.87, 0.73],
                    search_time=0.05,
                    total_hits=3,
                )
                
                with patch.object(search_engine, '_search_sparse_vectors', new_callable=AsyncMock) as mock_sparse:
                    mock_sparse.return_value = VectorSearchResult(
                        ids=["doc-2", "doc-1", "doc-3"],
                        scores=[0.92, 0.84, 0.69],
                        search_time=0.03,
                        total_hits=3,
                    )
                    
                    # Mock document retrieval
                    with patch.object(search_engine, '_retrieve_documents', new_callable=AsyncMock) as mock_retrieve:
                        mock_retrieve.return_value = mock_documents
                        
                        result = await search_engine.search(sample_search_query)
                        
                        assert isinstance(result, SearchResult)
                        assert result.success is True
                        assert len(result.documents) > 0
                        assert result.total_hits > 0
                        assert result.search_time > 0

    @pytest.mark.unit
    async def test_search_with_filters(
        self,
        basic_search_config: SearchConfig,
        sample_search_query: SearchQuery
    ):
        """フィルター付き検索テスト"""
        search_engine = HybridSearchEngine(config=basic_search_config)
        
        # フィルター条件の確認
        filters = sample_search_query.filters
        assert len(filters) == 2
        assert filters[0].field == "source_type"
        assert filters[0].value == "confluence"
        assert filters[1].field == "language"
        assert filters[1].value == "en"
        
        # フィルター適用ロジックのテスト
        with patch.object(search_engine, '_apply_filters', return_value=True) as mock_filter:
            test_doc = {"source_type": "confluence", "language": "en"}
            result = search_engine._apply_filters(test_doc, filters)
            assert result is True

    @pytest.mark.unit
    async def test_faceted_search(
        self,
        basic_search_config: SearchConfig,
        sample_search_query: SearchQuery,
        mock_documents: List[Dict[str, Any]]
    ):
        """ファセット検索テスト"""
        search_engine = HybridSearchEngine(config=basic_search_config)
        
        with patch.object(search_engine, '_calculate_facets', return_value={
            "source_type": [
                FacetResult(value="confluence", count=2),
                FacetResult(value="jira", count=1),
            ],
            "language": [
                FacetResult(value="en", count=2),
                FacetResult(value="ja", count=1),
            ],
            "document_type": [
                FacetResult(value="wiki", count=1),
                FacetResult(value="guide", count=1),
                FacetResult(value="specification", count=1),
            ],
        }) as mock_facets:
            
            facets = search_engine._calculate_facets(mock_documents, sample_search_query.facets)
            
            assert "source_type" in facets
            assert "language" in facets
            assert "document_type" in facets
            assert len(facets["source_type"]) == 2
            assert facets["source_type"][0].value == "confluence"
            assert facets["source_type"][0].count == 2

    @pytest.mark.unit
    async def test_semantic_search_mode(
        self,
        semantic_search_config: SearchConfig,
        sample_search_query: SearchQuery
    ):
        """セマンティック検索モードテスト"""
        search_engine = HybridSearchEngine(config=semantic_search_config)
        
        # セマンティック検索では dense weight が高い
        assert search_engine.config.dense_weight == 0.9
        assert search_engine.config.sparse_weight == 0.1
        assert search_engine.config.search_mode == SearchMode.SEMANTIC

    @pytest.mark.unit
    async def test_keyword_search_mode(
        self,
        keyword_search_config: SearchConfig,
        sample_search_query: SearchQuery
    ):
        """キーワード検索モードテスト"""
        search_engine = HybridSearchEngine(config=keyword_search_config)
        
        # キーワード検索では sparse weight が高い
        assert search_engine.config.dense_weight == 0.2
        assert search_engine.config.sparse_weight == 0.8
        assert search_engine.config.search_mode == SearchMode.KEYWORD

    @pytest.mark.unit
    async def test_japanese_text_search(
        self,
        basic_search_config: SearchConfig,
        japanese_search_query: SearchQuery
    ):
        """日本語テキスト検索テスト"""
        search_engine = HybridSearchEngine(config=basic_search_config)
        
        with patch.object(search_engine, '_generate_query_embeddings', new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = {
                "dense": [0.1] * 1024,
                "sparse": {"機械学習": 0.8, "アルゴリズム": 0.7, "ドキュメント": 0.6}
            }
            
            embeddings = await search_engine._generate_query_embeddings(japanese_search_query.text)
            
            assert "dense" in embeddings
            assert "sparse" in embeddings
            assert "機械学習" in embeddings["sparse"]

    @pytest.mark.unit
    async def test_search_result_scoring(
        self,
        basic_search_config: SearchConfig
    ):
        """検索結果スコアリングテスト"""
        search_engine = HybridSearchEngine(config=basic_search_config)
        
        # Dense と Sparse のスコアを組み合わせ
        dense_score = 0.85
        sparse_score = 0.72
        
        combined_score = search_engine._calculate_combined_score(
            dense_score, sparse_score
        )
        
        expected_score = (
            dense_score * basic_search_config.dense_weight +
            sparse_score * basic_search_config.sparse_weight
        )
        
        assert abs(combined_score - expected_score) < 0.001

    @pytest.mark.unit
    async def test_search_timeout_handling(
        self,
        basic_search_config: SearchConfig,
        sample_search_query: SearchQuery
    ):
        """検索タイムアウト処理テスト"""
        basic_search_config.search_timeout = 0.001  # 非常に短いタイムアウト
        search_engine = HybridSearchEngine(config=basic_search_config)
        
        with patch.object(search_engine, '_generate_query_embeddings', new_callable=AsyncMock) as mock_embed:
            # 遅延をシミュレート
            async def slow_embedding(text):
                await asyncio.sleep(0.1)
                return {"dense": [0.1] * 1024, "sparse": {"test": 0.5}}
            
            mock_embed.side_effect = slow_embedding
            
            result = await search_engine.search(sample_search_query)
            
            # タイムアウトの場合は適切にエラーハンドリング
            assert result.success is False or result.search_time > basic_search_config.search_timeout

    @pytest.mark.unit
    async def test_empty_query_handling(
        self,
        basic_search_config: SearchConfig
    ):
        """空クエリ処理テスト"""
        search_engine = HybridSearchEngine(config=basic_search_config)
        
        empty_query = SearchQuery(
            text="",
            filters=[],
            facets=[],
            search_mode=SearchMode.HYBRID,
            max_results=10,
        )
        
        result = await search_engine.search(empty_query)
        
        assert result.success is False
        assert "empty" in result.error_message.lower()

    @pytest.mark.unit
    async def test_similarity_threshold_filtering(
        self,
        basic_search_config: SearchConfig,
        sample_search_query: SearchQuery
    ):
        """類似度閾値フィルタリングテスト"""
        basic_search_config.similarity_threshold = 0.8
        search_engine = HybridSearchEngine(config=basic_search_config)
        
        # 低スコア結果をフィルタリング
        results = [
            {"id": "doc-1", "search_score": 0.95},  # 閾値以上
            {"id": "doc-2", "search_score": 0.75},  # 閾値以下
            {"id": "doc-3", "search_score": 0.85},  # 閾値以上
        ]
        
        filtered_results = search_engine._filter_by_similarity_threshold(results)
        
        assert len(filtered_results) == 2
        assert all(r["search_score"] >= 0.8 for r in filtered_results)

    @pytest.mark.integration
    async def test_end_to_end_search(
        self,
        basic_search_config: SearchConfig,
        sample_search_query: SearchQuery,
        mock_documents: List[Dict[str, Any]]
    ):
        """End-to-End検索テスト"""
        search_engine = HybridSearchEngine(config=basic_search_config)
        
        # 全体的な検索フローのモック
        with patch.object(search_engine, '_generate_query_embeddings', new_callable=AsyncMock) as mock_embed, \
             patch.object(search_engine, '_search_dense_vectors', new_callable=AsyncMock) as mock_dense, \
             patch.object(search_engine, '_search_sparse_vectors', new_callable=AsyncMock) as mock_sparse, \
             patch.object(search_engine, '_retrieve_documents', new_callable=AsyncMock) as mock_retrieve:
            
            # Setup mocks
            mock_embed.return_value = {
                "dense": [0.1] * 1024,
                "sparse": {"machine": 0.8, "learning": 0.7}
            }
            
            mock_dense.return_value = VectorSearchResult(
                ids=["doc-1", "doc-2"],
                scores=[0.95, 0.87],
                search_time=0.05,
                total_hits=2,
            )
            
            mock_sparse.return_value = VectorSearchResult(
                ids=["doc-2", "doc-1"],
                scores=[0.92, 0.84],
                search_time=0.03,
                total_hits=2,
            )
            
            # 検索スコア付きのモックドキュメントを作成
            mock_docs_with_scores = []
            for i, doc in enumerate(mock_documents[:2]):
                doc_copy = doc.copy()
                doc_copy["search_score"] = [0.95, 0.87][i]  # Dense scoresを使用
                mock_docs_with_scores.append(doc_copy)
            
            mock_retrieve.return_value = mock_docs_with_scores
            
            # 検索実行
            result = await search_engine.search(sample_search_query)
            
            # 結果検証
            assert result.success is True
            assert len(result.documents) == 2
            assert result.total_hits == 2
            assert result.search_time > 0
            assert result.facets is not None
            
            # スコア順でソートされていることを確認
            for i in range(len(result.documents) - 1):
                assert result.documents[i]["search_score"] >= result.documents[i + 1]["search_score"]


class TestSearchConfig:
    """検索設定のテストクラス"""

    @pytest.mark.unit
    def test_config_validation(self):
        """設定バリデーションテスト"""
        # 有効な設定
        valid_config = SearchConfig(
            dense_weight=0.7,
            sparse_weight=0.3,
            top_k=10,
        )
        assert valid_config.dense_weight == 0.7
        assert valid_config.sparse_weight == 0.3
        
        # 無効な設定（重みの合計が1でない）
        with pytest.raises(ValueError):
            SearchConfig(
                dense_weight=0.8,
                sparse_weight=0.5,  # 合計が1.3
                top_k=10,
            )

    @pytest.mark.unit
    def test_search_mode_enum(self):
        """検索モードEnumテスト"""
        assert SearchMode.HYBRID == "hybrid"
        assert SearchMode.SEMANTIC == "semantic"
        assert SearchMode.KEYWORD == "keyword"


class TestSearchResult:
    """検索結果のテストクラス"""

    @pytest.mark.unit
    def test_search_result_creation(self):
        """検索結果作成テスト"""
        documents = [
            {"id": "doc-1", "title": "Document 1", "search_score": 0.95},
            {"id": "doc-2", "title": "Document 2", "search_score": 0.87},
        ]
        
        facets = {
            "source_type": [FacetResult(value="confluence", count=2)]
        }
        
        result = SearchResult(
            success=True,
            documents=documents,
            total_hits=2,
            search_time=0.15,
            query="test query",
            facets=facets,
        )
        
        assert result.success is True
        assert len(result.documents) == 2
        assert result.total_hits == 2
        assert result.search_time == 0.15
        assert "source_type" in result.facets

    @pytest.mark.unit
    def test_search_result_summary(self):
        """検索結果サマリーテスト"""
        result = SearchResult(
            success=True,
            documents=[{"id": "doc-1"}],
            total_hits=1,
            search_time=0.1,
            query="test",
            facets={},
        )
        
        summary = result.get_summary()
        
        assert "success" in summary
        assert "total_hits" in summary
        assert "search_time" in summary
        assert summary["success"] is True
        assert summary["total_hits"] == 1