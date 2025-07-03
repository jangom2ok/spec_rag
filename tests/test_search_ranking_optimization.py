"""検索ランキング最適化のテストモジュール

TDD実装：関連性スコアリングとランキング最適化機能
"""

import pytest
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta

from app.services.hybrid_search_engine import (
    HybridSearchEngine,
    SearchConfig,
    SearchQuery,
    SearchResult,
    VectorSearchResult,
    SearchFilter,
    SearchMode,
    RankingAlgorithm,
    FacetResult,
)


class TestSearchRankingOptimization:
    """検索ランキング最適化のテストクラス"""

    @pytest.fixture
    def ranking_config(self) -> SearchConfig:
        """ランキング用検索設定"""
        return SearchConfig(
            dense_weight=0.6,
            sparse_weight=0.4,
            top_k=20,
            ranking_algorithm=RankingAlgorithm.RRF,
            enable_reranking=True,
            rerank_top_k=100,
        )

    @pytest.fixture
    def weighted_sum_config(self) -> SearchConfig:
        """重み付き和用設定"""
        return SearchConfig(
            dense_weight=0.7,
            sparse_weight=0.3,
            ranking_algorithm=RankingAlgorithm.WEIGHTED_SUM,
            enable_reranking=False,
        )

    @pytest.fixture
    def borda_count_config(self) -> SearchConfig:
        """ボルダカウント用設定"""
        return SearchConfig(
            dense_weight=0.5,
            sparse_weight=0.5,
            ranking_algorithm=RankingAlgorithm.BORDA_COUNT,
            enable_reranking=True,
        )

    @pytest.fixture
    def sample_dense_results(self) -> VectorSearchResult:
        """サンプルdense検索結果"""
        return VectorSearchResult(
            ids=["doc-1", "doc-2", "doc-3", "doc-4", "doc-5"],
            scores=[0.95, 0.87, 0.73, 0.68, 0.54],
            search_time=0.05,
            total_hits=5,
        )

    @pytest.fixture
    def sample_sparse_results(self) -> VectorSearchResult:
        """サンプルsparse検索結果"""
        return VectorSearchResult(
            ids=["doc-3", "doc-1", "doc-4", "doc-2", "doc-6"],
            scores=[0.92, 0.84, 0.71, 0.65, 0.48],
            search_time=0.03,
            total_hits=5,
        )

    @pytest.fixture
    def documents_with_features(self) -> List[Dict[str, Any]]:
        """特徴量付きドキュメント"""
        base_date = datetime(2024, 1, 1)
        return [
            {
                "id": "doc-1",
                "title": "Machine Learning Fundamentals",
                "content": "Comprehensive guide to machine learning algorithms",
                "search_score": 0.95,
                "metadata": {
                    "author": "expert_author",
                    "view_count": 5000,
                    "rating": 4.8,
                    "created_at": base_date.isoformat(),
                    "updated_at": (base_date + timedelta(days=30)).isoformat(),
                    "tags": ["ml", "tutorial", "fundamentals"],
                    "category": "ai",
                    "word_count": 2500,
                    "reading_time": 12,
                    "likes": 450,
                    "shares": 120,
                },
            },
            {
                "id": "doc-2",
                "title": "Advanced Neural Networks",
                "content": "Deep dive into neural network architectures",
                "search_score": 0.87,
                "metadata": {
                    "author": "researcher",
                    "view_count": 3200,
                    "rating": 4.6,
                    "created_at": (base_date + timedelta(days=5)).isoformat(),
                    "updated_at": (base_date + timedelta(days=35)).isoformat(),
                    "tags": ["neural", "deep", "advanced"],
                    "category": "ai",
                    "word_count": 3200,
                    "reading_time": 15,
                    "likes": 320,
                    "shares": 85,
                },
            },
            {
                "id": "doc-3",
                "title": "Data Processing Pipeline",
                "content": "Building efficient data processing workflows",
                "search_score": 0.73,
                "metadata": {
                    "author": "data_engineer",
                    "view_count": 1800,
                    "rating": 4.2,
                    "created_at": (base_date + timedelta(days=10)).isoformat(),
                    "updated_at": (base_date + timedelta(days=40)).isoformat(),
                    "tags": ["data", "pipeline", "processing"],
                    "category": "engineering",
                    "word_count": 1800,
                    "reading_time": 8,
                    "likes": 180,
                    "shares": 45,
                },
            },
        ]

    @pytest.mark.unit
    def test_rrf_fusion_basic(
        self,
        ranking_config: SearchConfig,
        sample_dense_results: VectorSearchResult,
        sample_sparse_results: VectorSearchResult,
    ):
        """基本的なRRF融合テスト"""
        search_engine = HybridSearchEngine(config=ranking_config)
        
        fused_results = search_engine.rrf.fuse_rankings(
            sample_dense_results, sample_sparse_results, k=60
        )
        
        # 結果がタプル形式であることを確認
        assert isinstance(fused_results, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in fused_results)
        
        # スコア降順でソートされていることを確認
        scores = [score for _, score in fused_results]
        assert scores == sorted(scores, reverse=True)
        
        # 両方の結果に含まれるドキュメントが上位に来ることを確認
        result_ids = [doc_id for doc_id, _ in fused_results]
        assert "doc-1" in result_ids[:3]  # 両方に高位で登場
        assert "doc-2" in result_ids[:3]  # 両方に高位で登場

    @pytest.mark.unit
    def test_rrf_k_parameter_effect(
        self,
        ranking_config: SearchConfig,
        sample_dense_results: VectorSearchResult,
        sample_sparse_results: VectorSearchResult,
    ):
        """RRFのkパラメータ効果テスト"""
        search_engine = HybridSearchEngine(config=ranking_config)
        
        # k=30での結果
        fused_k30 = search_engine.rrf.fuse_rankings(
            sample_dense_results, sample_sparse_results, k=30
        )
        
        # k=120での結果
        fused_k120 = search_engine.rrf.fuse_rankings(
            sample_dense_results, sample_sparse_results, k=120
        )
        
        # kが小さいほど上位ランクの影響が大きくなる
        scores_k30 = dict(fused_k30)
        scores_k120 = dict(fused_k120)
        
        # 順位は同じでもスコアが異なることを確認
        for doc_id in scores_k30:
            if doc_id in scores_k120:
                assert scores_k30[doc_id] != scores_k120[doc_id]

    @pytest.mark.unit
    def test_weighted_sum_fusion(
        self,
        weighted_sum_config: SearchConfig,
        sample_dense_results: VectorSearchResult,
        sample_sparse_results: VectorSearchResult,
    ):
        """重み付き和融合テスト"""
        search_engine = HybridSearchEngine(config=weighted_sum_config)
        
        fused_results = search_engine._weighted_sum_fusion(
            sample_dense_results, sample_sparse_results
        )
        
        # 結果がタプル形式であることを確認
        assert isinstance(fused_results, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in fused_results)
        
        # 重み付き和の計算確認
        # doc-1: dense_score * 0.7 + sparse_score * 0.3
        doc1_expected = 0.95 * 0.7 + 0.84 * 0.3  # 0.665 + 0.252 = 0.917
        doc1_actual = next((score for doc_id, score in fused_results if doc_id == "doc-1"), None)
        assert doc1_actual is not None
        assert abs(doc1_actual - doc1_expected) < 0.001

    @pytest.mark.unit
    def test_content_relevance_scoring(
        self,
        ranking_config: SearchConfig,
        documents_with_features: List[Dict[str, Any]],
    ):
        """コンテンツ関連性スコアリングテスト"""
        search_engine = HybridSearchEngine(config=ranking_config)
        
        query_text = "machine learning fundamentals"
        
        # タイトル一致度を計算
        relevance_scores = search_engine._calculate_content_relevance(
            documents_with_features, query_text
        )
        
        assert len(relevance_scores) == len(documents_with_features)
        
        # "Machine Learning Fundamentals"が最高スコアであることを確認
        doc1_score = relevance_scores["doc-1"]
        doc2_score = relevance_scores["doc-2"]
        doc3_score = relevance_scores["doc-3"]
        
        assert doc1_score > doc2_score
        assert doc1_score > doc3_score

    @pytest.mark.unit
    def test_freshness_scoring(
        self,
        ranking_config: SearchConfig,
        documents_with_features: List[Dict[str, Any]],
    ):
        """新しさスコアリングテスト"""
        search_engine = HybridSearchEngine(config=ranking_config)
        
        freshness_scores = search_engine._calculate_freshness_scores(documents_with_features)
        
        assert len(freshness_scores) == len(documents_with_features)
        
        # より新しいドキュメントが高いスコアを持つことを確認
        doc1_freshness = freshness_scores["doc-1"]
        doc2_freshness = freshness_scores["doc-2"]
        doc3_freshness = freshness_scores["doc-3"]
        
        # doc-2, doc-3がdoc-1より新しい
        assert doc2_freshness > doc1_freshness
        assert doc3_freshness > doc1_freshness

    @pytest.mark.unit
    def test_popularity_scoring(
        self,
        ranking_config: SearchConfig,
        documents_with_features: List[Dict[str, Any]],
    ):
        """人気度スコアリングテスト"""
        search_engine = HybridSearchEngine(config=ranking_config)
        
        popularity_scores = search_engine._calculate_popularity_scores(documents_with_features)
        
        assert len(popularity_scores) == len(documents_with_features)
        
        # より多くのビューとレーティングを持つドキュメントが高いスコア
        doc1_popularity = popularity_scores["doc-1"]
        doc2_popularity = popularity_scores["doc-2"]
        doc3_popularity = popularity_scores["doc-3"]
        
        assert doc1_popularity > doc2_popularity  # より高いビューとレーティング
        assert doc2_popularity > doc3_popularity

    @pytest.mark.unit
    def test_authority_scoring(
        self,
        ranking_config: SearchConfig,
        documents_with_features: List[Dict[str, Any]],
    ):
        """権威性スコアリングテスト"""
        search_engine = HybridSearchEngine(config=ranking_config)
        
        authority_scores = search_engine._calculate_authority_scores(documents_with_features)
        
        assert len(authority_scores) == len(documents_with_features)
        
        # expert_authorが最高の権威性スコアを持つことを確認
        doc1_authority = authority_scores["doc-1"]  # expert_author
        doc2_authority = authority_scores["doc-2"]  # researcher
        doc3_authority = authority_scores["doc-3"]  # data_engineer
        
        assert doc1_authority > doc2_authority
        assert doc2_authority > doc3_authority

    @pytest.mark.unit
    def test_quality_scoring(
        self,
        ranking_config: SearchConfig,
        documents_with_features: List[Dict[str, Any]],
    ):
        """品質スコアリングテスト"""
        search_engine = HybridSearchEngine(config=ranking_config)
        
        quality_scores = search_engine._calculate_quality_scores(documents_with_features)
        
        assert len(quality_scores) == len(documents_with_features)
        
        # レーティング、単語数、エンゲージメントが高いほど高品質
        doc1_quality = quality_scores["doc-1"]
        doc2_quality = quality_scores["doc-2"]
        doc3_quality = quality_scores["doc-3"]
        
        assert doc1_quality > doc3_quality  # より高いレーティングとエンゲージメント
        assert doc2_quality > doc3_quality

    @pytest.mark.unit
    def test_combined_relevance_scoring(
        self,
        ranking_config: SearchConfig,
        documents_with_features: List[Dict[str, Any]],
    ):
        """統合関連性スコアリングテスト"""
        search_engine = HybridSearchEngine(config=ranking_config)
        
        query_text = "machine learning fundamentals"
        
        combined_scores = search_engine._calculate_combined_relevance_scores(
            documents_with_features, query_text
        )
        
        assert len(combined_scores) == len(documents_with_features)
        
        # 各スコアが0-1の範囲内であることを確認
        for doc_id, score in combined_scores.items():
            assert 0 <= score <= 1
        
        # 最適なドキュメント（doc-1）が最高スコアを持つことを確認
        doc1_score = combined_scores["doc-1"]
        doc2_score = combined_scores["doc-2"]
        doc3_score = combined_scores["doc-3"]
        
        assert doc1_score > doc2_score
        assert doc1_score > doc3_score

    @pytest.mark.unit
    def test_reranking_with_features(
        self,
        ranking_config: SearchConfig,
        documents_with_features: List[Dict[str, Any]],
    ):
        """特徴量を使ったリランキングテスト"""
        search_engine = HybridSearchEngine(config=ranking_config)
        
        query_text = "machine learning tutorial"
        
        # 初期ランキング（検索スコア順）
        initial_ranking = sorted(
            documents_with_features, 
            key=lambda x: x["search_score"], 
            reverse=True
        )
        
        # リランキング実行
        reranked_docs = search_engine._rerank_with_features(
            initial_ranking, query_text
        )
        
        assert len(reranked_docs) == len(documents_with_features)
        
        # 全てのドキュメントにrerank_scoreが追加されていることを確認
        for doc in reranked_docs:
            assert "rerank_score" in doc
            assert 0 <= doc["rerank_score"] <= 1

    @pytest.mark.unit
    def test_personalization_scoring(
        self,
        ranking_config: SearchConfig,
        documents_with_features: List[Dict[str, Any]],
    ):
        """パーソナライゼーションスコアリングテスト"""
        search_engine = HybridSearchEngine(config=ranking_config)
        
        user_profile = {
            "preferred_categories": ["ai", "ml"],
            "preferred_authors": ["expert_author"],
            "reading_level": "advanced",
            "interaction_history": {
                "doc-1": {"views": 5, "rating": 5},
                "doc-2": {"views": 2, "rating": 4},
            }
        }
        
        personalization_scores = search_engine._calculate_personalization_scores(
            documents_with_features, user_profile
        )
        
        assert len(personalization_scores) == len(documents_with_features)
        
        # ユーザーの好みに合致するドキュメントが高いスコア
        doc1_personal = personalization_scores["doc-1"]
        doc2_personal = personalization_scores["doc-2"]
        doc3_personal = personalization_scores["doc-3"]
        
        assert doc1_personal > doc3_personal  # preferred author + category match

    @pytest.mark.unit
    def test_diversity_ranking(
        self,
        ranking_config: SearchConfig,
        documents_with_features: List[Dict[str, Any]],
    ):
        """多様性ランキングテスト"""
        search_engine = HybridSearchEngine(config=ranking_config)
        
        # より多様なドキュメントセットを作成
        diverse_docs = documents_with_features + [
            {
                "id": "doc-4",
                "title": "Web Development Basics",
                "content": "Introduction to web development",
                "search_score": 0.82,
                "metadata": {
                    "category": "web",
                    "tags": ["web", "development", "basics"],
                    "author": "web_developer",
                },
            }
        ]
        
        diversified_ranking = search_engine._apply_diversity_ranking(
            diverse_docs, diversity_factor=0.3
        )
        
        assert len(diversified_ranking) == len(diverse_docs)
        
        # 多様性が考慮されて異なるカテゴリのドキュメントが含まれることを確認
        categories = [doc["metadata"]["category"] for doc in diversified_ranking[:3]]
        unique_categories = set(categories)
        assert len(unique_categories) > 1  # 複数のカテゴリが含まれる

    @pytest.mark.unit
    def test_temporal_boost_scoring(
        self,
        ranking_config: SearchConfig,
        documents_with_features: List[Dict[str, Any]],
    ):
        """時間的ブーストスコアリングテスト"""
        search_engine = HybridSearchEngine(config=ranking_config)
        
        # 現在時刻を基準とした時間的ブースト
        current_time = datetime.now()
        boost_scores = search_engine._calculate_temporal_boost_scores(
            documents_with_features, current_time, boost_recent=True
        )
        
        assert len(boost_scores) == len(documents_with_features)
        
        # 最近更新されたドキュメントが高いブーストを受けることを確認
        for doc_id, boost in boost_scores.items():
            assert 0.5 <= boost <= 1.5  # ブースト範囲

    @pytest.mark.integration
    def test_end_to_end_ranking_optimization(
        self,
        ranking_config: SearchConfig,
        documents_with_features: List[Dict[str, Any]],
    ):
        """End-to-Endランキング最適化テスト"""
        search_engine = HybridSearchEngine(config=ranking_config)
        
        query_text = "machine learning algorithms"
        user_profile = {
            "preferred_categories": ["ai"],
            "reading_level": "intermediate",
        }
        
        # 完全なランキング最適化パイプライン
        optimized_ranking = search_engine._optimize_ranking_pipeline(
            documents_with_features, query_text, user_profile
        )
        
        assert len(optimized_ranking) == len(documents_with_features)
        
        # 各ドキュメントに最終的なランキングスコアが付与されていることを確認
        for doc in optimized_ranking:
            assert "final_ranking_score" in doc
            assert 0 <= doc["final_ranking_score"] <= 1
        
        # 最終スコア順でソートされていることを確認
        final_scores = [doc["final_ranking_score"] for doc in optimized_ranking]
        assert final_scores == sorted(final_scores, reverse=True)

    @pytest.mark.unit
    def test_ranking_algorithm_comparison(
        self,
        ranking_config: SearchConfig,
        weighted_sum_config: SearchConfig,
        sample_dense_results: VectorSearchResult,
        sample_sparse_results: VectorSearchResult,
    ):
        """ランキングアルゴリズム比較テスト"""
        rrf_engine = HybridSearchEngine(config=ranking_config)
        weighted_engine = HybridSearchEngine(config=weighted_sum_config)
        
        # RRF結果
        rrf_results = rrf_engine._fuse_search_results(
            sample_dense_results, sample_sparse_results, None
        )
        
        # 重み付き和結果
        weighted_results = weighted_engine._fuse_search_results(
            sample_dense_results, sample_sparse_results, None
        )
        
        # 異なるアルゴリズムで異なる結果が得られることを確認
        rrf_ranking = [doc_id for doc_id, _ in rrf_results]
        weighted_ranking = [doc_id for doc_id, _ in weighted_results]
        
        # 完全に同じ順序ではないことを確認
        assert rrf_ranking != weighted_ranking

    @pytest.mark.unit
    def test_ranking_performance_metrics(
        self,
        ranking_config: SearchConfig,
        documents_with_features: List[Dict[str, Any]],
    ):
        """ランキング性能メトリクステスト"""
        search_engine = HybridSearchEngine(config=ranking_config)
        
        query_text = "machine learning"
        
        # 関連性の ground truth（テスト用）
        relevance_labels = {"doc-1": 1, "doc-2": 1, "doc-3": 0}  # 1: relevant, 0: not relevant
        
        # ランキング実行
        ranked_docs = search_engine._rerank_with_features(documents_with_features, query_text)
        
        # メトリクス計算
        metrics = search_engine._calculate_ranking_metrics(ranked_docs, relevance_labels)
        
        assert "precision_at_1" in metrics
        assert "precision_at_3" in metrics
        assert "map" in metrics  # Mean Average Precision
        assert "ndcg" in metrics  # Normalized Discounted Cumulative Gain
        
        # メトリクス値が有効範囲内であることを確認
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1

    @pytest.mark.unit
    def test_ranking_explainability(
        self,
        ranking_config: SearchConfig,
        documents_with_features: List[Dict[str, Any]],
    ):
        """ランキング説明可能性テスト"""
        search_engine = HybridSearchEngine(config=ranking_config)
        
        query_text = "machine learning fundamentals"
        
        # 説明付きランキング
        ranked_docs_with_explanation = search_engine._rank_with_explanation(
            documents_with_features, query_text
        )
        
        assert len(ranked_docs_with_explanation) == len(documents_with_features)
        
        # 各ドキュメントに説明が付与されていることを確認
        for doc in ranked_docs_with_explanation:
            assert "ranking_explanation" in doc
            explanation = doc["ranking_explanation"]
            
            assert "content_relevance" in explanation
            assert "freshness_score" in explanation
            assert "popularity_score" in explanation
            assert "quality_score" in explanation
            assert "final_score_breakdown" in explanation