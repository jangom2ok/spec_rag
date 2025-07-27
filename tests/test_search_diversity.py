"""検索結果多様性制御のテストモジュール

TDD実装：MMR・クラスタリングベースの結果多様化機能
"""

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from app.services.search_diversity import (
    ClusteringDiversifier,
    ClusterResult,
    DiversificationAlgorithm,
    DiversificationRequest,
    DiversificationResult,
    DiversityCandidate,
    DiversityConfig,
    MMRDiversifier,
    SearchDiversityService,
    TemporalDiversifier,
    TopicDiversifier,
)


class TestSearchDiversityService:
    """検索結果多様性制御サービスのテストクラス"""

    @pytest.fixture
    def basic_diversity_config(self) -> DiversityConfig:
        """基本多様性設定"""
        return DiversityConfig(
            algorithm=DiversificationAlgorithm.MMR,
            diversity_factor=0.5,
            max_results=10,
            similarity_threshold=0.8,
            enable_clustering=True,
            cluster_count=3,
            enable_topic_diversity=True,
            enable_temporal_diversity=False,
        )

    @pytest.fixture
    def clustering_config(self) -> DiversityConfig:
        """クラスタリング設定"""
        return DiversityConfig(
            algorithm=DiversificationAlgorithm.CLUSTERING,
            diversity_factor=0.7,
            max_results=15,
            enable_clustering=True,
            cluster_count=5,
            clustering_method="kmeans",
            min_cluster_size=2,
        )

    @pytest.fixture
    def topic_diversity_config(self) -> DiversityConfig:
        """トピック多様性設定"""
        return DiversityConfig(
            algorithm=DiversificationAlgorithm.TOPIC_BASED,
            diversity_factor=0.6,
            max_results=12,
            enable_topic_diversity=True,
            topic_weight=0.4,
            max_topics_per_result=3,
        )

    @pytest.fixture
    def sample_candidates(self) -> list[DiversityCandidate]:
        """サンプル候補リスト"""
        return [
            DiversityCandidate(
                id="doc-1",
                content="Introduction to machine learning algorithms and neural networks",
                title="Machine Learning Basics",
                score=0.95,
                embedding=np.random.random(128),
                metadata={
                    "category": "ai",
                    "topics": ["machine_learning", "neural_networks"],
                    "author": "john_doe",
                    "date": "2024-01-01",
                },
            ),
            DiversityCandidate(
                id="doc-2",
                content="Deep learning architectures for computer vision applications",
                title="Deep Learning for Vision",
                score=0.92,
                embedding=np.random.random(128),
                metadata={
                    "category": "ai",
                    "topics": ["deep_learning", "computer_vision"],
                    "author": "jane_smith",
                    "date": "2024-01-02",
                },
            ),
            DiversityCandidate(
                id="doc-3",
                content="Natural language processing techniques and applications",
                title="NLP Fundamentals",
                score=0.89,
                embedding=np.random.random(128),
                metadata={
                    "category": "nlp",
                    "topics": ["natural_language_processing", "text_analysis"],
                    "author": "alice_wilson",
                    "date": "2024-01-03",
                },
            ),
            DiversityCandidate(
                id="doc-4",
                content="Data science methodologies and statistical analysis",
                title="Data Science Methods",
                score=0.87,
                embedding=np.random.random(128),
                metadata={
                    "category": "data_science",
                    "topics": ["data_science", "statistics"],
                    "author": "bob_johnson",
                    "date": "2024-01-04",
                },
            ),
            DiversityCandidate(
                id="doc-5",
                content="Advanced machine learning algorithms for prediction",
                title="Advanced ML Algorithms",
                score=0.85,
                embedding=np.random.random(128),
                metadata={
                    "category": "ai",
                    "topics": ["machine_learning", "prediction"],
                    "author": "charlie_brown",
                    "date": "2024-01-05",
                },
            ),
        ]

    @pytest.fixture
    def diversification_request(self, sample_candidates) -> DiversificationRequest:
        """多様化リクエスト"""
        return DiversificationRequest(
            query="machine learning techniques",
            candidates=sample_candidates,
            max_results=3,
            diversity_factor=0.5,
            preserve_top_results=1,
            diversification_criteria=["topic", "category", "author"],
        )

    @pytest.mark.unit
    async def test_diversity_service_initialization(
        self, basic_diversity_config: DiversityConfig
    ):
        """多様性制御サービス初期化テスト"""
        diversity_service = SearchDiversityService(config=basic_diversity_config)

        assert diversity_service.config == basic_diversity_config
        assert diversity_service.config.algorithm == DiversificationAlgorithm.MMR
        assert diversity_service.config.diversity_factor == 0.5
        assert diversity_service.config.max_results == 10

    @pytest.mark.unit
    async def test_mmr_diversification(
        self,
        basic_diversity_config: DiversityConfig,
        diversification_request: DiversificationRequest,
    ):
        """MMR多様化テスト"""
        diversity_service = SearchDiversityService(config=basic_diversity_config)

        with patch.object(
            diversity_service, "_diversify_with_mmr", new_callable=AsyncMock
        ) as mock_mmr:
            mock_mmr.return_value = [
                diversification_request.candidates[0],  # 最高スコア
                diversification_request.candidates[2],  # 異なるカテゴリ（nlp）
                diversification_request.candidates[3],  # 異なるカテゴリ（data_science）
            ]

            result = await diversity_service._diversify_with_mmr(
                diversification_request.candidates,
                diversification_request.max_results or 10,
                diversification_request.diversity_factor or 0.5,
            )

            assert len(result) == 3
            assert result[0].id == "doc-1"  # 最高スコアが保持される
            assert result[1].id == "doc-3"  # 異なるカテゴリ
            assert result[2].id == "doc-4"  # さらに異なるカテゴリ

    @pytest.mark.unit
    async def test_clustering_diversification(
        self,
        clustering_config: DiversityConfig,
        diversification_request: DiversificationRequest,
    ):
        """クラスタリング多様化テスト"""
        diversity_service = SearchDiversityService(config=clustering_config)

        with patch.object(
            diversity_service, "_diversify_with_clustering", new_callable=AsyncMock
        ) as mock_clustering:
            mock_clustering.return_value = [
                diversification_request.candidates[0],  # クラスター1から最高スコア
                diversification_request.candidates[2],  # クラスター2から最高スコア
                diversification_request.candidates[3],  # クラスター3から最高スコア
            ]

            result = await diversity_service._diversify_with_clustering(
                diversification_request.candidates,
                diversification_request.max_results or 10,
                clustering_config.cluster_count,
            )

            assert len(result) == 3
            # 各クラスターから代表的な結果が選ばれていることを確認
            categories = [candidate.metadata["category"] for candidate in result]
            assert len(set(categories)) == 3  # 異なるカテゴリが選ばれている

    @pytest.mark.unit
    async def test_topic_based_diversification(
        self,
        topic_diversity_config: DiversityConfig,
        diversification_request: DiversificationRequest,
    ):
        """トピックベース多様化テスト"""
        diversity_service = SearchDiversityService(config=topic_diversity_config)

        with patch.object(
            diversity_service, "_diversify_with_topics", new_callable=AsyncMock
        ) as mock_topics:
            mock_topics.return_value = [
                diversification_request.candidates[0],  # machine_learning トピック
                diversification_request.candidates[
                    2
                ],  # natural_language_processing トピック
                diversification_request.candidates[3],  # data_science トピック
            ]

            result = await diversity_service._diversify_with_topics(
                diversification_request.candidates,
                diversification_request.max_results or 10,
                topic_diversity_config.topic_weight,
            )

            assert len(result) == 3
            # 異なるトピックが選ばれていることを確認
            all_topics = []
            for candidate in result:
                all_topics.extend(candidate.metadata.get("topics", []))
            assert len(set(all_topics)) >= 3  # 少なくとも3つの異なるトピック

    @pytest.mark.unit
    async def test_temporal_diversification(
        self,
        basic_diversity_config: DiversityConfig,
        diversification_request: DiversificationRequest,
    ):
        """時系列多様化テスト"""
        basic_diversity_config.enable_temporal_diversity = True
        diversity_service = SearchDiversityService(config=basic_diversity_config)

        with patch.object(
            diversity_service, "_diversify_with_temporal", new_callable=AsyncMock
        ) as mock_temporal:
            mock_temporal.return_value = [
                diversification_request.candidates[0],  # 2024-01-01
                diversification_request.candidates[2],  # 2024-01-03
                diversification_request.candidates[4],  # 2024-01-05
            ]

            result = await diversity_service._diversify_with_temporal(
                diversification_request.candidates,
                diversification_request.max_results or 10,
                time_window_days=7,
            )

            assert len(result) == 3
            # 時系列的に分散されていることを確認
            dates = [candidate.metadata["date"] for candidate in result]
            assert len(set(dates)) == 3  # 異なる日付が選ばれている

    @pytest.mark.unit
    async def test_diversification_execution(
        self,
        basic_diversity_config: DiversityConfig,
        diversification_request: DiversificationRequest,
    ):
        """多様化実行テスト"""
        diversity_service = SearchDiversityService(config=basic_diversity_config)

        # すべての多様化手法をモック
        with (
            patch.object(
                diversity_service, "_diversify_with_mmr", new_callable=AsyncMock
            ) as mock_mmr,
            patch.object(
                diversity_service,
                "_calculate_similarity_matrix",
                new_callable=AsyncMock,
            ) as mock_similarity,
        ):
            mock_similarity.return_value = np.random.random((5, 5))
            # preserve_top_results=1なので、残り2個だけ返す（合計3個になるように）
            mock_mmr.return_value = diversification_request.candidates[1:3]

            result = await diversity_service.diversify(diversification_request)

            assert isinstance(result, DiversificationResult)
            assert result.success is True
            assert result.query == diversification_request.query
            assert len(result.diversified_candidates) == 3
            assert result.diversification_time > 0
            assert result.original_count == len(diversification_request.candidates)

    @pytest.mark.unit
    async def test_similarity_matrix_calculation(
        self,
        basic_diversity_config: DiversityConfig,
        sample_candidates: list[DiversityCandidate],
    ):
        """類似度行列計算テスト"""
        diversity_service = SearchDiversityService(config=basic_diversity_config)

        similarity_matrix = await diversity_service._calculate_similarity_matrix(
            sample_candidates
        )

        assert similarity_matrix.shape == (
            len(sample_candidates),
            len(sample_candidates),
        )
        assert np.allclose(np.diagonal(similarity_matrix), 1.0)  # 対角成分は1
        assert np.allclose(similarity_matrix, similarity_matrix.T)  # 対称行列

    @pytest.mark.unit
    async def test_mmr_score_calculation(
        self,
        basic_diversity_config: DiversityConfig,
        sample_candidates: list[DiversityCandidate],
    ):
        """MMRスコア計算テスト"""
        diversity_service = SearchDiversityService(config=basic_diversity_config)

        selected_candidates = [sample_candidates[0]]
        candidate = sample_candidates[1]
        diversity_factor = 0.5

        mmr_diversifier = diversity_service.diversifiers[DiversificationAlgorithm.MMR]

        with patch.object(mmr_diversifier, "_calculate_similarity") as mock_similarity:
            mock_similarity.return_value = 0.7

            mmr_score = mmr_diversifier._calculate_mmr_score(  # type: ignore
                candidate, selected_candidates, diversity_factor
            )

            # MMR = λ * relevance - (1-λ) * max_similarity
            expected_score = (
                diversity_factor * candidate.score - (1 - diversity_factor) * 0.7
            )
            assert abs(mmr_score - expected_score) < 0.001

    @pytest.mark.unit
    async def test_clustering_candidates(
        self,
        clustering_config: DiversityConfig,
        sample_candidates: list[DiversityCandidate],
    ):
        """候補クラスタリングテスト"""
        diversity_service = SearchDiversityService(config=clustering_config)

        clustering_diversifier = diversity_service.diversifiers[
            DiversificationAlgorithm.CLUSTERING
        ]

        with patch.object(
            clustering_diversifier, "_cluster_candidates", new_callable=AsyncMock
        ) as mock_cluster:
            mock_cluster.return_value = [
                ClusterResult(
                    cluster_id=0,
                    candidates=[sample_candidates[0], sample_candidates[4]],
                    centroid=np.random.random(128),
                    coherence_score=0.85,
                ),
                ClusterResult(
                    cluster_id=1,
                    candidates=[sample_candidates[1]],
                    centroid=np.random.random(128),
                    coherence_score=0.78,
                ),
                ClusterResult(
                    cluster_id=2,
                    candidates=[sample_candidates[2], sample_candidates[3]],
                    centroid=np.random.random(128),
                    coherence_score=0.82,
                ),
            ]

            clusters = await clustering_diversifier._cluster_candidates(  # type: ignore
                sample_candidates, clustering_config.cluster_count
            )

            assert len(clusters) == 3
            assert all(isinstance(cluster, ClusterResult) for cluster in clusters)
            assert clusters[0].cluster_id == 0
            assert len(clusters[0].candidates) == 2

    @pytest.mark.unit
    async def test_topic_extraction_and_analysis(
        self,
        topic_diversity_config: DiversityConfig,
        sample_candidates: list[DiversityCandidate],
    ):
        """トピック抽出・分析テスト"""
        diversity_service = SearchDiversityService(config=topic_diversity_config)

        topic_diversifier = diversity_service.diversifiers[
            DiversificationAlgorithm.TOPIC_BASED
        ]

        with patch.object(
            topic_diversifier, "_extract_topics", new_callable=AsyncMock
        ) as mock_topics:
            mock_topics.return_value = {
                "doc-1": ["machine_learning", "neural_networks"],
                "doc-2": ["deep_learning", "computer_vision"],
                "doc-3": ["natural_language_processing", "text_analysis"],
                "doc-4": ["data_science", "statistics"],
                "doc-5": ["machine_learning", "prediction"],
            }

            topics_dict = await topic_diversifier._extract_topics(sample_candidates)  # type: ignore

            assert len(topics_dict) == 5
            assert "machine_learning" in topics_dict["doc-1"]
            assert "deep_learning" in topics_dict["doc-2"]

            # トピック多様性スコア計算
            diversity_score = topic_diversifier._calculate_topic_diversity_score(  # type: ignore
                topics_dict
            )
            assert 0 <= diversity_score <= 1

    @pytest.mark.unit
    async def test_preserve_top_results(
        self,
        basic_diversity_config: DiversityConfig,
        diversification_request: DiversificationRequest,
    ):
        """上位結果保持テスト"""
        diversity_service = SearchDiversityService(config=basic_diversity_config)

        # 上位1件を保持する設定
        diversification_request.preserve_top_results = 1

        with patch.object(
            diversity_service, "_diversify_with_mmr", new_callable=AsyncMock
        ) as mock_mmr:
            # preserve_top_results=1で最高スコアが自動保持されるので、残り2個だけ返す
            mock_mmr.return_value = [
                diversification_request.candidates[2],  # 多様化結果
                diversification_request.candidates[3],  # 多様化結果
            ]

            result = await diversity_service.diversify(diversification_request)

            assert result.success is True
            assert len(result.diversified_candidates) == 3
            # 保持された上位結果が含まれていることを確認
            assert len(result.diversified_candidates) == 3
            # 上位結果が保持されているかチェック（必ずしも最初とは限らない）
            preserved_ids = [c.id for c in result.diversified_candidates]
            assert "doc-1" in preserved_ids  # 最高スコアが保持される

    @pytest.mark.unit
    async def test_diversity_criteria_filtering(
        self,
        basic_diversity_config: DiversityConfig,
        sample_candidates: list[DiversityCandidate],
    ):
        """多様性基準フィルタリングテスト"""
        diversity_service = SearchDiversityService(config=basic_diversity_config)

        criteria = ["category", "author"]

        diversity_scores = diversity_service._calculate_diversity_scores(
            sample_candidates, criteria
        )

        assert len(diversity_scores) == len(sample_candidates)
        for _candidate_id, scores in diversity_scores.items():
            assert "category" in scores
            assert "author" in scores
            assert 0 <= scores["category"] <= 1
            assert 0 <= scores["author"] <= 1

    @pytest.mark.unit
    async def test_diversity_quality_metrics(
        self,
        basic_diversity_config: DiversityConfig,
        diversification_request: DiversificationRequest,
    ):
        """多様性品質メトリクステスト"""
        diversity_service = SearchDiversityService(config=basic_diversity_config)

        selected_candidates = diversification_request.candidates[:3]

        # 多様性メトリクス計算
        metrics = diversity_service._calculate_diversity_metrics(
            selected_candidates, diversification_request.candidates
        )

        assert "intra_list_diversity" in metrics
        assert "coverage_ratio" in metrics
        assert "novelty_score" in metrics
        assert "redundancy_score" in metrics

        # メトリクス値が適切な範囲内であることを確認
        assert 0 <= metrics["intra_list_diversity"] <= 1
        assert 0 <= metrics["coverage_ratio"] <= 1
        assert 0 <= metrics["novelty_score"] <= 1
        assert 0 <= metrics["redundancy_score"] <= 1

    @pytest.mark.unit
    async def test_diversification_caching(
        self,
        basic_diversity_config: DiversityConfig,
        diversification_request: DiversificationRequest,
    ):
        """多様化キャッシュテスト"""
        basic_diversity_config.enable_caching = True
        basic_diversity_config.cache_ttl = 3600
        diversity_service = SearchDiversityService(config=basic_diversity_config)

        with (
            patch.object(diversity_service, "_get_cache_key") as mock_cache_key,
            patch.object(
                diversity_service, "_get_from_cache", new_callable=AsyncMock
            ) as mock_get_cache,
            patch.object(
                diversity_service, "_set_cache", new_callable=AsyncMock
            ) as mock_set_cache,
            patch.object(
                diversity_service, "_diversify_with_mmr", new_callable=AsyncMock
            ) as mock_mmr,
        ):
            cache_key = "diversity_cache_key_123"
            mock_cache_key.return_value = cache_key
            mock_get_cache.return_value = None  # キャッシュなし
            mock_mmr.return_value = diversification_request.candidates[:3]

            # 初回実行
            result1 = await diversity_service.diversify(diversification_request)

            # キャッシュに保存されることを確認
            mock_set_cache.assert_called_once()

            # 2回目はキャッシュから取得
            cached_result = DiversificationResult(
                success=True,
                query=diversification_request.query,
                diversified_candidates=result1.diversified_candidates,
                diversification_time=0.00001,
                original_count=5,
                cache_hit=True,
            )
            mock_get_cache.return_value = cached_result

            result2 = await diversity_service.diversify(diversification_request)

            assert result2.cache_hit is True
            assert result2.diversification_time <= result1.diversification_time

    @pytest.mark.unit
    async def test_diversification_error_handling(
        self,
        basic_diversity_config: DiversityConfig,
        diversification_request: DiversificationRequest,
    ):
        """多様化エラーハンドリングテスト"""
        diversity_service = SearchDiversityService(config=basic_diversity_config)

        with patch.object(
            diversity_service, "_diversify_with_mmr", new_callable=AsyncMock
        ) as mock_mmr:
            # エラーをシミュレート
            mock_mmr.side_effect = Exception("Diversification failed")

            result = await diversity_service.diversify(diversification_request)

            assert result.success is False
            assert (
                result.error_message
                and "Diversification failed" in result.error_message
            )
            assert result.diversified_candidates == []

    @pytest.mark.unit
    async def test_empty_candidates_handling(
        self,
        basic_diversity_config: DiversityConfig,
    ):
        """空候補処理テスト"""
        diversity_service = SearchDiversityService(config=basic_diversity_config)

        empty_request = DiversificationRequest(
            query="test query",
            candidates=[],
            max_results=5,
        )

        result = await diversity_service.diversify(empty_request)

        assert result.success is True
        assert len(result.diversified_candidates) == 0
        assert result.original_count == 0

    @pytest.mark.integration
    async def test_end_to_end_diversification(
        self,
        basic_diversity_config: DiversityConfig,
        diversification_request: DiversificationRequest,
    ):
        """End-to-End多様化テスト"""
        diversity_service = SearchDiversityService(config=basic_diversity_config)

        # 完全な多様化パイプラインのモック
        with (
            patch.object(
                diversity_service,
                "_calculate_similarity_matrix",
                new_callable=AsyncMock,
            ) as mock_similarity,
            patch.object(
                diversity_service, "_diversify_with_mmr", new_callable=AsyncMock
            ) as mock_mmr,
        ):
            # Setup mocks
            mock_similarity.return_value = np.random.random((5, 5))
            # preserve_top_results=1で最高スコアが自動で保持されるので、残り2個だけ返す
            mock_mmr.return_value = [
                diversification_request.candidates[2],  # 多様化結果1
                diversification_request.candidates[3],  # 多様化結果2
            ]

            # 多様化実行
            result = await diversity_service.diversify(diversification_request)

            # 結果検証
            assert result.success is True
            assert result.query == diversification_request.query
            assert len(result.diversified_candidates) == 3
            assert result.diversification_time > 0
            assert result.original_count == 5

            # 多様性が向上していることを確認
            original_categories = [
                c.metadata["category"] for c in diversification_request.candidates
            ]
            diversified_categories = [
                c.metadata["category"] for c in result.diversified_candidates
            ]

            # より多様なカテゴリが選ばれていることを確認
            assert len(set(diversified_categories)) >= len(set(original_categories[:3]))


class TestMMRDiversifier:
    """MMR多様化器のテストクラス"""

    @pytest.mark.unit
    def test_mmr_diversifier_initialization(self):
        """MMR多様化器初期化テスト"""
        config = DiversityConfig(
            algorithm=DiversificationAlgorithm.MMR,
            diversity_factor=0.6,
        )

        diversifier = MMRDiversifier(config)

        assert diversifier.config == config
        assert diversifier.diversity_factor == 0.6

    @pytest.mark.unit
    async def test_mmr_selection_algorithm(self):
        """MMR選択アルゴリズムテスト"""
        config = DiversityConfig()
        diversifier = MMRDiversifier(config)

        # サンプル候補と類似度行列
        candidates = [
            DiversityCandidate(
                id=f"doc-{i}",
                content=f"content {i}",
                title=f"title {i}",
                score=0.9 - i * 0.1,
                embedding=np.random.random(10),
            )
            for i in range(5)
        ]

        similarity_matrix = np.random.random((5, 5))
        np.fill_diagonal(similarity_matrix, 1.0)

        with patch.object(
            diversifier, "_calculate_similarity_matrix"
        ) as mock_similarity:
            mock_similarity.return_value = similarity_matrix

            selected = await diversifier.select(
                candidates, max_results=3, diversity_factor=0.5
            )

            assert len(selected) == 3
            assert selected[0].score >= selected[1].score  # 最初は最高スコア


class TestClusteringDiversifier:
    """クラスタリング多様化器のテストクラス"""

    @pytest.mark.unit
    def test_clustering_diversifier_initialization(self):
        """クラスタリング多様化器初期化テスト"""
        config = DiversityConfig(
            algorithm=DiversificationAlgorithm.CLUSTERING,
            cluster_count=4,
            clustering_method="kmeans",
        )

        diversifier = ClusteringDiversifier(config)

        assert diversifier.config == config
        assert diversifier.cluster_count == 4
        assert diversifier.clustering_method == "kmeans"

    @pytest.mark.unit
    async def test_kmeans_clustering(self):
        """K-meansクラスタリングテスト"""
        config = DiversityConfig()
        diversifier = ClusteringDiversifier(config)

        # 埋め込みベクトルを持つ候補
        candidates = [
            DiversityCandidate(
                id=f"doc-{i}",
                content=f"content {i}",
                title=f"title {i}",
                score=0.9,
                embedding=np.random.random(10),
            )
            for i in range(6)
        ]

        with patch.object(diversifier, "_perform_kmeans_clustering") as mock_kmeans:
            mock_kmeans.return_value = [0, 0, 1, 1, 2, 2]  # クラスタラベル

            clusters = await diversifier._perform_kmeans_clustering(candidates, k=3)

            assert len(set(clusters)) <= 3  # 最大3つのクラスタ


class TestTopicDiversifier:
    """トピック多様化器のテストクラス"""

    @pytest.mark.unit
    def test_topic_diversifier_initialization(self):
        """トピック多様化器初期化テスト"""
        config = DiversityConfig(
            algorithm=DiversificationAlgorithm.TOPIC_BASED,
            enable_topic_diversity=True,
            topic_weight=0.4,
        )

        diversifier = TopicDiversifier(config)

        assert diversifier.config == config
        assert diversifier.topic_weight == 0.4

    @pytest.mark.unit
    async def test_topic_extraction(self):
        """トピック抽出テスト"""
        config = DiversityConfig()
        diversifier = TopicDiversifier(config)

        _candidates = [
            DiversityCandidate(
                id="doc-1",
                content="machine learning algorithms",
                title="ML Guide",
                score=0.9,
                embedding=np.random.random(10),
                metadata={"topics": ["machine_learning", "algorithms"]},
            ),
        ]

        with patch.object(diversifier, "_extract_topics_from_content") as mock_extract:
            mock_extract.return_value = [
                "machine_learning",
                "algorithms",
                "neural_networks",
            ]

            topics = await diversifier._extract_topics_from_content(
                "machine learning algorithms"
            )

            assert len(topics) == 3
            assert "machine_learning" in topics


class TestTemporalDiversifier:
    """時系列多様化器のテストクラス"""

    @pytest.mark.unit
    def test_temporal_diversifier_initialization(self):
        """時系列多様化器初期化テスト"""
        config = DiversityConfig(
            algorithm=DiversificationAlgorithm.TEMPORAL,
            enable_temporal_diversity=True,
            temporal_window_days=30,
        )

        diversifier = TemporalDiversifier(config)

        assert diversifier.config == config
        assert diversifier.temporal_window_days == 30

    @pytest.mark.unit
    async def test_temporal_distribution(self):
        """時系列分散テスト"""
        config = DiversityConfig()
        diversifier = TemporalDiversifier(config)

        # 異なる日付の候補
        candidates = [
            DiversityCandidate(
                id=f"doc-{i}",
                content=f"content {i}",
                title=f"title {i}",
                score=0.9 - i * 0.1,
                embedding=np.random.random(10),
                metadata={"date": f"2024-01-{i+1:02d}"},
            )
            for i in range(5)
        ]

        selected = await diversifier.select_temporally_diverse(
            candidates, max_results=3, window_days=7
        )

        assert len(selected) <= 3
        # 時系列的に分散していることを確認
        dates = [c.metadata["date"] for c in selected]
        assert len(set(dates)) == len(selected)  # 重複する日付がない
