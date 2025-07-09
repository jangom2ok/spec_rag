"""検索結果多様性制御サービス

TDD実装：MMR・クラスタリングベースの結果多様化機能
- MMR (Maximal Marginal Relevance): 関連性と多様性のバランス
- クラスタリング: K-means・階層クラスタリングベース
- トピック多様性: トピック分散による多様化
- 時系列多様性: 時間軸での多様化
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
from sklearn.cluster import KMeans  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

logger = logging.getLogger(__name__)


class DiversificationAlgorithm(str, Enum):
    """多様化アルゴリズム"""

    MMR = "mmr"
    CLUSTERING = "clustering"
    TOPIC_BASED = "topic_based"
    TEMPORAL = "temporal"
    HYBRID = "hybrid"


@dataclass
class DiversityConfig:
    """多様性制御設定"""

    algorithm: DiversificationAlgorithm = DiversificationAlgorithm.MMR
    diversity_factor: float = 0.5  # λパラメータ（0: 多様性重視, 1: 関連性重視）
    max_results: int = 10
    similarity_threshold: float = 0.8

    # クラスタリング設定
    enable_clustering: bool = True
    cluster_count: int = 3
    clustering_method: str = "kmeans"
    min_cluster_size: int = 1

    # トピック多様性設定
    enable_topic_diversity: bool = True
    topic_weight: float = 0.3
    max_topics_per_result: int = 5

    # 時系列多様性設定
    enable_temporal_diversity: bool = False
    temporal_window_days: int = 30
    temporal_weight: float = 0.2

    # キャッシュ設定
    enable_caching: bool = True
    cache_ttl: int = 3600

    # タイムアウト設定
    timeout: float = 30.0

    def __post_init__(self):
        """設定値のバリデーション"""
        if not 0 <= self.diversity_factor <= 1:
            raise ValueError("diversity_factor must be between 0 and 1")
        if self.max_results <= 0:
            raise ValueError("max_results must be greater than 0")
        if self.cluster_count <= 0:
            raise ValueError("cluster_count must be greater than 0")


@dataclass
class DiversificationRequest:
    """多様化リクエスト"""

    query: str
    candidates: list["DiversityCandidate"]
    max_results: int | None = None
    diversity_factor: float | None = None
    preserve_top_results: int = 0  # 上位n件を必ず含める
    diversification_criteria: list[str] = field(
        default_factory=lambda: ["topic", "category"]
    )


@dataclass
class DiversityCandidate:
    """多様性候補"""

    id: str
    content: str
    title: str
    score: float
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """後処理"""
        if self.embedding is None:
            # デフォルトの埋め込みベクトルを生成（実際はembedding serviceから取得）
            self.embedding = np.random.random(128)


@dataclass
class ClusterResult:
    """クラスタ結果"""

    cluster_id: int
    candidates: list[DiversityCandidate]
    centroid: np.ndarray
    coherence_score: float = 0.0


@dataclass
class DiversificationResult:
    """多様化結果"""

    success: bool
    query: str
    diversified_candidates: list[DiversityCandidate]
    diversification_time: float
    original_count: int
    error_message: str | None = None
    cache_hit: bool = False
    diversity_metrics: dict[str, float] = field(default_factory=dict)

    def get_summary(self) -> dict[str, Any]:
        """多様化結果のサマリーを取得"""
        return {
            "success": self.success,
            "query": self.query,
            "original_count": self.original_count,
            "diversified_count": len(self.diversified_candidates),
            "diversification_time": self.diversification_time,
            "cache_hit": self.cache_hit,
            "diversity_metrics": self.diversity_metrics,
        }


class BaseDiversifier:
    """多様化器ベースクラス"""

    def __init__(self, config: DiversityConfig):
        self.config = config

    async def select(
        self, candidates: list[DiversityCandidate], **kwargs: Any
    ) -> list[DiversityCandidate]:
        """多様化選択（オーバーライド必須）"""
        raise NotImplementedError


class MMRDiversifier(BaseDiversifier):
    """MMR多様化器"""

    def __init__(self, config: DiversityConfig):
        super().__init__(config)
        self.diversity_factor = config.diversity_factor

    async def select(
        self,
        candidates: list[DiversityCandidate],
        max_results: int | None = None,
        diversity_factor: float | None = None,
        **kwargs: Any,
    ) -> list[DiversityCandidate]:
        """MMRアルゴリズムによる多様化選択"""
        max_results = max_results or self.config.max_results
        diversity_factor = diversity_factor or self.diversity_factor

        if not candidates:
            return []

        # 類似度行列を計算
        similarity_matrix = await self._calculate_similarity_matrix(candidates)

        selected = []
        remaining = candidates.copy()

        # 最初は最高スコアの候補を選択
        first_candidate = max(remaining, key=lambda x: x.score)
        selected.append(first_candidate)
        remaining.remove(first_candidate)

        # 残りの候補をMMRで選択
        while len(selected) < max_results and remaining:
            best_mmr_score = -float("inf")
            best_candidate: DiversityCandidate | None = None

            for candidate in remaining:
                mmr_score = self._calculate_mmr_score(
                    candidate, selected, diversity_factor, similarity_matrix, candidates
                )

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break

        return selected

    def _calculate_mmr_score(
        self,
        candidate: DiversityCandidate,
        selected: list[DiversityCandidate],
        diversity_factor: float,
        similarity_matrix: np.ndarray = None,
        all_candidates: list[DiversityCandidate] = None,
    ) -> float:
        """MMRスコア計算"""
        relevance_score = candidate.score

        if not selected:
            return relevance_score

        # 選択済み候補との最大類似度を計算
        max_similarity = 0.0
        if similarity_matrix is not None and all_candidates:
            candidate_idx = all_candidates.index(candidate)
            for selected_candidate in selected:
                selected_idx = all_candidates.index(selected_candidate)
                similarity = similarity_matrix[candidate_idx][selected_idx]
                max_similarity = max(max_similarity, similarity)
        else:
            # フォールバック: 直接計算
            for selected_candidate in selected:
                similarity = self._calculate_similarity(candidate, selected_candidate)
                max_similarity = max(max_similarity, similarity)

        # MMR = λ * relevance - (1-λ) * max_similarity
        mmr_score = (
            diversity_factor * relevance_score - (1 - diversity_factor) * max_similarity
        )
        return mmr_score

    def _calculate_similarity(
        self, candidate1: DiversityCandidate, candidate2: DiversityCandidate
    ) -> float:
        """2つの候補間の類似度計算"""
        if candidate1.embedding is not None and candidate2.embedding is not None:
            # コサイン類似度
            similarity = np.dot(candidate1.embedding, candidate2.embedding) / (
                np.linalg.norm(candidate1.embedding)
                * np.linalg.norm(candidate2.embedding)
            )
            return max(0.0, min(1.0, similarity))

        # フォールバック: テキスト類似度（簡易実装）
        words1 = set(candidate1.content.lower().split())
        words2 = set(candidate2.content.lower().split())

        if len(words1) == 0 and len(words2) == 0:
            return 1.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    async def _calculate_similarity_matrix(
        self, candidates: list[DiversityCandidate]
    ) -> np.ndarray:
        """類似度行列計算"""
        n = len(candidates)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    similarity = self._calculate_similarity(
                        candidates[i], candidates[j]
                    )
                    similarity_matrix[i][j] = similarity

        return similarity_matrix


class ClusteringDiversifier(BaseDiversifier):
    """クラスタリング多様化器"""

    def __init__(self, config: DiversityConfig):
        super().__init__(config)
        self.cluster_count = config.cluster_count
        self.clustering_method = config.clustering_method
        self.min_cluster_size = config.min_cluster_size

    async def select(
        self,
        candidates: list[DiversityCandidate],
        max_results: int | None = None,
        cluster_count: int | None = None,
        **kwargs: Any,
    ) -> list[DiversityCandidate]:
        """クラスタリングによる多様化選択"""
        max_results = max_results or self.config.max_results
        cluster_count = cluster_count or self.cluster_count

        if not candidates or len(candidates) <= cluster_count:
            return candidates[:max_results]

        # クラスタリング実行
        clusters = await self._cluster_candidates(candidates, cluster_count)

        # 各クラスターから代表候補を選択
        selected: list[DiversityCandidate] = []
        clusters_sorted = sorted(
            clusters, key=lambda c: len(c.candidates), reverse=True
        )

        for cluster in clusters_sorted:
            if len(selected) >= max_results:
                break

            if len(cluster.candidates) >= self.min_cluster_size:
                # クラスター内で最高スコアの候補を選択
                best_candidate = max(cluster.candidates, key=lambda x: x.score)
                selected.append(best_candidate)

        # 必要に応じて残りの候補で埋める
        remaining_slots = max_results - len(selected)
        if remaining_slots > 0:
            remaining_candidates = [c for c in candidates if c not in selected]
            remaining_candidates.sort(key=lambda x: x.score, reverse=True)
            selected.extend(remaining_candidates[:remaining_slots])

        return selected

    async def _cluster_candidates(
        self, candidates: list[DiversityCandidate], k: int
    ) -> list[ClusterResult]:
        """候補をクラスタリング"""
        if self.clustering_method == "kmeans":
            return await self._perform_kmeans_clustering(candidates, k)
        else:
            # デフォルトでk-meansを使用
            return await self._perform_kmeans_clustering(candidates, k)

    async def _perform_kmeans_clustering(
        self, candidates: list[DiversityCandidate], k: int
    ) -> list[ClusterResult]:
        """K-meansクラスタリング実行"""
        # 埋め込みベクトルを抽出
        embeddings = np.array([c.embedding for c in candidates])

        # K-meansクラスタリング
        try:
            kmeans = KMeans(
                n_clusters=min(k, len(candidates)), random_state=42, n_init=10
            )
            cluster_labels = kmeans.fit_predict(embeddings)
            centroids = kmeans.cluster_centers_
        except Exception as e:
            logger.warning(f"K-means clustering failed: {e}")
            # フォールバック: 単純な分割
            cluster_labels = [i % k for i in range(len(candidates))]
            centroids = [np.mean(embeddings, axis=0) for _ in range(k)]

        # クラスタ結果を構築
        clusters = []
        for cluster_id in range(k):
            cluster_candidates = [
                candidates[i]
                for i, label in enumerate(cluster_labels)
                if label == cluster_id
            ]

            if cluster_candidates:
                centroid = (
                    centroids[cluster_id]
                    if cluster_id < len(centroids)
                    else np.mean(
                        [
                            c.embedding
                            for c in cluster_candidates
                            if c.embedding is not None
                        ],
                        axis=0,
                    )
                )

                # コヒーレンススコア計算（クラスター内類似度）
                coherence_score = self._calculate_cluster_coherence(cluster_candidates)

                cluster = ClusterResult(
                    cluster_id=cluster_id,
                    candidates=cluster_candidates,
                    centroid=centroid,
                    coherence_score=coherence_score,
                )
                clusters.append(cluster)

        return clusters

    def _calculate_cluster_coherence(
        self, candidates: list[DiversityCandidate]
    ) -> float:
        """クラスターコヒーレンススコア計算"""
        if len(candidates) <= 1:
            return 1.0

        embeddings = np.array([c.embedding for c in candidates])
        similarities = cosine_similarity(embeddings)

        # 対角成分を除いた平均類似度
        n = len(similarities)
        total_similarity = np.sum(similarities) - np.trace(similarities)
        avg_similarity = total_similarity / (n * n - n) if n > 1 else 0.0

        return max(0.0, min(1.0, avg_similarity))


class TopicDiversifier(BaseDiversifier):
    """トピック多様化器"""

    def __init__(self, config: DiversityConfig):
        super().__init__(config)
        self.topic_weight = config.topic_weight
        self.max_topics_per_result = config.max_topics_per_result

    async def select(
        self,
        candidates: list[DiversityCandidate],
        max_results: int | None = None,
        topic_weight: float | None = None,
        **kwargs: Any,
    ) -> list[DiversityCandidate]:
        """トピックベース多様化選択"""
        max_results = max_results or self.config.max_results
        topic_weight = topic_weight or self.topic_weight

        if not candidates:
            return []

        # トピック抽出
        topics_dict = await self._extract_topics(candidates)

        # トピック多様性スコア計算
        selected: list[DiversityCandidate] = []
        remaining = candidates.copy()
        covered_topics: set[str] = set()

        while len(selected) < max_results and remaining:
            best_score = -float("inf")
            best_candidate = None

            for candidate in remaining:
                candidate_topics = set(topics_dict.get(candidate.id, []))

                # 新規トピックボーナス
                new_topics = candidate_topics - covered_topics
                topic_novelty = len(new_topics) / max(len(candidate_topics), 1)

                # 総合スコア = 関連性スコア + トピック新規性
                total_score = (
                    1 - topic_weight
                ) * candidate.score + topic_weight * topic_novelty

                if total_score > best_score:
                    best_score = total_score
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
                covered_topics.update(topics_dict.get(best_candidate.id, []))
            else:
                break

        return selected

    async def _extract_topics(
        self, candidates: list[DiversityCandidate]
    ) -> dict[str, list[str]]:
        """候補からトピック抽出"""
        topics_dict = {}

        for candidate in candidates:
            # メタデータからトピックを取得
            if "topics" in candidate.metadata:
                topics_dict[candidate.id] = candidate.metadata["topics"]
            else:
                # テキストからトピック抽出
                extracted_topics = await self._extract_topics_from_content(
                    candidate.content
                )
                topics_dict[candidate.id] = extracted_topics

        return topics_dict

    async def _extract_topics_from_content(self, content: str) -> list[str]:
        """コンテンツからトピック抽出（簡易実装）"""
        # 実際の実装では NLP ライブラリ（spaCy, NLTK等）を使用
        keywords = {
            "machine_learning": ["machine learning", "ml", "algorithm", "model"],
            "deep_learning": ["deep learning", "neural network", "cnn", "rnn"],
            "natural_language_processing": ["nlp", "text analysis", "language model"],
            "computer_vision": ["computer vision", "image", "visual", "opencv"],
            "data_science": ["data science", "statistics", "analysis", "visualization"],
            "artificial_intelligence": ["ai", "artificial intelligence", "intelligent"],
            "programming": ["programming", "code", "software", "development"],
            "python": ["python", "pandas", "numpy", "scikit"],
        }

        content_lower = content.lower()
        extracted_topics = []

        for topic, terms in keywords.items():
            if any(term in content_lower for term in terms):
                extracted_topics.append(topic)

        return extracted_topics[: self.max_topics_per_result]

    def _calculate_topic_diversity_score(
        self, topics_dict: dict[str, list[str]]
    ) -> float:
        """トピック多様性スコア計算"""
        all_topics = set()
        for topics in topics_dict.values():
            all_topics.update(topics)

        if not all_topics:
            return 0.0

        # ユニークトピック数 / 全候補数
        unique_topic_ratio = len(all_topics) / len(topics_dict) if topics_dict else 0.0
        return min(1.0, unique_topic_ratio)


class TemporalDiversifier(BaseDiversifier):
    """時系列多様化器"""

    def __init__(self, config: DiversityConfig):
        super().__init__(config)
        self.temporal_window_days = config.temporal_window_days
        self.temporal_weight = config.temporal_weight

    async def select_temporally_diverse(
        self,
        candidates: list[DiversityCandidate],
        max_results: int = None,
        window_days: int = None,
    ) -> list[DiversityCandidate]:
        """時系列多様化選択"""
        max_results = max_results or self.config.max_results
        window_days = window_days or self.temporal_window_days

        if not candidates:
            return []

        # 日付でソート
        candidates_with_dates = []
        for candidate in candidates:
            date_str = candidate.metadata.get("date", "")
            try:
                date_obj = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                candidates_with_dates.append((candidate, date_obj))
            except (ValueError, AttributeError):
                # 日付がない場合は現在時刻を使用
                candidates_with_dates.append((candidate, datetime.now()))

        # スコア順でソート
        candidates_with_dates.sort(key=lambda x: x[0].score, reverse=True)

        selected: list[DiversityCandidate] = []
        used_dates: set[datetime] = set()

        for candidate, date_obj in candidates_with_dates:
            if len(selected) >= max_results:
                break

            # 時間窓内に同じ日付がないかチェック
            date_conflict = False
            for used_date in used_dates:
                if abs((date_obj - used_date).days) < window_days:
                    date_conflict = True
                    break

            if not date_conflict:
                selected.append(candidate)
                used_dates.add(date_obj)

        # 必要に応じて残りの候補で埋める
        if len(selected) < max_results:
            remaining_candidates = [
                c for c, _ in candidates_with_dates if c not in selected
            ]
            remaining_slots = max_results - len(selected)
            selected.extend(remaining_candidates[:remaining_slots])

        return selected


class SearchDiversityService:
    """検索結果多様性制御メインサービス"""

    def __init__(self, config: DiversityConfig):
        self.config = config
        self.diversifiers = self._create_diversifiers()
        self.cache: dict[str, tuple[DiversificationResult, datetime]] = (
            {}
        )  # 簡易キャッシュ実装

    def _create_diversifiers(self) -> dict[DiversificationAlgorithm, BaseDiversifier]:
        """多様化器インスタンス作成"""
        diversifiers: dict[DiversificationAlgorithm, BaseDiversifier] = {}

        diversifiers[DiversificationAlgorithm.MMR] = MMRDiversifier(self.config)
        diversifiers[DiversificationAlgorithm.CLUSTERING] = ClusteringDiversifier(
            self.config
        )
        diversifiers[DiversificationAlgorithm.TOPIC_BASED] = TopicDiversifier(
            self.config
        )
        diversifiers[DiversificationAlgorithm.TEMPORAL] = TemporalDiversifier(
            self.config
        )

        return diversifiers

    async def diversify(self, request: DiversificationRequest) -> DiversificationResult:
        """多様化実行"""
        start_time = datetime.now()

        try:
            # 入力バリデーション
            if not request.candidates:
                return DiversificationResult(
                    success=True,
                    query=request.query,
                    diversified_candidates=[],
                    diversification_time=0.0,
                    original_count=0,
                )

            # キャッシュチェック
            if self.config.enable_caching:
                cache_key = self._get_cache_key(request)
                cached_result = await self._get_from_cache(cache_key)
                if cached_result:
                    return cached_result

            # 設定値のマージ
            max_results = request.max_results or self.config.max_results
            diversity_factor = request.diversity_factor or self.config.diversity_factor

            # 上位結果の保持
            preserved_candidates = []
            if request.preserve_top_results > 0:
                sorted_candidates = sorted(
                    request.candidates, key=lambda x: x.score, reverse=True
                )
                preserved_candidates = sorted_candidates[: request.preserve_top_results]
                remaining_candidates = sorted_candidates[request.preserve_top_results :]
            else:
                remaining_candidates = request.candidates

            # アルゴリズムに応じた多様化実行
            if self.config.algorithm == DiversificationAlgorithm.MMR:
                diversified = await self._diversify_with_mmr(
                    remaining_candidates,
                    max_results - len(preserved_candidates),
                    diversity_factor,
                )
            elif self.config.algorithm == DiversificationAlgorithm.CLUSTERING:
                diversified = await self._diversify_with_clustering(
                    remaining_candidates,
                    max_results - len(preserved_candidates),
                    self.config.cluster_count,
                )
            elif self.config.algorithm == DiversificationAlgorithm.TOPIC_BASED:
                diversified = await self._diversify_with_topics(
                    remaining_candidates,
                    max_results - len(preserved_candidates),
                    self.config.topic_weight,
                )
            elif self.config.algorithm == DiversificationAlgorithm.TEMPORAL:
                diversified = await self._diversify_with_temporal(
                    remaining_candidates,
                    max_results - len(preserved_candidates),
                    self.config.temporal_window_days,
                )
            else:
                # デフォルトでMMRを使用
                diversified = await self._diversify_with_mmr(
                    remaining_candidates,
                    max_results - len(preserved_candidates),
                    diversity_factor,
                )

            # 保持された上位結果と多様化結果をマージ
            final_candidates = preserved_candidates + diversified

            # 多様性メトリクス計算
            diversity_metrics = self._calculate_diversity_metrics(
                final_candidates, request.candidates
            )

            end_time = datetime.now()
            diversification_time = (end_time - start_time).total_seconds()

            result = DiversificationResult(
                success=True,
                query=request.query,
                diversified_candidates=final_candidates,
                diversification_time=diversification_time,
                original_count=len(request.candidates),
                diversity_metrics=diversity_metrics,
            )

            # キャッシュに保存
            if self.config.enable_caching:
                await self._set_cache(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Diversification failed: {e}")
            end_time = datetime.now()
            diversification_time = (end_time - start_time).total_seconds()

            return DiversificationResult(
                success=False,
                query=request.query,
                diversified_candidates=[],
                diversification_time=diversification_time,
                original_count=len(request.candidates),
                error_message=str(e),
            )

    async def _diversify_with_mmr(
        self,
        candidates: list[DiversityCandidate],
        max_results: int,
        diversity_factor: float,
    ) -> list[DiversityCandidate]:
        """MMRによる多様化"""
        diversifier = self.diversifiers[DiversificationAlgorithm.MMR]
        assert isinstance(diversifier, MMRDiversifier)
        return await diversifier.select(candidates, max_results, diversity_factor)

    async def _diversify_with_clustering(
        self, candidates: list[DiversityCandidate], max_results: int, cluster_count: int
    ) -> list[DiversityCandidate]:
        """クラスタリングによる多様化"""
        diversifier = self.diversifiers[DiversificationAlgorithm.CLUSTERING]
        assert isinstance(diversifier, ClusteringDiversifier)
        return await diversifier.select(candidates, max_results, cluster_count)

    async def _diversify_with_topics(
        self,
        candidates: list[DiversityCandidate],
        max_results: int,
        topic_weight: float,
    ) -> list[DiversityCandidate]:
        """トピックによる多様化"""
        diversifier = self.diversifiers[DiversificationAlgorithm.TOPIC_BASED]
        assert isinstance(diversifier, TopicDiversifier)
        return await diversifier.select(candidates, max_results, topic_weight)

    async def _diversify_with_temporal(
        self,
        candidates: list[DiversityCandidate],
        max_results: int,
        time_window_days: int,
    ) -> list[DiversityCandidate]:
        """時系列による多様化"""
        diversifier = self.diversifiers[DiversificationAlgorithm.TEMPORAL]
        assert isinstance(diversifier, TemporalDiversifier)
        return await diversifier.select_temporally_diverse(
            candidates, max_results, time_window_days
        )

    async def _calculate_similarity_matrix(
        self, candidates: list[DiversityCandidate]
    ) -> np.ndarray:
        """類似度行列計算"""
        mmr_diversifier = self.diversifiers[DiversificationAlgorithm.MMR]
        assert isinstance(mmr_diversifier, MMRDiversifier)
        return await mmr_diversifier._calculate_similarity_matrix(candidates)

    def _calculate_diversity_metrics(
        self,
        selected_candidates: list[DiversityCandidate],
        original_candidates: list[DiversityCandidate],
    ) -> dict[str, float]:
        """多様性メトリクス計算"""
        if not selected_candidates:
            return {
                "intra_list_diversity": 0.0,
                "coverage_ratio": 0.0,
                "novelty_score": 0.0,
                "redundancy_score": 1.0,
            }

        # リスト内多様性（平均ペアワイズ非類似度）
        total_dissimilarity = 0.0
        pairs_count = 0

        for i in range(len(selected_candidates)):
            for j in range(i + 1, len(selected_candidates)):
                similarity = self._calculate_similarity(
                    selected_candidates[i], selected_candidates[j]
                )
                dissimilarity = 1.0 - similarity
                total_dissimilarity += dissimilarity
                pairs_count += 1

        intra_list_diversity = (
            total_dissimilarity / pairs_count if pairs_count > 0 else 0.0
        )

        # カバレッジ比率（異なるカテゴリの割合）
        selected_categories = set()
        original_categories = set()

        for candidate in selected_candidates:
            category = candidate.metadata.get("category", "unknown")
            selected_categories.add(category)

        for candidate in original_candidates:
            category = candidate.metadata.get("category", "unknown")
            original_categories.add(category)

        coverage_ratio = (
            len(selected_categories) / len(original_categories)
            if original_categories
            else 0.0
        )

        # 新規性スコア（トピック新規性）
        selected_topics = set()
        for candidate in selected_candidates:
            topics = candidate.metadata.get("topics", [])
            selected_topics.update(topics)

        novelty_score = (
            len(selected_topics) / len(selected_candidates)
            if selected_candidates
            else 0.0
        )
        novelty_score = min(
            1.0, novelty_score / 3.0
        )  # 正規化（平均3トピック/候補と仮定）

        # 冗長性スコア（1 - 多様性）
        redundancy_score = 1.0 - intra_list_diversity

        return {
            "intra_list_diversity": intra_list_diversity,
            "coverage_ratio": coverage_ratio,
            "novelty_score": novelty_score,
            "redundancy_score": redundancy_score,
        }

    def _calculate_similarity(
        self, candidate1: DiversityCandidate, candidate2: DiversityCandidate
    ) -> float:
        """候補間類似度計算"""
        mmr_diversifier = self.diversifiers[DiversificationAlgorithm.MMR]
        assert isinstance(mmr_diversifier, MMRDiversifier)
        return mmr_diversifier._calculate_similarity(candidate1, candidate2)

    def _calculate_diversity_scores(
        self, candidates: list[DiversityCandidate], criteria: list[str]
    ) -> dict[str, dict[str, float]]:
        """多様性基準スコア計算"""
        diversity_scores = {}

        for candidate in candidates:
            scores = {}

            for criterion in criteria:
                if criterion == "category":
                    # カテゴリ多様性
                    scores["category"] = self._calculate_category_diversity_score(
                        candidate, candidates
                    )
                elif criterion == "author":
                    # 著者多様性
                    scores["author"] = self._calculate_author_diversity_score(
                        candidate, candidates
                    )
                elif criterion == "topic":
                    # トピック多様性
                    scores["topic"] = (
                        self._calculate_topic_diversity_score_for_candidate(
                            candidate, candidates
                        )
                    )

            diversity_scores[candidate.id] = scores

        return diversity_scores

    def _calculate_category_diversity_score(
        self, candidate: DiversityCandidate, all_candidates: list[DiversityCandidate]
    ) -> float:
        """カテゴリ多様性スコア計算"""
        candidate_category = candidate.metadata.get("category", "unknown")
        same_category_count = sum(
            1
            for c in all_candidates
            if c.metadata.get("category", "unknown") == candidate_category
        )

        # 希少なカテゴリほど高スコア
        diversity_score = 1.0 - (same_category_count / len(all_candidates))
        return max(0.0, diversity_score)

    def _calculate_author_diversity_score(
        self, candidate: DiversityCandidate, all_candidates: list[DiversityCandidate]
    ) -> float:
        """著者多様性スコア計算"""
        candidate_author = candidate.metadata.get("author", "unknown")
        same_author_count = sum(
            1
            for c in all_candidates
            if c.metadata.get("author", "unknown") == candidate_author
        )

        # 希少な著者ほど高スコア
        diversity_score = 1.0 - (same_author_count / len(all_candidates))
        return max(0.0, diversity_score)

    def _calculate_topic_diversity_score_for_candidate(
        self, candidate: DiversityCandidate, all_candidates: list[DiversityCandidate]
    ) -> float:
        """候補のトピック多様性スコア計算"""
        candidate_topics = set(candidate.metadata.get("topics", []))

        if not candidate_topics:
            return 0.0

        # 他の候補との重複トピック数を計算
        total_overlap = 0
        for other_candidate in all_candidates:
            if other_candidate.id != candidate.id:
                other_topics = set(other_candidate.metadata.get("topics", []))
                overlap = len(candidate_topics.intersection(other_topics))
                total_overlap += overlap

        # 重複が少ないほど高スコア
        max_possible_overlap = len(candidate_topics) * (len(all_candidates) - 1)
        diversity_score = (
            1.0 - (total_overlap / max_possible_overlap)
            if max_possible_overlap > 0
            else 1.0
        )

        return max(0.0, diversity_score)

    def _get_cache_key(self, request: DiversificationRequest) -> str:
        """キャッシュキー生成"""
        cache_content = {
            "query": request.query,
            "candidate_ids": [c.id for c in request.candidates],
            "max_results": request.max_results or self.config.max_results,
            "diversity_factor": request.diversity_factor
            or self.config.diversity_factor,
            "preserve_top_results": request.preserve_top_results,
            "diversification_criteria": request.diversification_criteria,
            "algorithm": self.config.algorithm.value,
        }
        content_str = json.dumps(cache_content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

    async def _get_from_cache(self, cache_key: str) -> DiversificationResult | None:
        """キャッシュから取得"""
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.config.cache_ttl):
                # DiversificationResultオブジェクトを正しくコピー
                result = DiversificationResult(
                    success=cached_data.success,
                    query=cached_data.query,
                    diversified_candidates=cached_data.diversified_candidates.copy(),
                    diversification_time=cached_data.diversification_time,
                    original_count=cached_data.original_count,
                    error_message=cached_data.error_message,
                    cache_hit=True,
                    diversity_metrics=cached_data.diversity_metrics.copy(),
                )
                return result
            else:
                # 期限切れキャッシュを削除
                del self.cache[cache_key]
        return None

    async def _set_cache(self, cache_key: str, result: DiversificationResult) -> None:
        """キャッシュに保存"""
        self.cache[cache_key] = (result, datetime.now())
