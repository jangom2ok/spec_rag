"""ハイブリッド検索エンジンサービス

TDD実装：BGE-M3を活用したdense/sparse vectorsハイブリッド検索
- Dense Vector Search (セマンティック検索)
- Sparse Vector Search (キーワード検索)
- Reciprocal Rank Fusion (RRF)
- フィルタリング・ファセット機能
"""

import asyncio
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from app.repositories.chunk_repository import DocumentChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    """検索モード"""

    HYBRID = "hybrid"
    SEMANTIC = "semantic"  # Dense vector重視
    KEYWORD = "keyword"  # Sparse vector重視


class RankingAlgorithm(str, Enum):
    """ランキングアルゴリズム"""

    RRF = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED_SUM = "weighted_sum"
    BORDA_COUNT = "borda_count"


@dataclass
class SearchConfig:
    """検索設定"""

    # ハイブリッド検索重み
    dense_weight: float = 0.7
    sparse_weight: float = 0.3

    # 検索パラメータ
    top_k: int = 10
    search_mode: SearchMode = SearchMode.HYBRID
    ranking_algorithm: RankingAlgorithm = RankingAlgorithm.RRF

    # Vector検索パラメータ
    dense_search_params: dict[str, Any] = field(
        default_factory=lambda: {
            "metric_type": "IP",  # Inner Product
            "nprobe": 16,
        }
    )
    sparse_search_params: dict[str, Any] = field(
        default_factory=lambda: {
            "drop_ratio_search": 0.2,
        }
    )

    # 結果フィルタリング
    similarity_threshold: float = 0.0
    enable_reranking: bool = True
    rerank_top_k: int = 100

    # パフォーマンス設定
    search_timeout: float = 30.0
    max_concurrent_searches: int = 3

    def __post_init__(self):
        """設定値のバリデーション"""
        if abs(self.dense_weight + self.sparse_weight - 1.0) > 0.001:
            raise ValueError("dense_weight + sparse_weight must equal 1.0")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            raise ValueError("similarity_threshold must be between 0 and 1")


@dataclass
class SearchFilter:
    """検索フィルター"""

    field: str
    value: Any
    operator: str = "eq"  # eq, ne, gt, lt, gte, lte, in, not_in


@dataclass
class FacetResult:
    """ファセット結果"""

    value: str
    count: int


@dataclass
class SearchQuery:
    """検索クエリ"""

    text: str
    filters: list[SearchFilter] = field(default_factory=list)
    facets: list[str] = field(default_factory=list)
    search_mode: SearchMode = SearchMode.HYBRID
    max_results: int = 10
    offset: int = 0


@dataclass
class VectorSearchResult:
    """ベクター検索結果"""

    ids: list[str]
    scores: list[float]
    search_time: float
    total_hits: int


@dataclass
class SearchResult:
    """検索結果"""

    success: bool
    documents: list[dict[str, Any]]
    total_hits: int
    search_time: float
    query: str
    facets: dict[str, list[FacetResult]] | None = None
    error_message: str | None = None

    def get_summary(self) -> dict[str, Any]:
        """検索結果のサマリーを取得"""
        return {
            "success": self.success,
            "total_hits": self.total_hits,
            "returned_count": len(self.documents),
            "search_time": self.search_time,
            "query": self.query,
            "has_facets": self.facets is not None,
        }


class RecipriocalRankFusion:
    """Reciprocal Rank Fusion (RRF) 実装"""

    @staticmethod
    def fuse_rankings(
        dense_results: VectorSearchResult,
        sparse_results: VectorSearchResult,
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """RRFを使用してランキングを融合"""

        # Dense結果のRRFスコア計算
        dense_scores = {}
        for i, doc_id in enumerate(dense_results.ids):
            dense_scores[doc_id] = 1.0 / (k + i + 1)

        # Sparse結果のRRFスコア計算
        sparse_scores = {}
        for i, doc_id in enumerate(sparse_results.ids):
            sparse_scores[doc_id] = 1.0 / (k + i + 1)

        # 全ドキュメントIDを収集
        all_doc_ids = set(dense_results.ids + sparse_results.ids)

        # RRFスコアを結合
        fused_scores = []
        for doc_id in all_doc_ids:
            dense_score = dense_scores.get(doc_id, 0.0)
            sparse_score = sparse_scores.get(doc_id, 0.0)
            combined_score = dense_score + sparse_score
            fused_scores.append((doc_id, combined_score))

        # スコア降順でソート
        fused_scores.sort(key=lambda x: x[1], reverse=True)

        return fused_scores


class HybridSearchEngine:
    """ハイブリッド検索エンジンメインクラス"""

    def __init__(
        self,
        config: SearchConfig,
        embedding_service: EmbeddingService | None = None,
        document_repository: DocumentRepository | None = None,
        chunk_repository: DocumentChunkRepository | None = None,
    ):
        self.config = config
        self.embedding_service = embedding_service
        self.document_repository = document_repository
        self.chunk_repository = chunk_repository
        self.rrf = RecipriocalRankFusion()

    async def search(self, query: SearchQuery) -> SearchResult:
        """ハイブリッド検索を実行"""
        start_time = datetime.now()

        try:
            # 入力バリデーション
            if not query.text.strip():
                return SearchResult(
                    success=False,
                    documents=[],
                    total_hits=0,
                    search_time=0.0,
                    query=query.text,
                    error_message="Query text is empty",
                )

            # タイムアウト設定
            search_timeout = self.config.search_timeout

            # クエリ埋め込み生成
            try:
                embeddings = await asyncio.wait_for(
                    self._generate_query_embeddings(query.text),
                    timeout=search_timeout / 3,
                )
            except TimeoutError:
                return SearchResult(
                    success=False,
                    documents=[],
                    total_hits=0,
                    search_time=(datetime.now() - start_time).total_seconds(),
                    query=query.text,
                    error_message="Query embedding generation timeout",
                )

            # 並行でDense/Sparse検索を実行
            search_tasks = []

            if self.config.search_mode in [SearchMode.HYBRID, SearchMode.SEMANTIC]:
                dense_task = self._search_dense_vectors(embeddings["dense"], query)
                search_tasks.append(("dense", dense_task))

            if self.config.search_mode in [SearchMode.HYBRID, SearchMode.KEYWORD]:
                sparse_task = self._search_sparse_vectors(embeddings["sparse"], query)
                search_tasks.append(("sparse", sparse_task))

            # 検索結果を並行取得
            search_results = {}
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*[task for _, task in search_tasks]),
                    timeout=search_timeout * 2 / 3,
                )

                for i, (search_type, _) in enumerate(search_tasks):
                    search_results[search_type] = results[i]

            except TimeoutError:
                return SearchResult(
                    success=False,
                    documents=[],
                    total_hits=0,
                    search_time=(datetime.now() - start_time).total_seconds(),
                    query=query.text,
                    error_message="Vector search timeout",
                )

            # 結果融合
            if self.config.search_mode == SearchMode.HYBRID:
                fused_results = self._fuse_search_results(
                    search_results.get("dense"), search_results.get("sparse"), query
                )
            elif self.config.search_mode == SearchMode.SEMANTIC:
                fused_results = self._process_single_search_result(
                    search_results["dense"], query
                )
            else:  # KEYWORD
                fused_results = self._process_single_search_result(
                    search_results["sparse"], query
                )

            # ドキュメント詳細を取得
            documents = await self._retrieve_documents(fused_results, query)

            # フィルタリング適用
            filtered_documents = self._apply_query_filters(documents, query.filters)

            # 類似度閾値でフィルタリング
            if self.config.similarity_threshold > 0:
                filtered_documents = self._filter_by_similarity_threshold(
                    filtered_documents
                )

            # ページネーション
            paginated_documents = filtered_documents[
                query.offset : query.offset + query.max_results
            ]

            # ファセット計算
            facets = None
            if query.facets:
                facets = self._calculate_facets(filtered_documents, query.facets)

            end_time = datetime.now()
            search_time = (end_time - start_time).total_seconds()

            return SearchResult(
                success=True,
                documents=paginated_documents,
                total_hits=len(filtered_documents),
                search_time=search_time,
                query=query.text,
                facets=facets,
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            end_time = datetime.now()
            search_time = (end_time - start_time).total_seconds()

            return SearchResult(
                success=False,
                documents=[],
                total_hits=0,
                search_time=search_time,
                query=query.text,
                error_message=str(e),
            )

    async def _generate_query_embeddings(self, text: str) -> dict[str, Any]:
        """クエリの埋め込みを生成"""
        if not self.embedding_service:
            # モック実装
            return {
                "dense": np.random.random(1024).tolist(),
                "sparse": self._extract_keywords_as_sparse(text),
            }

        # 実際の埋め込み生成
        embedding_result = await self.embedding_service.embed_text(text)

        return {
            "dense": embedding_result.dense_vector,
            "sparse": embedding_result.sparse_vector,
        }

    def _extract_keywords_as_sparse(self, text: str) -> dict[str, float]:
        """テキストからキーワードを抽出してsparse vectorとして返す"""
        # 簡易的なキーワード抽出
        import re
        from collections import Counter

        # 単語を抽出
        words = re.findall(r"\b\w+\b", text.lower())

        # ストップワードを除去（簡易版）
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
        }

        filtered_words = [
            word for word in words if word not in stop_words and len(word) > 2
        ]

        # 単語頻度を計算してsparse vectorに変換
        word_counts = Counter(filtered_words)
        total_words = len(filtered_words)

        sparse_vector = {}
        for word, count in word_counts.items():
            # TF-IDFライクなスコア（簡易版）
            tf = count / total_words if total_words > 0 else 0
            sparse_vector[word] = min(tf * 10, 1.0)  # 0-1にスケール

        return sparse_vector

    async def _search_dense_vectors(
        self, query_vector: list[float], query: SearchQuery
    ) -> VectorSearchResult:
        """Dense vector検索"""
        start_time = datetime.now()

        # モック実装（実際はMilvusを使用）
        # ここでは類似ドキュメントIDとスコアを返す
        mock_ids = ["doc-1", "doc-2", "doc-3", "doc-4", "doc-5"]
        mock_scores = [0.95, 0.87, 0.73, 0.68, 0.54]

        # top_kでフィルタリング
        top_k = min(self.config.top_k * 2, len(mock_ids))  # 後でRRFするため多めに取得
        filtered_ids = mock_ids[:top_k]
        filtered_scores = mock_scores[:top_k]

        search_time = (datetime.now() - start_time).total_seconds()

        return VectorSearchResult(
            ids=filtered_ids,
            scores=filtered_scores,
            search_time=search_time,
            total_hits=len(filtered_ids),
        )

    async def _search_sparse_vectors(
        self, query_vector: dict[str, float], query: SearchQuery
    ) -> VectorSearchResult:
        """Sparse vector検索"""
        start_time = datetime.now()

        # モック実装（実際はMilvusのsparse indexを使用）
        mock_ids = ["doc-2", "doc-1", "doc-4", "doc-3", "doc-5"]
        mock_scores = [0.92, 0.84, 0.71, 0.65, 0.48]

        # top_kでフィルタリング
        top_k = min(self.config.top_k * 2, len(mock_ids))
        filtered_ids = mock_ids[:top_k]
        filtered_scores = mock_scores[:top_k]

        search_time = (datetime.now() - start_time).total_seconds()

        return VectorSearchResult(
            ids=filtered_ids,
            scores=filtered_scores,
            search_time=search_time,
            total_hits=len(filtered_ids),
        )

    def _fuse_search_results(
        self,
        dense_result: VectorSearchResult | None,
        sparse_result: VectorSearchResult | None,
        query: SearchQuery,
    ) -> list[tuple[str, float]]:
        """検索結果を融合"""

        if self.config.ranking_algorithm == RankingAlgorithm.RRF:
            if dense_result and sparse_result:
                return self.rrf.fuse_rankings(dense_result, sparse_result)
            elif dense_result:
                return list(zip(dense_result.ids, dense_result.scores, strict=False))
            elif sparse_result:
                return list(zip(sparse_result.ids, sparse_result.scores, strict=False))
            else:
                return []

        elif self.config.ranking_algorithm == RankingAlgorithm.WEIGHTED_SUM:
            return self._weighted_sum_fusion(dense_result, sparse_result)

        else:
            # デフォルトはRRF
            if dense_result and sparse_result:
                return self.rrf.fuse_rankings(dense_result, sparse_result)
            else:
                return []

    def _weighted_sum_fusion(
        self,
        dense_result: VectorSearchResult | None,
        sparse_result: VectorSearchResult | None,
    ) -> list[tuple[str, float]]:
        """重み付き和による融合"""

        combined_scores = defaultdict(float)

        if dense_result:
            for doc_id, score in zip(
                dense_result.ids, dense_result.scores, strict=False
            ):
                combined_scores[doc_id] += score * self.config.dense_weight

        if sparse_result:
            for doc_id, score in zip(
                sparse_result.ids, sparse_result.scores, strict=False
            ):
                combined_scores[doc_id] += score * self.config.sparse_weight

        # スコア降順でソート
        sorted_results = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_results

    def _process_single_search_result(
        self, search_result: VectorSearchResult, query: SearchQuery
    ) -> list[tuple[str, float]]:
        """単一検索結果の処理"""
        if search_result:
            return list(zip(search_result.ids, search_result.scores, strict=False))
        else:
            return []

    async def _retrieve_documents(
        self, search_results: list[tuple[str, float]], query: SearchQuery
    ) -> list[dict[str, Any]]:
        """検索結果からドキュメント詳細を取得"""

        # モック実装
        mock_documents = [
            {
                "id": "doc-1",
                "title": "Machine Learning Fundamentals",
                "content": "Introduction to machine learning algorithms and their applications.",
                "source_type": "confluence",
                "language": "en",
                "document_type": "wiki",
                "search_score": 0.0,  # 後で設定
                "metadata": {
                    "author": "john_doe",
                    "tags": ["ml", "algorithm", "tutorial"],
                    "created_at": "2024-01-01T00:00:00Z",
                },
            },
            {
                "id": "doc-2",
                "title": "Natural Language Processing",
                "content": "Advanced NLP techniques for text analysis.",
                "source_type": "confluence",
                "language": "en",
                "document_type": "guide",
                "search_score": 0.0,
                "metadata": {
                    "author": "jane_smith",
                    "tags": ["nlp", "text", "processing"],
                    "created_at": "2024-01-02T00:00:00Z",
                },
            },
            {
                "id": "doc-3",
                "title": "Deep Learning Networks",
                "content": "Neural networks and deep learning architectures.",
                "source_type": "jira",
                "language": "en",
                "document_type": "specification",
                "search_score": 0.0,
                "metadata": {
                    "author": "bob_wilson",
                    "tags": ["deep", "learning", "neural"],
                    "created_at": "2024-01-03T00:00:00Z",
                },
            },
        ]

        # 検索結果のスコアを設定
        result_docs = []
        score_map = dict(search_results)

        for doc in mock_documents:
            if doc["id"] in score_map:
                doc_copy = doc.copy()
                doc_copy["search_score"] = score_map[doc["id"]]
                result_docs.append(doc_copy)

        # スコア順でソート
        result_docs.sort(key=lambda x: x["search_score"], reverse=True)

        return result_docs

    def _apply_query_filters(
        self, documents: list[dict[str, Any]], filters: list[SearchFilter]
    ) -> list[dict[str, Any]]:
        """クエリフィルターを適用"""
        if not filters:
            return documents

        filtered_docs = []
        for doc in documents:
            if self._apply_filters(doc, filters):
                filtered_docs.append(doc)

        return filtered_docs

    def _apply_filters(
        self, document: dict[str, Any], filters: list[SearchFilter]
    ) -> bool:
        """単一ドキュメントにフィルターを適用"""
        for filter_item in filters:
            field_value = self._get_nested_field(document, filter_item.field)

            if filter_item.operator == "eq":
                if field_value != filter_item.value:
                    return False
            elif filter_item.operator == "ne":
                if field_value == filter_item.value:
                    return False
            elif filter_item.operator == "in":
                if field_value not in filter_item.value:
                    return False
            elif filter_item.operator == "not_in":
                if field_value in filter_item.value:
                    return False
            # 他の演算子も必要に応じて実装

        return True

    def _get_nested_field(self, document: dict[str, Any], field_path: str) -> Any:
        """ネストされたフィールドの値を取得"""
        parts = field_path.split(".")
        value = document

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None

        return value

    def _filter_by_similarity_threshold(
        self, documents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """類似度閾値でフィルタリング"""
        return [
            doc
            for doc in documents
            if doc.get("search_score", 0) >= self.config.similarity_threshold
        ]

    def _calculate_facets(
        self, documents: list[dict[str, Any]], facet_fields: list[str]
    ) -> dict[str, list[FacetResult]]:
        """ファセットを計算"""
        facets = {}

        for facet_field in facet_fields:
            field_values = []

            for doc in documents:
                value = self._get_nested_field(doc, facet_field)
                if value is not None:
                    if isinstance(value, list):
                        field_values.extend(value)
                    else:
                        field_values.append(str(value))

            # 値をカウント
            value_counts = Counter(field_values)

            # FacetResultオブジェクトを作成
            facet_results = [
                FacetResult(value=value, count=count)
                for value, count in value_counts.most_common()
            ]

            facets[facet_field] = facet_results

        return facets

    def _calculate_combined_score(
        self, dense_score: float, sparse_score: float
    ) -> float:
        """Dense/Sparseスコアを組み合わせ"""
        return (
            dense_score * self.config.dense_weight
            + sparse_score * self.config.sparse_weight
        )

    def _calculate_content_relevance(
        self, documents: list[dict[str, Any]], query_text: str
    ) -> dict[str, float]:
        """コンテンツ関連性スコアを計算"""
        import re

        query_terms = set(re.findall(r"\b\w+\b", query_text.lower()))
        relevance_scores = {}

        for doc in documents:
            doc_id = doc["id"]
            title = doc.get("title", "").lower()
            content = doc.get("content", "").lower()

            # タイトルでの一致度（重み高）
            title_terms = set(re.findall(r"\b\w+\b", title))
            title_overlap = len(query_terms.intersection(title_terms)) / max(
                len(query_terms), 1
            )

            # コンテンツでの一致度
            content_terms = set(re.findall(r"\b\w+\b", content))
            content_overlap = len(query_terms.intersection(content_terms)) / max(
                len(query_terms), 1
            )

            # 統合スコア（タイトル重視）
            relevance_score = title_overlap * 0.7 + content_overlap * 0.3
            relevance_scores[doc_id] = min(relevance_score, 1.0)

        return relevance_scores

    def _calculate_freshness_scores(
        self, documents: list[dict[str, Any]]
    ) -> dict[str, float]:
        """新しさスコアを計算"""
        from datetime import datetime

        freshness_scores = {}

        # ドキュメント内で最新の日時を基準時刻とする（相対的新しさ）
        doc_times = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            updated_at = metadata.get("updated_at") or metadata.get("created_at")
            if updated_at:
                try:
                    doc_time = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                    doc_times.append(doc_time)
                except (ValueError, TypeError):
                    pass

        if not doc_times:
            # 日時がない場合はすべて同じスコア
            for doc in documents:
                freshness_scores[doc["id"]] = 0.5
            return freshness_scores

        # 最新の日時を基準とする
        latest_time = max(doc_times)

        for doc in documents:
            doc_id = doc["id"]
            metadata = doc.get("metadata", {})

            # 更新日時を取得（なければ作成日時）
            updated_at = metadata.get("updated_at") or metadata.get("created_at")

            if updated_at:
                try:
                    doc_time = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                    days_diff = (latest_time - doc_time).days

                    # 指数的減衰（30日で半減） - より短いスパンで計算
                    freshness_score = np.exp(-days_diff / 30.0)
                    freshness_scores[doc_id] = min(freshness_score, 1.0)
                except (ValueError, TypeError):
                    freshness_scores[doc_id] = 0.3  # デフォルト値（低め）
            else:
                freshness_scores[doc_id] = 0.3

        return freshness_scores

    def _calculate_popularity_scores(
        self, documents: list[dict[str, Any]]
    ) -> dict[str, float]:
        """人気度スコアを計算"""
        popularity_scores = {}

        # 最大値を取得してスケーリング
        view_counts = [
            doc.get("metadata", {}).get("view_count", 0) for doc in documents
        ]
        ratings = [doc.get("metadata", {}).get("rating", 0) for doc in documents]

        max_views = max(view_counts) if view_counts else 1
        max_rating = max(ratings) if ratings else 1

        for doc in documents:
            doc_id = doc["id"]
            metadata = doc.get("metadata", {})

            view_count = metadata.get("view_count", 0)
            rating = metadata.get("rating", 0)

            # 正規化して統合
            view_score = view_count / max_views if max_views > 0 else 0
            rating_score = rating / max_rating if max_rating > 0 else 0

            popularity_score = view_score * 0.6 + rating_score * 0.4
            popularity_scores[doc_id] = min(popularity_score, 1.0)

        return popularity_scores

    def _calculate_authority_scores(
        self, documents: list[dict[str, Any]]
    ) -> dict[str, float]:
        """権威性スコアを計算"""
        authority_scores = {}

        # 著者の権威性マッピング（実際の実装では外部データを使用）
        author_authority = {
            "expert_author": 1.0,
            "researcher": 0.8,
            "data_engineer": 0.6,
            "web_developer": 0.5,
        }

        for doc in documents:
            doc_id = doc["id"]
            metadata = doc.get("metadata", {})
            author = metadata.get("author", "unknown")

            authority_score = author_authority.get(author, 0.3)  # デフォルト値
            authority_scores[doc_id] = authority_score

        return authority_scores

    def _calculate_quality_scores(
        self, documents: list[dict[str, Any]]
    ) -> dict[str, float]:
        """品質スコアを計算"""
        quality_scores = {}

        # メトリクスの最大値を取得
        word_counts = [
            doc.get("metadata", {}).get("word_count", 0) for doc in documents
        ]
        likes = [doc.get("metadata", {}).get("likes", 0) for doc in documents]

        max_words = max(word_counts) if word_counts else 1
        max_likes = max(likes) if likes else 1

        for doc in documents:
            doc_id = doc["id"]
            metadata = doc.get("metadata", {})

            rating = metadata.get("rating", 0)
            word_count = metadata.get("word_count", 0)
            likes_count = metadata.get("likes", 0)

            # 品質要素を統合
            rating_score = rating / 5.0 if rating > 0 else 0
            length_score = min(word_count / max_words, 1.0) if max_words > 0 else 0
            engagement_score = likes_count / max_likes if max_likes > 0 else 0

            quality_score = (
                rating_score * 0.5 + length_score * 0.3 + engagement_score * 0.2
            )
            quality_scores[doc_id] = min(quality_score, 1.0)

        return quality_scores

    def _calculate_combined_relevance_scores(
        self, documents: list[dict[str, Any]], query_text: str
    ) -> dict[str, float]:
        """統合関連性スコアを計算"""
        content_scores = self._calculate_content_relevance(documents, query_text)
        freshness_scores = self._calculate_freshness_scores(documents)
        popularity_scores = self._calculate_popularity_scores(documents)
        authority_scores = self._calculate_authority_scores(documents)
        quality_scores = self._calculate_quality_scores(documents)

        combined_scores = {}

        for doc in documents:
            doc_id = doc["id"]

            # 重み付き統合
            combined_score = (
                content_scores.get(doc_id, 0) * 0.4
                + freshness_scores.get(doc_id, 0) * 0.15
                + popularity_scores.get(doc_id, 0) * 0.2
                + authority_scores.get(doc_id, 0) * 0.15
                + quality_scores.get(doc_id, 0) * 0.1
            )

            combined_scores[doc_id] = min(combined_score, 1.0)

        return combined_scores

    def _rerank_with_features(
        self, documents: list[dict[str, Any]], query_text: str
    ) -> list[dict[str, Any]]:
        """特徴量を使ったリランキング"""
        if not self.config.enable_reranking:
            return documents

        # 統合関連性スコアを計算
        relevance_scores = self._calculate_combined_relevance_scores(
            documents, query_text
        )

        # 各ドキュメントにリランキングスコアを追加
        reranked_docs = []
        for doc in documents:
            doc_copy = doc.copy()
            doc_id = doc["id"]

            # 元の検索スコアと関連性スコアを統合
            original_score = doc.get("search_score", 0)
            relevance_score = relevance_scores.get(doc_id, 0)

            # リランキングスコア計算
            rerank_score = original_score * 0.6 + relevance_score * 0.4
            doc_copy["rerank_score"] = min(rerank_score, 1.0)

            reranked_docs.append(doc_copy)

        # リランキングスコアでソート
        reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked_docs

    def _calculate_personalization_scores(
        self, documents: list[dict[str, Any]], user_profile: dict[str, Any]
    ) -> dict[str, float]:
        """パーソナライゼーションスコアを計算"""
        personalization_scores = {}

        preferred_categories = set(user_profile.get("preferred_categories", []))
        preferred_authors = set(user_profile.get("preferred_authors", []))
        interaction_history = user_profile.get("interaction_history", {})

        for doc in documents:
            doc_id = doc["id"]
            metadata = doc.get("metadata", {})

            score = 0.0

            # カテゴリ一致
            doc_category = metadata.get("category", "")
            if doc_category in preferred_categories:
                score += 0.3

            # 著者一致
            doc_author = metadata.get("author", "")
            if doc_author in preferred_authors:
                score += 0.3

            # インタラクション履歴
            if doc_id in interaction_history:
                history = interaction_history[doc_id]
                views = history.get("views", 0)
                rating = history.get("rating", 0)

                # 過去のエンゲージメントを考慮
                score += min(views / 10.0, 0.2)  # 最大0.2
                score += rating / 5.0 * 0.2  # 最大0.2

            personalization_scores[doc_id] = min(score, 1.0)

        return personalization_scores

    def _apply_diversity_ranking(
        self, documents: list[dict[str, Any]], diversity_factor: float = 0.3
    ) -> list[dict[str, Any]]:
        """多様性を考慮したランキング"""
        if diversity_factor <= 0:
            return documents

        diversified_docs = []
        remaining_docs = documents.copy()
        selected_categories = set()

        while remaining_docs:
            best_doc = None
            best_score = -1

            for doc in remaining_docs:
                category = doc.get("metadata", {}).get("category", "unknown")
                base_score = doc.get("search_score", 0)

                # 多様性ボーナス
                diversity_bonus = 0
                if category not in selected_categories:
                    diversity_bonus = diversity_factor

                total_score = base_score + diversity_bonus

                if total_score > best_score:
                    best_score = total_score
                    best_doc = doc

            if best_doc:
                diversified_docs.append(best_doc)
                remaining_docs.remove(best_doc)
                category = best_doc.get("metadata", {}).get("category", "unknown")
                selected_categories.add(category)

        return diversified_docs

    def _calculate_temporal_boost_scores(
        self,
        documents: list[dict[str, Any]],
        current_time: datetime,
        boost_recent: bool = True,
    ) -> dict[str, float]:
        """時間的ブーストスコアを計算"""
        boost_scores = {}

        for doc in documents:
            doc_id = doc["id"]
            metadata = doc.get("metadata", {})

            updated_at = metadata.get("updated_at") or metadata.get("created_at")

            if updated_at:
                try:
                    doc_time = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                    days_diff = (current_time - doc_time).days

                    if boost_recent:
                        # 最近のドキュメントをブースト
                        if days_diff <= 7:
                            boost = 1.3
                        elif days_diff <= 30:
                            boost = 1.1
                        else:
                            boost = 1.0
                    else:
                        # 古いドキュメントをブースト
                        if days_diff >= 365:
                            boost = 1.2
                        else:
                            boost = 1.0

                    boost_scores[doc_id] = boost
                except (ValueError, TypeError, KeyError):
                    boost_scores[doc_id] = 1.0
            else:
                boost_scores[doc_id] = 1.0

        return boost_scores

    def _optimize_ranking_pipeline(
        self,
        documents: list[dict[str, Any]],
        query_text: str,
        user_profile: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """ランキング最適化パイプライン"""
        if not documents:
            return documents

        # 1. 基本的な関連性スコア計算
        relevance_scores = self._calculate_combined_relevance_scores(
            documents, query_text
        )

        # 2. パーソナライゼーション
        personalization_scores = {}
        if user_profile:
            personalization_scores = self._calculate_personalization_scores(
                documents, user_profile
            )

        # 3. 時間的ブースト
        current_time = datetime.now()
        temporal_scores = self._calculate_temporal_boost_scores(documents, current_time)

        # 4. 最終スコア計算
        final_docs = []
        for doc in documents:
            doc_copy = doc.copy()
            doc_id = doc["id"]

            base_score = doc.get("search_score", 0)
            relevance_score = relevance_scores.get(doc_id, 0)
            personal_score = personalization_scores.get(doc_id, 0)
            temporal_boost = temporal_scores.get(doc_id, 1.0)

            # 最終スコア統合
            final_score = (
                base_score * 0.4 + relevance_score * 0.4 + personal_score * 0.2
            ) * temporal_boost

            doc_copy["final_ranking_score"] = min(final_score, 1.0)
            final_docs.append(doc_copy)

        # 5. 最終スコアでソート
        final_docs.sort(key=lambda x: x["final_ranking_score"], reverse=True)

        # 6. 多様性適用
        if len(final_docs) > 3:
            final_docs = self._apply_diversity_ranking(final_docs, diversity_factor=0.2)

        return final_docs

    def _calculate_ranking_metrics(
        self, ranked_documents: list[dict[str, Any]], relevance_labels: dict[str, int]
    ) -> dict[str, float]:
        """ランキング性能メトリクスを計算"""
        metrics = {}

        # Precision@K
        for k in [1, 3, 5]:
            if len(ranked_documents) >= k:
                top_k_docs = ranked_documents[:k]
                relevant_count = sum(
                    1 for doc in top_k_docs if relevance_labels.get(doc["id"], 0) == 1
                )
                metrics[f"precision_at_{k}"] = relevant_count / k
            else:
                metrics[f"precision_at_{k}"] = 0.0

        # Mean Average Precision (MAP)
        ap_sum = 0
        relevant_found = 0
        for i, doc in enumerate(ranked_documents):
            if relevance_labels.get(doc["id"], 0) == 1:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                ap_sum += precision_at_i

        total_relevant = sum(1 for label in relevance_labels.values() if label == 1)
        metrics["map"] = ap_sum / total_relevant if total_relevant > 0 else 0.0

        # NDCG@5
        dcg = 0
        for i, doc in enumerate(ranked_documents[:5]):
            relevance = relevance_labels.get(doc["id"], 0)
            if i == 0:
                dcg += relevance
            else:
                dcg += relevance / np.log2(i + 1)

        # Ideal DCG
        ideal_relevances = sorted(relevance_labels.values(), reverse=True)[:5]
        idcg = 0
        for i, rel in enumerate(ideal_relevances):
            if i == 0:
                idcg += rel
            else:
                idcg += rel / np.log2(i + 1)

        metrics["ndcg"] = dcg / idcg if idcg > 0 else 0.0

        return metrics

    def _rank_with_explanation(
        self, documents: list[dict[str, Any]], query_text: str
    ) -> list[dict[str, Any]]:
        """説明付きランキング"""
        content_scores = self._calculate_content_relevance(documents, query_text)
        freshness_scores = self._calculate_freshness_scores(documents)
        popularity_scores = self._calculate_popularity_scores(documents)
        quality_scores = self._calculate_quality_scores(documents)

        explained_docs = []
        for doc in documents:
            doc_copy = doc.copy()
            doc_id = doc["id"]

            explanation = {
                "content_relevance": content_scores.get(doc_id, 0),
                "freshness_score": freshness_scores.get(doc_id, 0),
                "popularity_score": popularity_scores.get(doc_id, 0),
                "quality_score": quality_scores.get(doc_id, 0),
                "final_score_breakdown": {
                    "content_weight": 0.4,
                    "freshness_weight": 0.15,
                    "popularity_weight": 0.2,
                    "quality_weight": 0.25,
                },
            }

            doc_copy["ranking_explanation"] = explanation
            explained_docs.append(doc_copy)

        return explained_docs
