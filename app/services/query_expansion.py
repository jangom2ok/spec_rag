"""クエリ拡張サービス

TDD実装：クエリ拡張・同義語・関連語・概念拡張機能
- 同義語拡張: WordNet/類語辞書ベース
- セマンティック拡張: 埋め込みベースの類似語検索
- 概念拡張: 概念階層・ドメイン知識ベース
- 統計的拡張: 共起語・頻度統計ベース
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ExpansionMethod(str, Enum):
    """拡張手法"""

    SYNONYM = "synonym"
    SEMANTIC = "semantic"
    CONCEPTUAL = "conceptual"
    STATISTICAL = "statistical"


@dataclass
class QueryExpansionConfig:
    """クエリ拡張設定"""

    expansion_methods: list[ExpansionMethod] = field(
        default_factory=lambda: [ExpansionMethod.SYNONYM]
    )
    max_expansions: int = 5
    similarity_threshold: float = 0.7
    expansion_weight: float = 0.3
    preserve_original_weight: float = 0.7

    # セマンティック拡張設定
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # 概念拡張設定
    enable_concept_expansion: bool = True
    concept_depth: int = 2

    # 統計的拡張設定
    enable_statistical_expansion: bool = False
    statistical_corpus_size: int = 100000
    min_term_frequency: int = 10

    # キャッシュ設定
    enable_caching: bool = True
    cache_ttl: int = 3600

    # タイムアウト設定
    timeout: float = 30.0

    def __post_init__(self):
        """設定値のバリデーション"""
        if self.max_expansions <= 0:
            raise ValueError("max_expansions must be greater than 0")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if not 0 <= self.expansion_weight <= 1:
            raise ValueError("expansion_weight must be between 0 and 1")


@dataclass
class ExpansionRequest:
    """拡張リクエスト"""

    query: str
    language: str = "en"
    domain: str | None = None
    max_expansions: int | None = None
    preserve_original: bool = True
    expansion_types: list[str] | None = None
    user_context: dict[str, Any] | None = None


@dataclass
class QueryTerm:
    """クエリ用語"""

    term: str
    pos_tag: str
    importance: float
    synonyms: list[str] = field(default_factory=list)
    semantic_neighbors: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    statistical_related: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class ExpandedQuery:
    """拡張クエリ"""

    original_query: str
    expanded_query: str
    expansion_terms: list[str]
    term_weights: dict[str, float]
    total_weight: float
    expansion_methods_used: list[str]


@dataclass
class ExpansionResult:
    """拡張結果"""

    success: bool
    original_query: str
    expanded_terms: list[str]
    expanded_query: str
    expansion_time: float
    language: str
    error_message: str | None = None
    cache_hit: bool = False
    expanded_query_obj: ExpandedQuery | None = None

    def get_summary(self) -> dict[str, Any]:
        """拡張結果のサマリーを取得"""
        return {
            "success": self.success,
            "original_query": self.original_query,
            "expansion_count": len(self.expanded_terms),
            "expansion_time": self.expansion_time,
            "cache_hit": self.cache_hit,
            "language": self.language,
        }


class BaseExpander:
    """拡張器ベースクラス"""

    def __init__(self, config: QueryExpansionConfig):
        self.config = config

    async def expand(
        self, terms: list[str], language: str = "en", **kwargs
    ) -> dict[str, list[str]]:
        """拡張実行（オーバーライド必須）"""
        raise NotImplementedError


class SynonymExpander(BaseExpander):
    """同義語拡張器"""

    def __init__(self, config: QueryExpansionConfig):
        super().__init__(config)
        self.max_expansions = config.max_expansions
        self.wordnet_cache = {}

    async def expand_term(self, term: str, language: str = "en") -> list[str]:
        """単一用語の同義語拡張"""
        synonyms = []

        if language == "en":
            synonyms.extend(self._get_wordnet_synonyms(term))
        elif language == "ja":
            synonyms.extend(self._get_japanese_synonyms(term))

        return synonyms[: self.max_expansions]

    def _get_wordnet_synonyms(self, term: str) -> list[str]:
        """WordNet同義語取得（モック実装）"""
        # 実際の実装では nltk.corpus.wordnet を使用
        synonym_dict = {
            "machine": ["device", "apparatus", "system", "equipment"],
            "learning": ["education", "training", "study", "acquisition"],
            "algorithms": ["methods", "procedures", "techniques", "approaches"],
            "computer": ["device", "machine", "system", "processor"],
            "data": ["information", "facts", "statistics", "records"],
        }
        return synonym_dict.get(term.lower(), [])

    def _get_japanese_synonyms(self, term: str) -> list[str]:
        """日本語同義語取得（モック実装）"""
        synonym_dict = {
            "機械学習": ["マシンラーニング", "ML", "自動学習"],
            "アルゴリズム": ["手法", "方法", "プロシージャ", "演算法"],
            "コンピュータ": ["計算機", "PC", "マシン"],
            "データ": ["データー", "情報", "資料"],
        }
        return synonym_dict.get(term, [])


class SemanticExpander(BaseExpander):
    """セマンティック拡張器"""

    def __init__(self, config: QueryExpansionConfig):
        super().__init__(config)
        self.embedding_model = config.embedding_model
        self.similarity_threshold = config.similarity_threshold
        self.model = None

    async def load_model(self) -> None:
        """埋め込みモデル読み込み"""
        try:
            logger.info(f"Loading semantic model: {self.embedding_model}")
            # 実際の実装では sentence-transformers を使用
            self.model = MockSemanticModel(self.embedding_model)
            logger.info("Semantic model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load semantic model: {e}")
            raise

    async def expand_phrase(
        self, phrase: str, max_expansions: int = 5
    ) -> list[tuple[str, float]]:
        """フレーズのセマンティック拡張"""
        if not self.model:
            await self.load_model()

        # セマンティック類似語を取得
        similar_phrases = self._find_similar_phrases(phrase, max_expansions)

        # 閾値でフィルタリング
        filtered_phrases = [
            (phrase, score)
            for phrase, score in similar_phrases
            if score >= self.similarity_threshold
        ]

        return filtered_phrases

    def _find_similar_phrases(
        self, phrase: str, max_expansions: int
    ) -> list[tuple[str, float]]:
        """類似フレーズ検索（モック実装）"""
        # 実際の実装では埋め込みベクトルの類似度計算
        similarity_dict = {
            "machine learning": [
                ("artificial intelligence", 0.85),
                ("neural networks", 0.78),
                ("deep learning", 0.76),
                ("pattern recognition", 0.72),
                ("data mining", 0.69),
            ],
            "algorithms": [
                ("techniques", 0.82),
                ("methods", 0.79),
                ("approaches", 0.75),
                ("procedures", 0.71),
                ("strategies", 0.68),
            ],
            "neural network": [
                ("deep learning", 0.88),
                ("artificial intelligence", 0.83),
                ("machine learning", 0.80),
                ("neural computation", 0.77),
            ],
        }

        return similarity_dict.get(phrase.lower(), [])[:max_expansions]

    def _calculate_semantic_similarity(self, phrase1: str, phrase2: str) -> float:
        """セマンティック類似度計算"""
        # モック実装
        return 0.85


class ConceptualExpander(BaseExpander):
    """概念拡張器"""

    def __init__(self, config: QueryExpansionConfig):
        super().__init__(config)
        self.enable_concept_expansion = config.enable_concept_expansion
        self.concept_depth = config.concept_depth
        self.concept_hierarchy = {}

    async def expand_concept(
        self, concept: str, domain: str = None
    ) -> dict[str, list[str]]:
        """概念拡張"""
        if not self.enable_concept_expansion:
            return {}

        hierarchy = self._get_concept_hierarchy(concept, domain)

        expanded_concepts = {
            "broader_concepts": hierarchy.get("broader", []),
            "narrower_concepts": hierarchy.get("narrower", []),
            "related_concepts": hierarchy.get("related", []),
        }

        return expanded_concepts

    def _get_concept_hierarchy(
        self, concept: str, domain: str = None
    ) -> dict[str, list[str]]:
        """概念階層取得（モック実装）"""
        # 実際の実装では ConceptNet, WordNet, ドメイン知識ベースを使用
        hierarchy_dict = {
            "machine_learning": {
                "broader": ["artificial_intelligence", "computer_science"],
                "narrower": [
                    "supervised_learning",
                    "unsupervised_learning",
                    "reinforcement_learning",
                ],
                "related": [
                    "data_mining",
                    "pattern_recognition",
                    "statistical_learning",
                ],
            },
            "algorithms": {
                "broader": ["computer_science", "mathematics"],
                "narrower": [
                    "sorting_algorithms",
                    "search_algorithms",
                    "optimization_algorithms",
                ],
                "related": [
                    "data_structures",
                    "complexity_theory",
                    "computational_methods",
                ],
            },
            "neural_network": {
                "broader": ["artificial_intelligence", "machine_learning"],
                "narrower": [
                    "convolutional_neural_network",
                    "recurrent_neural_network",
                ],
                "related": ["deep_learning", "neural_computation", "connectionism"],
            },
        }

        # ドメイン固有の概念階層
        if domain == "medical":
            medical_concepts = {
                "neural_network": {
                    "broader": ["nervous_system", "neuroscience"],
                    "narrower": ["brain_network", "neural_pathway"],
                    "related": ["brain_function", "neurobiology", "cognitive_science"],
                }
            }
            hierarchy_dict.update(medical_concepts)

        return hierarchy_dict.get(
            concept, {"broader": [], "narrower": [], "related": []}
        )


class StatisticalExpander(BaseExpander):
    """統計的拡張器"""

    def __init__(self, config: QueryExpansionConfig):
        super().__init__(config)
        self.enable_statistical_expansion = config.enable_statistical_expansion
        self.min_term_frequency = config.min_term_frequency
        self.corpus_size = config.statistical_corpus_size

    async def get_co_occurrence_terms(
        self, term: str, min_frequency: int = None
    ) -> list[tuple[str, float, int]]:
        """共起語取得"""
        # enable_statistical_expansionを初期化で無効でも、明示的に呼び出された場合は実行
        min_freq = min_frequency or self.min_term_frequency
        cooccur_terms = self._get_co_occurrence_terms(term)

        # 最小頻度でフィルタリング
        filtered_terms = [
            (term, pmi, freq) for term, pmi, freq in cooccur_terms if freq >= min_freq
        ]

        return filtered_terms

    def _get_co_occurrence_terms(self, term: str) -> list[tuple[str, float, int]]:
        """共起語取得（モック実装）"""
        # 実際の実装では大規模コーパスから PMI (Pointwise Mutual Information) を計算
        cooccur_dict = {
            "machine": [
                ("learning", 0.75, 1250),
                ("intelligence", 0.68, 980),
                ("vision", 0.62, 750),
                ("translation", 0.58, 650),
            ],
            "learning": [
                ("machine", 0.75, 1250),
                ("deep", 0.72, 1120),
                ("supervised", 0.68, 890),
                ("neural", 0.65, 980),
            ],
            "algorithms": [
                ("machine", 0.70, 800),
                ("optimization", 0.68, 720),
                ("sorting", 0.65, 650),
                ("search", 0.62, 580),
            ],
        }

        return cooccur_dict.get(term.lower(), [])


class QueryExpansionService:
    """クエリ拡張メインサービス"""

    def __init__(self, config: QueryExpansionConfig):
        self.config = config
        self.expanders = self._create_expanders()
        self.cache = {}  # 簡易キャッシュ実装

    def _create_expanders(self) -> dict[ExpansionMethod, BaseExpander]:
        """拡張器インスタンス作成"""
        expanders = {}

        for method in self.config.expansion_methods:
            if method == ExpansionMethod.SYNONYM:
                expanders[method] = SynonymExpander(self.config)
            elif method == ExpansionMethod.SEMANTIC:
                expanders[method] = SemanticExpander(self.config)
            elif method == ExpansionMethod.CONCEPTUAL:
                expanders[method] = ConceptualExpander(self.config)
            elif method == ExpansionMethod.STATISTICAL:
                expanders[method] = StatisticalExpander(self.config)

        return expanders

    async def expand_query(self, request: ExpansionRequest) -> ExpansionResult:
        """クエリ拡張実行"""
        start_time = datetime.now()

        try:
            # 入力バリデーション
            if not request.query.strip():
                return ExpansionResult(
                    success=False,
                    original_query=request.query,
                    expanded_terms=[],
                    expanded_query="",
                    expansion_time=0.0,
                    language=request.language,
                    error_message="Query cannot be empty",
                )

            # キャッシュチェック
            if self.config.enable_caching:
                cache_key = self._get_cache_key(request)
                cached_result = await self._get_from_cache(cache_key)
                if cached_result:
                    return cached_result

            # クエリ用語抽出
            query_terms = await self._extract_query_terms(
                request.query, request.language
            )

            # 各拡張手法を適用
            all_expansions = {}

            if ExpansionMethod.SYNONYM in self.config.expansion_methods:
                synonym_expansions = await self._expand_with_synonyms(
                    [term.term for term in query_terms], request.language
                )
                all_expansions.update(synonym_expansions)

            if ExpansionMethod.SEMANTIC in self.config.expansion_methods:
                semantic_expansions = await self._expand_with_semantics(
                    request.query, self.config.similarity_threshold
                )
                all_expansions.update(semantic_expansions)

            if ExpansionMethod.CONCEPTUAL in self.config.expansion_methods:
                conceptual_expansions = await self._expand_with_concepts(
                    [term.term.replace(" ", "_") for term in query_terms],
                    request.domain,
                )
                all_expansions.update(conceptual_expansions)

            if ExpansionMethod.STATISTICAL in self.config.expansion_methods:
                statistical_expansions = await self._expand_with_statistics(
                    request.query, self.config.min_term_frequency
                )
                all_expansions.update(statistical_expansions)

            # 拡張用語のフィルタリングとランキング
            filtered_expansions = self._filter_and_rank_expansions(
                all_expansions,
                request.max_expansions or self.config.max_expansions,
                self.config.similarity_threshold,
            )

            # 拡張クエリ生成
            expanded_query = self._generate_expanded_query(
                request.query, filtered_expansions, self.config.preserve_original_weight
            )

            end_time = datetime.now()
            expansion_time = (end_time - start_time).total_seconds()

            result = ExpansionResult(
                success=True,
                original_query=request.query,
                expanded_terms=expanded_query.expansion_terms,
                expanded_query=expanded_query.expanded_query,
                expansion_time=expansion_time,
                language=request.language,
                expanded_query_obj=expanded_query,
            )

            # キャッシュに保存
            if self.config.enable_caching:
                await self._set_cache(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            end_time = datetime.now()
            expansion_time = (end_time - start_time).total_seconds()

            return ExpansionResult(
                success=False,
                original_query=request.query,
                expanded_terms=[],
                expanded_query="",
                expansion_time=expansion_time,
                language=request.language,
                error_message=str(e),
            )

    async def _extract_query_terms(
        self, query: str, language: str = "en"
    ) -> list[QueryTerm]:
        """クエリ用語抽出"""
        # 簡易実装（実際は形態素解析・POS tagging使用）
        words = re.findall(r"\b\w+\b", query.lower())
        terms = []

        for i, word in enumerate(words):
            # 重要度は単純に位置と長さで計算（実際はTF-IDF等使用）
            importance = 1.0 - (i * 0.1) + (len(word) * 0.05)
            importance = min(max(importance, 0.1), 1.0)

            term = QueryTerm(
                term=word,
                pos_tag="NOUN",  # 簡易実装
                importance=importance,
            )
            terms.append(term)

        return terms

    async def _expand_with_synonyms(
        self, terms: list[str], language: str
    ) -> dict[str, list[str]]:
        """同義語拡張"""
        synonym_expander = self.expanders.get(ExpansionMethod.SYNONYM)
        if not synonym_expander:
            return {}

        synonyms = {}
        for term in terms:
            term_synonyms = await synonym_expander.expand_term(term, language)
            if term_synonyms:
                synonyms[term] = term_synonyms

        return synonyms

    async def _expand_with_semantics(
        self, query: str, threshold: float
    ) -> dict[str, list[tuple[str, float]]]:
        """セマンティック拡張"""
        semantic_expander = self.expanders.get(ExpansionMethod.SEMANTIC)
        if not semantic_expander:
            return {}

        semantics = {}

        # フレーズレベルでの拡張
        phrases = [query]  # 単純化、実際はngramやエンティティ抽出

        for phrase in phrases:
            similar_phrases = await semantic_expander.expand_phrase(
                phrase, self.config.max_expansions
            )
            if similar_phrases:
                semantics[phrase] = similar_phrases

        # 単語レベルでの拡張
        words = query.split()
        for word in words:
            if len(word) > 2:  # 短い単語はスキップ
                similar_phrases = await semantic_expander.expand_phrase(
                    word, self.config.max_expansions
                )
                if similar_phrases:
                    semantics[word] = similar_phrases

        return semantics

    async def _expand_with_concepts(
        self, terms: list[str], domain: str = None
    ) -> dict[str, dict[str, list[str]]]:
        """概念拡張"""
        conceptual_expander = self.expanders.get(ExpansionMethod.CONCEPTUAL)
        if not conceptual_expander:
            return {}

        concepts = {}
        for term in terms:
            concept_hierarchy = await conceptual_expander.expand_concept(term, domain)
            if concept_hierarchy:
                concepts[term] = concept_hierarchy

        return concepts

    async def _expand_with_statistics(
        self, query: str, min_frequency: int
    ) -> dict[str, list[tuple[str, float, int]]]:
        """統計的拡張"""
        statistical_expander = self.expanders.get(ExpansionMethod.STATISTICAL)
        if not statistical_expander:
            return {}

        statistics = {}

        # 単語レベルでの共起語取得
        words = query.split()
        for word in words:
            if len(word) > 2:
                cooccur_terms = await statistical_expander.get_co_occurrence_terms(
                    word, min_frequency
                )
                if cooccur_terms:
                    statistics[word] = cooccur_terms

        # フレーズレベルでの共起語取得
        if len(words) > 1:
            phrase = " ".join(words)
            cooccur_terms = await statistical_expander.get_co_occurrence_terms(
                phrase, min_frequency
            )
            if cooccur_terms:
                statistics[phrase] = cooccur_terms

        return statistics

    def _filter_and_rank_expansions(
        self, all_expansions: dict[str, Any], max_expansions: int, threshold: float
    ) -> dict[str, list[str | tuple[str, float]]]:
        """拡張用語のフィルタリングとランキング"""
        filtered = {}

        for term, expansions in all_expansions.items():
            if isinstance(expansions, list):
                if expansions and isinstance(expansions[0], tuple):
                    # (term, score) の形式
                    filtered_list = [
                        (exp_term, score)
                        for exp_term, score in expansions
                        if score >= threshold
                    ]
                    # スコア降順でソート
                    filtered_list.sort(key=lambda x: x[1], reverse=True)
                    filtered[term] = filtered_list[:max_expansions]
                else:
                    # 単純な文字列リスト
                    filtered[term] = expansions[:max_expansions]

        return filtered

    def _generate_expanded_query(
        self,
        original_query: str,
        expansion_terms: dict[str, Any],
        preserve_weight: float,
    ) -> ExpandedQuery:
        """拡張クエリ生成"""
        original_words = original_query.split()
        expanded_words = []
        all_terms = []

        # 元の用語を追加
        for word in original_words:
            expanded_words.append(word)
            all_terms.append(word)

        # 拡張用語を追加
        for _term, expansions in expansion_terms.items():
            if isinstance(expansions, list):
                for expansion in expansions:
                    if isinstance(expansion, tuple):
                        exp_word = expansion[0]
                    else:
                        exp_word = expansion

                    if exp_word not in all_terms:
                        expanded_words.append(exp_word)
                        all_terms.append(exp_word)

        # 重み計算
        term_weights = self._calculate_expansion_weights(
            original_words, expanded_words, self.config.expansion_weight
        )

        expanded_query_text = " ".join(expanded_words)
        total_weight = sum(term_weights.values())

        return ExpandedQuery(
            original_query=original_query,
            expanded_query=expanded_query_text,
            expansion_terms=all_terms,
            term_weights=term_weights,
            total_weight=total_weight,
            expansion_methods_used=[
                method.value for method in self.config.expansion_methods
            ],
        )

    def _calculate_expansion_weights(
        self, original_terms: list[str], all_terms: list[str], expansion_weight: float
    ) -> dict[str, float]:
        """拡張用語重み計算"""
        weights = {}
        total_original = len(original_terms)
        expansion_terms = [term for term in all_terms if term not in original_terms]
        total_expansion = len(expansion_terms)

        # 元の用語の重み
        preserve_weight = 1.0 - expansion_weight
        original_weight_per_term = preserve_weight / max(total_original, 1)
        for term in original_terms:
            weights[term] = original_weight_per_term

        # 拡張用語の重み
        expansion_weight_per_term = (
            expansion_weight / max(total_expansion, 1) if total_expansion > 0 else 0
        )
        for term in expansion_terms:
            weights[term] = expansion_weight_per_term

        return weights

    def _get_cache_key(self, request: ExpansionRequest) -> str:
        """キャッシュキー生成"""
        cache_content = {
            "query": request.query,
            "language": request.language,
            "domain": request.domain,
            "max_expansions": request.max_expansions or self.config.max_expansions,
            "expansion_types": request.expansion_types,
        }
        content_str = json.dumps(cache_content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

    async def _get_from_cache(self, cache_key: str) -> ExpansionResult | None:
        """キャッシュから取得"""
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.config.cache_ttl):
                # ExpansionResultオブジェクトを正しくコピー
                result = ExpansionResult(
                    success=cached_data.success,
                    original_query=cached_data.original_query,
                    expanded_terms=cached_data.expanded_terms.copy(),
                    expanded_query=cached_data.expanded_query,
                    expansion_time=cached_data.expansion_time,
                    language=cached_data.language,
                    error_message=cached_data.error_message,
                    cache_hit=True,
                    expanded_query_obj=cached_data.expanded_query_obj,
                )
                return result
            else:
                # 期限切れキャッシュを削除
                del self.cache[cache_key]
        return None

    async def _set_cache(self, cache_key: str, result: ExpansionResult) -> None:
        """キャッシュに保存"""
        self.cache[cache_key] = (result, datetime.now())


# モッククラス（テスト用）
class MockSemanticModel:
    """モックセマンティックモデル"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        """エンコーディング（モック実装）"""
        # テキスト長に基づいたランダムな埋め込み
        embeddings = []
        for _text in texts:
            # 固定次元の埋め込みベクトル
            embedding = np.random.random(384)  # all-MiniLM-L6-v2の次元数
            embeddings.append(embedding)
        return np.array(embeddings)

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """類似度計算（モック実装）"""
        # コサイン類似度の簡易計算
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
