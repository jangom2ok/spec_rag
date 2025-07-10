"""検索候補・サジェスト機能サービス

TDD実装：インテリジェント検索補完・候補生成機能
- 自動補完: プレフィックスマッチング・頻度ベース
- コンテキスト候補: 検索履歴・ドメイン知識ベース
- パーソナライズ候補: ユーザー行動・嗜好ベース
- トレンド候補: 流行検索クエリ・成長率ベース
- タイポ修正: 編集距離・スペルチェック
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SuggestionType(str, Enum):
    """候補タイプ"""

    AUTOCOMPLETE = "autocomplete"
    CONTEXTUAL = "contextual"
    PERSONALIZED = "personalized"
    TRENDING = "trending"
    TYPO_CORRECTION = "typo_correction"


@dataclass
class SuggestionsConfig:
    """候補設定"""

    suggestion_types: list[SuggestionType] = field(
        default_factory=lambda: [SuggestionType.AUTOCOMPLETE]
    )
    max_suggestions: int = 8
    min_query_length: int = 2
    similarity_threshold: float = 0.75

    # 自動補完設定
    enable_prefix_matching: bool = True
    prefix_min_frequency: int = 10

    # コンテキスト設定
    enable_context_analysis: bool = True
    context_weight: float = 0.3

    # パーソナライゼーション設定
    enable_personalization: bool = False
    personalization_weight: float = 0.4

    # トレンド設定
    enable_trending: bool = True
    trending_window_hours: int = 24
    trend_threshold: float = 1.5

    # タイポ修正設定
    enable_typo_correction: bool = True
    max_edit_distance: int = 2
    min_confidence: float = 0.6

    # キャッシュ設定
    enable_caching: bool = True
    cache_ttl: int = 1800  # 30分

    # タイムアウト設定
    timeout: float = 5.0

    def __post_init__(self):
        """設定値のバリデーション"""
        if self.max_suggestions <= 0:
            raise ValueError("max_suggestions must be greater than 0")
        if self.min_query_length < 0:
            raise ValueError("min_query_length must be non-negative")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")


@dataclass
class SuggestionRequest:
    """候補リクエスト"""

    query: str
    user_id: str | None = None
    context: dict[str, Any] | None = None
    max_suggestions: int | None = None
    include_corrections: bool = True
    include_completions: bool = True
    filter_types: list[str] | None = None


@dataclass
class SuggestionCandidate:
    """候補候補"""

    text: str
    type: SuggestionType
    score: float
    frequency: int = 0
    recency: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """候補の後処理"""
        if self.recency is None:
            self.recency = datetime.now()


@dataclass
class QueryCompletion:
    """クエリ補完"""

    original_query: str
    completed_query: str
    completion_part: str
    confidence: float = 0.0
    type: str = "suffix"


@dataclass
class SuggestionResult:
    """候補結果"""

    success: bool
    query: str
    suggestions: list[SuggestionCandidate]
    suggestion_time: float
    error_message: str | None = None
    cache_hit: bool = False
    completions: list[QueryCompletion] = field(default_factory=list)

    def get_summary(self) -> dict[str, Any]:
        """候補結果のサマリーを取得"""
        return {
            "success": self.success,
            "query": self.query,
            "suggestion_count": len(self.suggestions),
            "suggestion_time": self.suggestion_time,
            "cache_hit": self.cache_hit,
            "suggestion_types": list({s.type.value for s in self.suggestions}),
        }


class BaseSuggester:
    """候補器ベースクラス"""

    def __init__(self, config: SuggestionsConfig):
        self.config = config

    async def suggest(self, query: str, **kwargs) -> list[SuggestionCandidate]:
        """候補生成（オーバーライド必須）"""
        raise NotImplementedError


class AutocompleteSuggester(BaseSuggester):
    """自動補完候補器"""

    def __init__(self, config: SuggestionsConfig):
        super().__init__(config)
        self.max_suggestions = config.max_suggestions
        self.min_frequency = config.prefix_min_frequency
        self.prefix_index: dict[str, list[str]] = {}  # 簡易プレフィックスインデックス

    async def suggest(self, query: str, **kwargs: Any) -> list[SuggestionCandidate]:
        """自動補完候補生成"""
        max_results = kwargs.get("max_results") or self.max_suggestions

        # プレフィックスマッチング
        prefix_matches = self._get_prefix_matches(query, max_results)

        # 候補オブジェクトに変換
        suggestions = []
        for i, match in enumerate(prefix_matches):
            frequency = self._get_frequency(match)
            score = self._calculate_autocomplete_score(query, match, frequency, i)

            suggestion = SuggestionCandidate(
                text=match,
                type=SuggestionType.AUTOCOMPLETE,
                score=score,
                frequency=frequency,
                metadata={"match_type": "prefix", "position": i},
            )
            suggestions.append(suggestion)

        return suggestions

    def _get_prefix_matches(self, query: str, max_results: int) -> list[str]:
        """プレフィックスマッチング（モック実装）"""
        # 実際の実装では Trie構造やElasticsearchを使用
        query_lower = query.lower()
        mock_data = {
            "machine": [
                "machine learning",
                "machine translation",
                "machine vision",
                "machine intelligence",
            ],
            "machine learn": [
                "machine learning",
                "machine learning algorithms",
                "machine learning tutorial",
            ],
            "neural": [
                "neural networks",
                "neural network",
                "neural computation",
                "neural architecture",
            ],
            "deep": [
                "deep learning",
                "deep neural networks",
                "deep reinforcement learning",
            ],
            "ai": [
                "artificial intelligence",
                "ai algorithms",
                "ai applications",
                "ai ethics",
            ],
            "data": [
                "data science",
                "data analysis",
                "data mining",
                "data visualization",
            ],
            "機械学": ["機械学習", "機械学習アルゴリズム", "機械学習チュートリアル"],
            "機械": ["機械学習", "機械学習アルゴリズム", "機械学習チュートリアル"],
        }

        # 部分マッチも考慮
        matches = []
        for prefix, completions in mock_data.items():
            if prefix.startswith(query_lower):
                matches.extend(completions)
            elif query_lower in prefix:
                matches.extend(completions)

        # 重複除去・頻度順ソート・結果数制限
        unique_matches = list(dict.fromkeys(matches))  # 順序保持で重複除去
        return unique_matches[:max_results]

    def _get_frequency(self, text: str) -> int:
        """テキストの検索頻度取得（モック実装）"""
        # 実際の実装では検索ログDBから取得
        frequency_dict = {
            "machine learning": 15000,
            "machine learning algorithms": 8500,
            "machine learning tutorial": 6200,
            "neural networks": 12000,
            "deep learning": 18000,
            "artificial intelligence": 20000,
            "data science": 14000,
            "機械学習": 8500,
            "機械学習アルゴリズム": 6200,
        }
        return frequency_dict.get(text, 1000)  # デフォルト頻度

    def _calculate_autocomplete_score(
        self, query: str, candidate: str, frequency: int, position: int
    ) -> float:
        """自動補完スコア計算"""
        # 基本スコア: 頻度ベース
        base_score = min(frequency / 20000.0, 1.0)

        # プレフィックス一致度
        prefix_match = 1.0 if candidate.lower().startswith(query.lower()) else 0.8

        # 位置ペナルティ
        position_penalty = 1.0 - (position * 0.05)

        # 長さボーナス（短すぎず長すぎず）
        length_bonus = 1.0
        if len(candidate) > len(query) * 3:
            length_bonus = 0.9

        final_score = base_score * prefix_match * position_penalty * length_bonus
        return min(max(final_score, 0.0), 1.0)


class ContextualSuggester(BaseSuggester):
    """コンテキスト候補器"""

    def __init__(self, config: SuggestionsConfig):
        super().__init__(config)
        self.context_weight = config.context_weight

    async def suggest(self, query: str, **kwargs: Any) -> list[SuggestionCandidate]:
        """コンテキスト候補生成"""
        context = kwargs.get("context")
        if not context:
            return []

        max_results = kwargs.get("max_results") or self.config.max_suggestions

        # コンテキスト分析
        context_analysis = self._analyze_context(context)

        # コンテキストベース候補生成
        contextual_candidates = self._generate_contextual_suggestions(
            query, context_analysis, max_results
        )

        return contextual_candidates

    def _analyze_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """コンテキスト分析"""
        analysis = {
            "topics": [],
            "preferences": [],
            "domain": context.get("domain", "general"),
            "recent_searches": context.get("recent_searches", []),
            "context_score": 0.0,
        }

        # ドメイン分析
        domain = context.get("domain", "general")
        if domain == "technology":
            analysis["topics"] = [
                "artificial_intelligence",
                "machine_learning",
                "programming",
            ]
        elif domain == "science":
            analysis["topics"] = ["research", "data_analysis", "methodology"]

        # 最近の検索から傾向分析
        recent_searches = context.get("recent_searches", [])
        if recent_searches:
            # 簡易的な傾向分析
            if any("AI" in search or "ML" in search for search in recent_searches):
                analysis["preferences"].append("artificial_intelligence")
            if any("algorithm" in search.lower() for search in recent_searches):
                analysis["preferences"].append("algorithms")
            if any("tutorial" in search.lower() for search in recent_searches):
                analysis["preferences"].append("tutorial")

        # コンテキストスコア計算
        analysis["context_score"] = (
            len(analysis["topics"]) * 0.2 + len(analysis["preferences"]) * 0.3
        )
        analysis["context_score"] = min(analysis["context_score"], 1.0)

        return analysis

    def _generate_contextual_suggestions(
        self, query: str, context_analysis: dict[str, Any], max_results: int
    ) -> list[SuggestionCandidate]:
        """コンテキストベース候補生成"""
        suggestions = []

        # ドメイン固有の候補
        domain_suggestions = self._get_domain_specific_suggestions(
            query, context_analysis["domain"]
        )

        # 傾向ベースの候補
        preference_suggestions = self._get_preference_based_suggestions(
            query, context_analysis["preferences"]
        )

        # 最近の検索関連候補
        recent_suggestions = self._get_recent_search_related_suggestions(
            query, context_analysis["recent_searches"]
        )

        # 全候補をマージ
        all_candidates = (
            domain_suggestions + preference_suggestions + recent_suggestions
        )

        # 重複除去・スコア順ソート
        unique_candidates = {}
        for candidate in all_candidates:
            if candidate["text"] not in unique_candidates:
                unique_candidates[candidate["text"]] = candidate
            else:
                # より高いスコアを保持
                if candidate["score"] > unique_candidates[candidate["text"]]["score"]:
                    unique_candidates[candidate["text"]] = candidate

        # SuggestionCandidateオブジェクトに変換
        for _i, (text, data) in enumerate(
            sorted(
                unique_candidates.items(), key=lambda x: x[1]["score"], reverse=True
            )[:max_results]
        ):
            suggestion = SuggestionCandidate(
                text=text,
                type=SuggestionType.CONTEXTUAL,
                score=data["score"],
                frequency=data.get("frequency", 5000),
                metadata={
                    "context_match": data.get("context_match", []),
                    "relevance": data.get("relevance", "medium"),
                    "domain": context_analysis["domain"],
                },
            )
            suggestions.append(suggestion)

        return suggestions

    def _get_domain_specific_suggestions(
        self, query: str, domain: str
    ) -> list[dict[str, Any]]:
        """ドメイン固有候補取得"""
        domain_data = {
            "technology": {
                "machine learn": [
                    {
                        "text": "machine learning algorithms",
                        "score": 0.88,
                        "frequency": 8500,
                    },
                    {
                        "text": "machine learning frameworks",
                        "score": 0.84,
                        "frequency": 7200,
                    },
                ],
                "ai": [
                    {"text": "AI applications", "score": 0.87, "frequency": 9200},
                    {"text": "AI ethics", "score": 0.82, "frequency": 6800},
                ],
            },
            "science": {
                "machine learn": [
                    {
                        "text": "machine learning research",
                        "score": 0.85,
                        "frequency": 7800,
                    },
                    {
                        "text": "machine learning methodology",
                        "score": 0.81,
                        "frequency": 6500,
                    },
                ],
            },
        }

        suggestions = []
        domain_queries = domain_data.get(domain, {})

        for pattern, candidates in domain_queries.items():
            if pattern in query.lower():
                for candidate in candidates:
                    candidate["context_match"] = [pattern]
                    candidate["relevance"] = "high"
                    suggestions.append(candidate)

        return suggestions

    def _get_preference_based_suggestions(
        self, query: str, preferences: list[str]
    ) -> list[dict[str, Any]]:
        """傾向ベース候補取得"""
        suggestions = []

        preference_data = {
            "algorithms": [
                {
                    "text": "machine learning algorithms",
                    "score": 0.86,
                    "frequency": 8500,
                },
                {"text": "neural network algorithms", "score": 0.83, "frequency": 7200},
            ],
            "tutorial": [
                {"text": "machine learning tutorial", "score": 0.84, "frequency": 6200},
                {"text": "AI tutorial", "score": 0.80, "frequency": 5800},
            ],
        }

        for preference in preferences:
            if preference in preference_data:
                for candidate in preference_data[preference]:
                    candidate_text = str(candidate.get("text", ""))
                    if query.lower() in candidate_text.lower():
                        candidate["context_match"] = [preference]
                        candidate["relevance"] = "high"
                        suggestions.append(candidate)

        return suggestions

    def _get_recent_search_related_suggestions(
        self, query: str, recent_searches: list[str]
    ) -> list[dict[str, Any]]:
        """最近の検索関連候補取得"""
        suggestions = []

        for recent_search in recent_searches:
            if (
                query.lower() in recent_search.lower()
                or recent_search.lower() in query.lower()
            ):
                # 最近の検索と関連する候補を生成
                related_text = f"{query} {recent_search}".strip()
                if len(related_text.split()) <= 4:  # 長すぎる候補は除外
                    suggestions.append(
                        {
                            "text": related_text,
                            "score": 0.82,
                            "frequency": 6000,
                            "context_match": [recent_search],
                            "relevance": "medium",
                        }
                    )

        return suggestions


class PersonalizedSuggester(BaseSuggester):
    """パーソナライズ候補器"""

    def __init__(self, config: SuggestionsConfig):
        super().__init__(config)
        self.personalization_weight = config.personalization_weight

    async def suggest(self, query: str, **kwargs: Any) -> list[SuggestionCandidate]:
        """パーソナライズ候補生成"""
        user_id = kwargs.get("user_id")
        if not user_id:
            return []

        max_results = kwargs.get("max_results") or self.config.max_suggestions

        # ユーザープロファイル取得
        user_profile = await self._get_user_profile(user_id)

        # パーソナライズ候補生成
        personalized_candidates = self._generate_personalized_suggestions(
            query, user_profile, max_results
        )

        return personalized_candidates

    async def _get_user_profile(self, user_id: str) -> dict[str, Any]:
        """ユーザープロファイル取得（モック実装）"""
        # 実際の実装ではユーザーDBから取得
        mock_profiles = {
            "user123": {
                "interests": ["machine_learning", "python", "tutorial"],
                "search_history": ["ML algorithms", "Python tutorial", "deep learning"],
                "interaction_patterns": {
                    "preferred_time": "morning",
                    "session_length": "long",
                },
                "personalization_score": 0.78,
            },
            "user456": {
                "interests": ["data_science", "statistics", "visualization"],
                "search_history": ["data analysis", "matplotlib", "pandas"],
                "interaction_patterns": {
                    "preferred_time": "evening",
                    "session_length": "medium",
                },
                "personalization_score": 0.82,
            },
        }

        return mock_profiles.get(
            user_id,
            {
                "interests": [],
                "search_history": [],
                "interaction_patterns": {},
                "personalization_score": 0.5,
            },
        )

    def _generate_personalized_suggestions(
        self, query: str, user_profile: dict[str, Any], max_results: int
    ) -> list[SuggestionCandidate]:
        """パーソナライズ候補生成"""
        suggestions = []
        interests = user_profile.get("interests", [])
        search_history = user_profile.get("search_history", [])

        # 興味ベースの候補
        interest_suggestions = self._get_interest_based_suggestions(query, interests)

        # 検索履歴ベースの候補
        history_suggestions = self._get_history_based_suggestions(query, search_history)

        # 全候補をマージ・スコア調整
        all_candidates = interest_suggestions + history_suggestions

        # パーソナライゼーションスコア適用
        personalization_score = user_profile.get("personalization_score", 0.5)

        for candidate in all_candidates:
            # パーソナライゼーション重みを適用
            candidate["score"] = candidate["score"] * (
                1 + personalization_score * self.personalization_weight
            )
            candidate["personalization_score"] = personalization_score

        # 重複除去・スコア順ソート
        unique_candidates: dict[str, dict[str, Any]] = {}
        for candidate in all_candidates:
            text = str(candidate.get("text", ""))
            if (
                text not in unique_candidates
                or candidate["score"] > unique_candidates[text]["score"]
            ):
                unique_candidates[text] = candidate

        # SuggestionCandidateオブジェクトに変換
        for _i, (text, data) in enumerate(
            sorted(
                unique_candidates.items(), key=lambda x: x[1]["score"], reverse=True
            )[:max_results]
        ):
            suggestion = SuggestionCandidate(
                text=text,
                type=SuggestionType.PERSONALIZED,
                score=data["score"],
                frequency=data.get("frequency", 5000),
                metadata={
                    "user_preference": data.get("user_preference", ""),
                    "past_interactions": data.get("past_interactions", []),
                    "personalization_score": data.get("personalization_score", 0.5),
                },
            )
            suggestions.append(suggestion)

        return suggestions

    def _get_interest_based_suggestions(
        self, query: str, interests: list[str]
    ) -> list[dict[str, Any]]:
        """興味ベース候補取得"""
        suggestions = []

        interest_data = {
            "machine_learning": [
                {"text": "machine learning tutorial", "score": 0.92, "frequency": 6200},
                {"text": "machine learning projects", "score": 0.88, "frequency": 5800},
            ],
            "python": [
                {"text": "machine learning python", "score": 0.88, "frequency": 5800},
                {"text": "python machine learning", "score": 0.85, "frequency": 5500},
            ],
            "tutorial": [
                {"text": "machine learning tutorial", "score": 0.90, "frequency": 6200},
            ],
        }

        for interest in interests:
            if interest in interest_data:
                for candidate in interest_data[interest]:
                    candidate_text = str(candidate.get("text", ""))
                    if query.lower() in candidate_text.lower():
                        candidate["user_preference"] = interest
                        candidate["past_interactions"] = [interest, "tutorial"]
                        suggestions.append(candidate)

        return suggestions

    def _get_history_based_suggestions(
        self, query: str, search_history: list[str]
    ) -> list[dict[str, Any]]:
        """検索履歴ベース候補取得"""
        suggestions = []

        for history_item in search_history:
            if query.lower() in history_item.lower():
                suggestions.append(
                    {
                        "text": history_item,
                        "score": 0.85,
                        "frequency": 5000,
                        "user_preference": "history",
                        "past_interactions": [history_item],
                    }
                )

        return suggestions


class TrendingSuggester(BaseSuggester):
    """トレンド候補器"""

    def __init__(self, config: SuggestionsConfig):
        super().__init__(config)
        self.trending_window_hours = config.trending_window_hours
        self.trend_threshold = config.trend_threshold

    async def suggest(self, query: str, **kwargs: Any) -> list[SuggestionCandidate]:
        """トレンド候補生成"""
        max_results = kwargs.get("max_results") or self.config.max_suggestions

        # トレンド分析
        trends = await self._analyze_trends(query, self.trending_window_hours)

        # トレンド候補生成
        trending_candidates = []
        for trend in trends[:max_results]:
            suggestion = SuggestionCandidate(
                text=trend["query"],
                type=SuggestionType.TRENDING,
                score=trend["trend_score"],
                frequency=trend["frequency"],
                recency=datetime.now() - timedelta(minutes=30),
                metadata={
                    "trend_score": trend["trend_score"],
                    "growth_rate": trend["growth_rate"],
                    "time_period": trend["time_period"],
                },
            )
            trending_candidates.append(suggestion)

        return trending_candidates

    async def _analyze_trends(
        self, query: str, time_window: int
    ) -> list[dict[str, Any]]:
        """トレンド分析（モック実装）"""
        # 実際の実装では時系列検索ログを分析
        mock_trends = {
            "machine learn": [
                {
                    "query": "machine learning ChatGPT",
                    "trend_score": 0.91,
                    "growth_rate": 2.3,
                    "frequency": 12000,
                    "time_period": "24h",
                },
                {
                    "query": "machine learning LLM",
                    "trend_score": 0.86,
                    "growth_rate": 1.8,
                    "frequency": 9800,
                    "time_period": "24h",
                },
            ],
            "ai": [
                {
                    "query": "AI tools",
                    "trend_score": 0.89,
                    "growth_rate": 2.1,
                    "frequency": 11500,
                    "time_period": "24h",
                },
            ],
        }

        trends = []
        for pattern, trend_list in mock_trends.items():
            if pattern in query.lower():
                trends.extend(trend_list)

        # トレンドスコア順でソート
        def get_trend_score(x: dict[str, Any]) -> float:
            score = x.get("trend_score", 0.0)
            return float(score) if score is not None else 0.0

        trends.sort(key=get_trend_score, reverse=True)
        return trends


class TypoCorrector:
    """タイポ修正器"""

    def __init__(self, config: SuggestionsConfig):
        self.config = config
        self.max_edit_distance = config.max_edit_distance
        self.min_confidence = config.min_confidence

    async def correct(self, query: str) -> list[SuggestionCandidate]:
        """タイポ修正"""
        corrections = await self._find_spelling_corrections(query)

        candidates = []
        for correction in corrections:
            if correction["confidence"] >= self.min_confidence:
                candidate = SuggestionCandidate(
                    text=correction["corrected"],
                    type=SuggestionType.TYPO_CORRECTION,
                    score=correction["confidence"],
                    frequency=15000,  # 修正後の頻度
                    metadata={
                        "original_query": correction["original"],
                        "edit_distance": correction["edit_distance"],
                        "confidence": correction["confidence"],
                        "corrections": correction["corrections"],
                    },
                )
                candidates.append(candidate)

        return candidates

    async def _find_spelling_corrections(self, query: str) -> list[dict[str, Any]]:
        """スペル修正検索（モック実装）"""
        # 実際の実装では spellchecker, symspell等を使用
        correction_dict = {
            "machne learing": {
                "corrected": "machine learning",
                "confidence": 0.85,
                "edit_distance": 2,
                "corrections": [{"machne": "machine"}, {"learing": "learning"}],
            },
            "artifical inteligence": {
                "corrected": "artificial intelligence",
                "confidence": 0.82,
                "edit_distance": 2,
                "corrections": [
                    {"artifical": "artificial"},
                    {"inteligence": "intelligence"},
                ],
            },
            "nueral network": {
                "corrected": "neural network",
                "confidence": 0.88,
                "edit_distance": 1,
                "corrections": [{"nueral": "neural"}],
            },
        }

        corrections = []
        if query in correction_dict:
            correction = correction_dict[query]
            correction["original"] = query
            corrections.append(correction)

        return corrections

    def _calculate_edit_distance(self, str1: str, str2: str) -> int:
        """編集距離計算（レーベンシュタイン距離）"""
        if len(str1) < len(str2):
            return self._calculate_edit_distance(str2, str1)

        if len(str2) == 0:
            return len(str1)

        previous_row = list(range(len(str2) + 1))
        for i, c1 in enumerate(str1):
            current_row = [i + 1]
            for j, c2 in enumerate(str2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


class SearchSuggestionsService:
    """検索候補メインサービス"""

    def __init__(self, config: SuggestionsConfig):
        self.config = config
        self.suggesters = self._create_suggesters()
        self.typo_corrector = (
            TypoCorrector(config) if config.enable_typo_correction else None
        )
        self.cache: dict[str, tuple[SuggestionResult, datetime]] = (
            {}
        )  # 簡易キャッシュ実装

    def _create_suggesters(self) -> dict[SuggestionType, BaseSuggester]:
        """候補器インスタンス作成"""
        suggesters: dict[SuggestionType, BaseSuggester] = {}

        for suggestion_type in self.config.suggestion_types:
            if suggestion_type == SuggestionType.AUTOCOMPLETE:
                suggesters[suggestion_type] = AutocompleteSuggester(self.config)
            elif suggestion_type == SuggestionType.CONTEXTUAL:
                suggesters[suggestion_type] = ContextualSuggester(self.config)
            elif suggestion_type == SuggestionType.PERSONALIZED:
                suggesters[suggestion_type] = PersonalizedSuggester(self.config)
            elif suggestion_type == SuggestionType.TRENDING:
                suggesters[suggestion_type] = TrendingSuggester(self.config)

        return suggesters

    async def get_suggestions(self, request: SuggestionRequest) -> SuggestionResult:
        """検索候補取得"""
        start_time = datetime.now()

        try:
            # 入力バリデーション
            if len(request.query.strip()) < self.config.min_query_length:
                return SuggestionResult(
                    success=False,
                    query=request.query,
                    suggestions=[],
                    suggestion_time=0.0,
                    error_message="Query is too short for suggestions",
                )

            # キャッシュチェック
            if self.config.enable_caching:
                cache_key = self._get_cache_key(request)
                cached_result = await self._get_from_cache(cache_key)
                if cached_result:
                    return cached_result

            # 各候補タイプを並行実行
            all_suggestions = []
            max_suggestions = request.max_suggestions or self.config.max_suggestions

            # 自動補完候補
            if SuggestionType.AUTOCOMPLETE in self.config.suggestion_types:
                autocomplete_suggestions = await self._get_autocomplete_suggestions(
                    request.query, max_suggestions
                )
                all_suggestions.extend(autocomplete_suggestions)

            # コンテキスト候補
            if (
                SuggestionType.CONTEXTUAL in self.config.suggestion_types
                and request.context
            ):
                contextual_suggestions = await self._get_contextual_suggestions(
                    request.query, request.context, max_suggestions
                )
                all_suggestions.extend(contextual_suggestions)

            # パーソナライズ候補
            if (
                SuggestionType.PERSONALIZED in self.config.suggestion_types
                and request.user_id
            ):
                personalized_suggestions = await self._get_personalized_suggestions(
                    request.query, request.user_id, max_suggestions
                )
                all_suggestions.extend(personalized_suggestions)

            # トレンド候補
            if SuggestionType.TRENDING in self.config.suggestion_types:
                trending_suggestions = await self._get_trending_suggestions(
                    request.query, max_suggestions
                )
                all_suggestions.extend(trending_suggestions)

            # タイポ修正
            typo_corrections = []
            if request.include_corrections and self.typo_corrector:
                typo_corrections = await self._get_typo_corrections(
                    request.query, max_suggestions
                )
                all_suggestions.extend(typo_corrections)

            # 候補のフィルタリング・ランキング・重複除去
            filtered_suggestions = self._filter_suggestions_by_threshold(
                all_suggestions, self.config.similarity_threshold
            )

            final_suggestions = self._rank_and_deduplicate_suggestions(
                filtered_suggestions, max_suggestions
            )

            # クエリ補完生成
            completions = []
            if request.include_completions:
                suggestion_texts = [
                    s.text
                    for s in final_suggestions
                    if s.type == SuggestionType.AUTOCOMPLETE
                ]
                completions = self._generate_query_completions(
                    request.query, suggestion_texts
                )

            end_time = datetime.now()
            suggestion_time = (end_time - start_time).total_seconds()

            result = SuggestionResult(
                success=True,
                query=request.query,
                suggestions=final_suggestions,
                suggestion_time=suggestion_time,
                completions=completions,
            )

            # キャッシュに保存
            if self.config.enable_caching:
                await self._set_cache(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Suggestion generation failed: {e}")
            end_time = datetime.now()
            suggestion_time = (end_time - start_time).total_seconds()

            return SuggestionResult(
                success=False,
                query=request.query,
                suggestions=[],
                suggestion_time=suggestion_time,
                error_message=str(e),
            )

    async def _get_autocomplete_suggestions(
        self, query: str, max_results: int
    ) -> list[SuggestionCandidate]:
        """自動補完候補取得"""
        suggester = self.suggesters.get(SuggestionType.AUTOCOMPLETE)
        if suggester:
            return await suggester.suggest(query, max_results=max_results)
        return []

    async def _get_contextual_suggestions(
        self, query: str, context: dict[str, Any], max_results: int
    ) -> list[SuggestionCandidate]:
        """コンテキスト候補取得"""
        suggester = self.suggesters.get(SuggestionType.CONTEXTUAL)
        if suggester:
            return await suggester.suggest(
                query, context=context, max_results=max_results
            )
        return []

    async def _get_personalized_suggestions(
        self, query: str, user_id: str, max_results: int
    ) -> list[SuggestionCandidate]:
        """パーソナライズ候補取得"""
        suggester = self.suggesters.get(SuggestionType.PERSONALIZED)
        if suggester:
            return await suggester.suggest(
                query, user_id=user_id, max_results=max_results
            )
        return []

    async def _get_trending_suggestions(
        self, query: str, max_results: int
    ) -> list[SuggestionCandidate]:
        """トレンド候補取得"""
        suggester = self.suggesters.get(SuggestionType.TRENDING)
        if suggester:
            return await suggester.suggest(query, max_results=max_results)
        return []

    async def _get_typo_corrections(
        self, query: str, max_results: int
    ) -> list[SuggestionCandidate]:
        """タイポ修正取得"""
        if self.typo_corrector:
            return await self.typo_corrector.correct(query)
        return []

    def _filter_suggestions_by_threshold(
        self, suggestions: list[SuggestionCandidate], threshold: float
    ) -> list[SuggestionCandidate]:
        """閾値による候補フィルタリング"""
        return [s for s in suggestions if s.score >= threshold]

    def _rank_and_deduplicate_suggestions(
        self, suggestions: list[SuggestionCandidate], max_suggestions: int
    ) -> list[SuggestionCandidate]:
        """候補ランキング・重複除去"""
        # 重複除去（テキストが同じものは高スコアを保持）
        unique_suggestions: dict[str, SuggestionCandidate] = {}
        for suggestion in suggestions:
            text = suggestion.text.lower()
            if (
                text not in unique_suggestions
                or suggestion.score > unique_suggestions[text].score
            ):
                unique_suggestions[text] = suggestion

        # スコア順でソート
        ranked_suggestions = sorted(
            unique_suggestions.values(), key=lambda s: s.score, reverse=True
        )

        return ranked_suggestions[:max_suggestions]

    def _generate_query_completions(
        self, original_query: str, suggestion_texts: list[str]
    ) -> list[QueryCompletion]:
        """クエリ補完生成"""
        completions = []

        for text in suggestion_texts:
            if text.lower().startswith(original_query.lower()) and len(text) > len(
                original_query
            ):
                completion_part = text[len(original_query) :].strip()
                if completion_part:
                    completion = QueryCompletion(
                        original_query=original_query,
                        completed_query=text,
                        completion_part=completion_part,
                        confidence=0.8,  # 簡易的な信頼度
                        type="suffix",
                    )
                    completions.append(completion)

        return completions

    def _get_cache_key(self, request: SuggestionRequest) -> str:
        """キャッシュキー生成"""
        cache_content = {
            "query": request.query,
            "user_id": request.user_id,
            "context": request.context,
            "max_suggestions": request.max_suggestions or self.config.max_suggestions,
            "include_corrections": request.include_corrections,
            "include_completions": request.include_completions,
        }
        content_str = json.dumps(cache_content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()

    async def _get_from_cache(self, cache_key: str) -> SuggestionResult | None:
        """キャッシュから取得"""
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.config.cache_ttl):
                # SuggestionResultオブジェクトを正しくコピー
                result = SuggestionResult(
                    success=cached_data.success,
                    query=cached_data.query,
                    suggestions=cached_data.suggestions.copy(),
                    suggestion_time=cached_data.suggestion_time,
                    error_message=cached_data.error_message,
                    cache_hit=True,
                    completions=cached_data.completions.copy(),
                )
                return result
            else:
                # 期限切れキャッシュを削除
                del self.cache[cache_key]
        return None

    async def _set_cache(self, cache_key: str, result: SuggestionResult) -> None:
        """キャッシュに保存"""
        self.cache[cache_key] = (result, datetime.now())
