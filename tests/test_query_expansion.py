"""クエリ拡張サービスのテストモジュール

TDD実装：クエリ拡張・同義語・関連語・概念拡張機能
"""

import pytest
import asyncio
from typing import Dict, Any, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.services.query_expansion import (
    QueryExpansionService,
    QueryExpansionConfig,
    ExpansionRequest,
    ExpansionResult,
    ExpansionMethod,
    SynonymExpander,
    SemanticExpander,
    ConceptualExpander,
    StatisticalExpander,
    QueryTerm,
    ExpandedQuery,
)


class TestQueryExpansionService:
    """クエリ拡張サービスのテストクラス"""

    @pytest.fixture
    def basic_expansion_config(self) -> QueryExpansionConfig:
        """基本クエリ拡張設定"""
        return QueryExpansionConfig(
            expansion_methods=[ExpansionMethod.SYNONYM, ExpansionMethod.SEMANTIC],
            max_expansions=5,
            similarity_threshold=0.7,
            enable_concept_expansion=True,
            enable_statistical_expansion=False,
            expansion_weight=0.3,
            preserve_original_weight=0.7,
        )

    @pytest.fixture
    def semantic_expansion_config(self) -> QueryExpansionConfig:
        """セマンティック拡張設定"""
        return QueryExpansionConfig(
            expansion_methods=[ExpansionMethod.SEMANTIC, ExpansionMethod.CONCEPTUAL],
            max_expansions=10,
            similarity_threshold=0.6,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            enable_concept_expansion=True,
        )

    @pytest.fixture
    def statistical_expansion_config(self) -> QueryExpansionConfig:
        """統計的拡張設定"""
        return QueryExpansionConfig(
            expansion_methods=[ExpansionMethod.STATISTICAL],
            max_expansions=8,
            enable_statistical_expansion=True,
            statistical_corpus_size=100000,
            min_term_frequency=10,
        )

    @pytest.fixture
    def sample_expansion_request(self) -> ExpansionRequest:
        """サンプル拡張リクエスト"""
        return ExpansionRequest(
            query="machine learning algorithms",
            language="en",
            domain="technology",
            max_expansions=5,
            preserve_original=True,
            expansion_types=["synonym", "semantic", "conceptual"],
        )

    @pytest.fixture
    def japanese_expansion_request(self) -> ExpansionRequest:
        """日本語拡張リクエスト"""
        return ExpansionRequest(
            query="機械学習アルゴリズム",
            language="ja",
            domain="technology",
            max_expansions=7,
            preserve_original=True,
            expansion_types=["synonym", "semantic"],
        )

    @pytest.fixture
    def query_terms(self) -> List[QueryTerm]:
        """クエリ用語リスト"""
        return [
            QueryTerm(
                term="machine",
                pos_tag="NOUN",
                importance=0.8,
                synonyms=["device", "apparatus", "system"],
                semantic_neighbors=["computer", "robot", "automation"],
            ),
            QueryTerm(
                term="learning",
                pos_tag="NOUN",
                importance=0.9,
                synonyms=["education", "training", "study"],
                semantic_neighbors=["training", "education", "acquisition"],
            ),
            QueryTerm(
                term="algorithms",
                pos_tag="NOUN",
                importance=0.7,
                synonyms=["methods", "procedures", "techniques"],
                semantic_neighbors=["techniques", "methods", "approaches"],
            ),
        ]

    @pytest.mark.unit
    async def test_query_expansion_service_initialization(
        self, basic_expansion_config: QueryExpansionConfig
    ):
        """クエリ拡張サービス初期化テスト"""
        expansion_service = QueryExpansionService(config=basic_expansion_config)
        
        assert expansion_service.config == basic_expansion_config
        assert expansion_service.config.max_expansions == 5
        assert expansion_service.config.similarity_threshold == 0.7
        assert ExpansionMethod.SYNONYM in expansion_service.config.expansion_methods
        assert ExpansionMethod.SEMANTIC in expansion_service.config.expansion_methods

    @pytest.mark.unit
    async def test_query_term_extraction(
        self,
        basic_expansion_config: QueryExpansionConfig,
        sample_expansion_request: ExpansionRequest,
    ):
        """クエリ用語抽出テスト"""
        expansion_service = QueryExpansionService(config=basic_expansion_config)
        
        with patch.object(expansion_service, '_extract_query_terms', new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = [
                QueryTerm(term="machine", pos_tag="NOUN", importance=0.8),
                QueryTerm(term="learning", pos_tag="NOUN", importance=0.9),
                QueryTerm(term="algorithms", pos_tag="NOUN", importance=0.7),
            ]
            
            terms = await expansion_service._extract_query_terms(sample_expansion_request.query)
            
            assert len(terms) == 3
            assert all(isinstance(term, QueryTerm) for term in terms)
            assert terms[1].term == "learning"
            assert terms[1].importance == 0.9

    @pytest.mark.unit
    async def test_synonym_expansion(
        self,
        basic_expansion_config: QueryExpansionConfig,
        sample_expansion_request: ExpansionRequest,
    ):
        """同義語拡張テスト"""
        expansion_service = QueryExpansionService(config=basic_expansion_config)
        
        with patch.object(expansion_service, '_expand_with_synonyms', new_callable=AsyncMock) as mock_synonyms:
            mock_synonyms.return_value = {
                "machine": ["device", "apparatus", "system"],
                "learning": ["education", "training", "study"],
                "algorithms": ["methods", "procedures", "techniques"],
            }
            
            synonyms = await expansion_service._expand_with_synonyms(
                ["machine", "learning", "algorithms"], sample_expansion_request.language
            )
            
            assert "machine" in synonyms
            assert "device" in synonyms["machine"]
            assert "training" in synonyms["learning"]
            assert len(synonyms["algorithms"]) == 3

    @pytest.mark.unit
    async def test_semantic_expansion(
        self,
        semantic_expansion_config: QueryExpansionConfig,
        sample_expansion_request: ExpansionRequest,
    ):
        """セマンティック拡張テスト"""
        expansion_service = QueryExpansionService(config=semantic_expansion_config)
        
        with patch.object(expansion_service, '_expand_with_semantics', new_callable=AsyncMock) as mock_semantics:
            mock_semantics.return_value = {
                "machine learning": [
                    ("artificial intelligence", 0.85),
                    ("neural networks", 0.78),
                    ("deep learning", 0.76),
                ],
                "algorithms": [
                    ("techniques", 0.82),
                    ("methods", 0.79),
                    ("approaches", 0.75),
                ],
            }
            
            semantic_terms = await expansion_service._expand_with_semantics(
                sample_expansion_request.query, semantic_expansion_config.similarity_threshold
            )
            
            assert "machine learning" in semantic_terms
            assert len(semantic_terms["machine learning"]) == 3
            assert semantic_terms["machine learning"][0][0] == "artificial intelligence"
            assert semantic_terms["machine learning"][0][1] == 0.85

    @pytest.mark.unit
    async def test_conceptual_expansion(
        self,
        semantic_expansion_config: QueryExpansionConfig,
        sample_expansion_request: ExpansionRequest,
    ):
        """概念拡張テスト"""
        expansion_service = QueryExpansionService(config=semantic_expansion_config)
        
        with patch.object(expansion_service, '_expand_with_concepts', new_callable=AsyncMock) as mock_concepts:
            mock_concepts.return_value = {
                "machine_learning": {
                    "broader_concepts": ["artificial_intelligence", "computer_science"],
                    "narrower_concepts": ["supervised_learning", "unsupervised_learning"],
                    "related_concepts": ["data_mining", "pattern_recognition"],
                },
                "algorithms": {
                    "broader_concepts": ["computer_science", "mathematics"],
                    "narrower_concepts": ["sorting_algorithms", "search_algorithms"],
                    "related_concepts": ["data_structures", "optimization"],
                },
            }
            
            concepts = await expansion_service._expand_with_concepts(
                ["machine_learning", "algorithms"], sample_expansion_request.domain
            )
            
            assert "machine_learning" in concepts
            assert "broader_concepts" in concepts["machine_learning"]
            assert "artificial_intelligence" in concepts["machine_learning"]["broader_concepts"]
            assert "supervised_learning" in concepts["machine_learning"]["narrower_concepts"]

    @pytest.mark.unit
    async def test_statistical_expansion(
        self,
        statistical_expansion_config: QueryExpansionConfig,
        sample_expansion_request: ExpansionRequest,
    ):
        """統計的拡張テスト"""
        expansion_service = QueryExpansionService(config=statistical_expansion_config)
        
        with patch.object(expansion_service, '_expand_with_statistics', new_callable=AsyncMock) as mock_stats:
            mock_stats.return_value = {
                "machine learning": [
                    ("deep learning", 0.65, 1250),
                    ("neural networks", 0.58, 980),
                    ("data science", 0.52, 1450),
                ],
                "algorithms": [
                    ("techniques", 0.72, 890),
                    ("methods", 0.68, 1120),
                    ("optimization", 0.61, 750),
                ],
            }
            
            statistical_terms = await expansion_service._expand_with_statistics(
                sample_expansion_request.query, statistical_expansion_config.min_term_frequency
            )
            
            assert "machine learning" in statistical_terms
            assert len(statistical_terms["algorithms"]) == 3
            assert statistical_terms["algorithms"][0][0] == "techniques"
            assert statistical_terms["algorithms"][0][2] == 890  # frequency

    @pytest.mark.unit
    async def test_query_expansion_execution(
        self,
        basic_expansion_config: QueryExpansionConfig,
        sample_expansion_request: ExpansionRequest,
    ):
        """クエリ拡張実行テスト"""
        expansion_service = QueryExpansionService(config=basic_expansion_config)
        
        # Mock all expansion methods
        with patch.object(expansion_service, '_extract_query_terms', new_callable=AsyncMock) as mock_terms, \
             patch.object(expansion_service, '_expand_with_synonyms', new_callable=AsyncMock) as mock_synonyms, \
             patch.object(expansion_service, '_expand_with_semantics', new_callable=AsyncMock) as mock_semantics:
            
            mock_terms.return_value = [
                QueryTerm(term="machine", pos_tag="NOUN", importance=0.8),
                QueryTerm(term="learning", pos_tag="NOUN", importance=0.9),
                QueryTerm(term="algorithms", pos_tag="NOUN", importance=0.7),
            ]
            
            mock_synonyms.return_value = {
                "machine": ["device", "apparatus"],
                "learning": ["training", "education"],
                "algorithms": ["methods", "techniques"],
            }
            
            mock_semantics.return_value = {
                "machine learning": [("artificial intelligence", 0.85), ("neural networks", 0.78)],
                "algorithms": [("techniques", 0.82), ("methods", 0.79)],
            }
            
            result = await expansion_service.expand_query(sample_expansion_request)
            
            assert isinstance(result, ExpansionResult)
            assert result.success is True
            assert result.original_query == sample_expansion_request.query
            assert len(result.expanded_terms) > 0
            assert result.expansion_time > 0

    @pytest.mark.unit
    async def test_expanded_query_generation(
        self,
        basic_expansion_config: QueryExpansionConfig,
        sample_expansion_request: ExpansionRequest,
    ):
        """拡張クエリ生成テスト"""
        expansion_service = QueryExpansionService(config=basic_expansion_config)
        
        expanded_terms = {
            "machine": ["device", "apparatus"],
            "learning": ["training", "education"],
            "algorithms": ["methods", "techniques"],
        }
        
        expanded_query = expansion_service._generate_expanded_query(
            sample_expansion_request.query, expanded_terms, basic_expansion_config.preserve_original_weight
        )
        
        assert isinstance(expanded_query, ExpandedQuery)
        assert expanded_query.original_query == sample_expansion_request.query
        assert expanded_query.expanded_query != sample_expansion_request.query
        assert len(expanded_query.expansion_terms) > 0
        assert expanded_query.total_weight > 0

    @pytest.mark.unit
    async def test_japanese_query_expansion(
        self,
        basic_expansion_config: QueryExpansionConfig,
        japanese_expansion_request: ExpansionRequest,
    ):
        """日本語クエリ拡張テスト"""
        expansion_service = QueryExpansionService(config=basic_expansion_config)
        
        with patch.object(expansion_service, '_extract_query_terms', new_callable=AsyncMock) as mock_terms, \
             patch.object(expansion_service, '_expand_with_synonyms', new_callable=AsyncMock) as mock_synonyms:
            
            mock_terms.return_value = [
                QueryTerm(term="機械学習", pos_tag="NOUN", importance=0.9),
                QueryTerm(term="アルゴリズム", pos_tag="NOUN", importance=0.8),
            ]
            
            mock_synonyms.return_value = {
                "機械学習": ["マシンラーニング", "ML", "自動学習"],
                "アルゴリズム": ["手法", "方法", "プロシージャ"],
            }
            
            result = await expansion_service.expand_query(japanese_expansion_request)
            
            assert result.success is True
            assert result.original_query == japanese_expansion_request.query
            assert result.language == "ja"
            # 日本語の拡張用語が含まれていることを確認
            japanese_terms = [term for term in result.expanded_terms if any('\u3040' <= char <= '\u30ff' for char in term)]
            assert len(japanese_terms) > 0

    @pytest.mark.unit
    async def test_multi_domain_expansion(
        self,
        basic_expansion_config: QueryExpansionConfig,
    ):
        """マルチドメイン拡張テスト"""
        expansion_service = QueryExpansionService(config=basic_expansion_config)
        
        # Technology domain request
        tech_request = ExpansionRequest(
            query="neural network",
            language="en",
            domain="technology",
            max_expansions=5,
        )
        
        # Medical domain request
        medical_request = ExpansionRequest(
            query="neural network",
            language="en",
            domain="medical",
            max_expansions=5,
        )
        
        with patch.object(expansion_service, '_expand_with_concepts', new_callable=AsyncMock) as mock_concepts:
            # Domain-specific concept returns
            mock_concepts.side_effect = [
                {
                    "neural_network": {
                        "related_concepts": ["artificial_intelligence", "machine_learning", "deep_learning"],
                    }
                },
                {
                    "neural_network": {
                        "related_concepts": ["nervous_system", "brain_function", "neuroscience"],
                    }
                },
            ]
            
            tech_result = await expansion_service._expand_with_concepts(["neural_network"], "technology")
            medical_result = await expansion_service._expand_with_concepts(["neural_network"], "medical")
            
            # Different domains should return different related concepts
            tech_concepts = tech_result["neural_network"]["related_concepts"]
            medical_concepts = medical_result["neural_network"]["related_concepts"]
            
            assert "artificial_intelligence" in tech_concepts
            assert "nervous_system" in medical_concepts
            assert set(tech_concepts) != set(medical_concepts)

    @pytest.mark.unit
    async def test_expansion_ranking_and_filtering(
        self,
        basic_expansion_config: QueryExpansionConfig,
        sample_expansion_request: ExpansionRequest,
    ):
        """拡張用語ランキング・フィルタリングテスト"""
        expansion_service = QueryExpansionService(config=basic_expansion_config)
        
        # 大量の拡張用語をシミュレート
        all_expansions = {
            "machine": [
                ("device", 0.95), ("apparatus", 0.85), ("system", 0.75), 
                ("equipment", 0.65), ("tool", 0.55), ("instrument", 0.45)
            ],
            "learning": [
                ("training", 0.92), ("education", 0.88), ("study", 0.78),
                ("acquisition", 0.68), ("development", 0.58), ("growth", 0.48)
            ],
        }
        
        filtered_expansions = expansion_service._filter_and_rank_expansions(
            all_expansions, basic_expansion_config.max_expansions, basic_expansion_config.similarity_threshold
        )
        
        # フィルタリング結果の確認
        for term, expansions in filtered_expansions.items():
            assert len(expansions) <= basic_expansion_config.max_expansions
            for expansion, score in expansions:
                assert score >= basic_expansion_config.similarity_threshold
            # スコア降順でソートされていることを確認
            scores = [score for _, score in expansions]
            assert scores == sorted(scores, reverse=True)

    @pytest.mark.unit
    async def test_expansion_weight_calculation(
        self,
        basic_expansion_config: QueryExpansionConfig,
    ):
        """拡張用語重み計算テスト"""
        expansion_service = QueryExpansionService(config=basic_expansion_config)
        
        original_terms = ["machine", "learning"]
        expansion_terms = ["device", "training", "artificial", "intelligence"]
        
        weights = expansion_service._calculate_expansion_weights(
            original_terms, expansion_terms, basic_expansion_config.expansion_weight
        )
        
        # 重みの合計が1.0になることを確認
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01  # 浮動小数点誤差を考慮
        
        # 元の用語の重みが拡張用語より高いことを確認
        for original_term in original_terms:
            for expansion_term in expansion_terms:
                if original_term in weights and expansion_term in weights:
                    assert weights[original_term] > weights[expansion_term]

    @pytest.mark.unit
    async def test_expansion_caching(
        self,
        basic_expansion_config: QueryExpansionConfig,
        sample_expansion_request: ExpansionRequest,
    ):
        """拡張キャッシュテスト"""
        basic_expansion_config.enable_caching = True
        basic_expansion_config.cache_ttl = 3600
        expansion_service = QueryExpansionService(config=basic_expansion_config)
        
        with patch.object(expansion_service, '_get_cache_key') as mock_cache_key, \
             patch.object(expansion_service, '_get_from_cache', new_callable=AsyncMock) as mock_get_cache, \
             patch.object(expansion_service, '_set_cache', new_callable=AsyncMock) as mock_set_cache, \
             patch.object(expansion_service, '_extract_query_terms', new_callable=AsyncMock) as mock_terms:
            
            cache_key = "expansion_cache_key_123"
            mock_cache_key.return_value = cache_key
            mock_get_cache.return_value = None  # キャッシュなし
            mock_terms.return_value = [QueryTerm(term="test", pos_tag="NOUN", importance=0.8)]
            
            # 初回実行
            result1 = await expansion_service.expand_query(sample_expansion_request)
            
            # キャッシュに保存されることを確認
            mock_set_cache.assert_called_once()
            
            # 2回目はキャッシュから取得
            cached_result = ExpansionResult(
                success=True,
                original_query=sample_expansion_request.query,
                expanded_terms=["test", "example"],
                expanded_query="test example",
                expansion_time=0.00001,  # キャッシュヒット時は非常に短い時間
                language="en",
                cache_hit=True,
            )
            mock_get_cache.return_value = cached_result
            
            result2 = await expansion_service.expand_query(sample_expansion_request)
            
            assert result2.cache_hit is True
            assert result2.expansion_time <= result1.expansion_time

    @pytest.mark.unit
    async def test_expansion_error_handling(
        self,
        basic_expansion_config: QueryExpansionConfig,
        sample_expansion_request: ExpansionRequest,
    ):
        """拡張エラーハンドリングテスト"""
        expansion_service = QueryExpansionService(config=basic_expansion_config)
        
        with patch.object(expansion_service, '_extract_query_terms', new_callable=AsyncMock) as mock_terms:
            # エラーをシミュレート
            mock_terms.side_effect = Exception("Term extraction failed")
            
            result = await expansion_service.expand_query(sample_expansion_request)
            
            assert result.success is False
            assert "Term extraction failed" in result.error_message
            assert result.expanded_terms == []

    @pytest.mark.unit
    async def test_empty_query_handling(
        self,
        basic_expansion_config: QueryExpansionConfig,
    ):
        """空クエリ処理テスト"""
        expansion_service = QueryExpansionService(config=basic_expansion_config)
        
        empty_request = ExpansionRequest(
            query="",
            language="en",
            max_expansions=5,
        )
        
        result = await expansion_service.expand_query(empty_request)
        
        assert result.success is False
        assert "empty" in result.error_message.lower()
        assert result.expanded_terms == []

    @pytest.mark.integration
    async def test_end_to_end_expansion(
        self,
        basic_expansion_config: QueryExpansionConfig,
        sample_expansion_request: ExpansionRequest,
    ):
        """End-to-End拡張テスト"""
        expansion_service = QueryExpansionService(config=basic_expansion_config)
        
        # 完全な拡張パイプラインのモック
        with patch.object(expansion_service, '_extract_query_terms', new_callable=AsyncMock) as mock_terms, \
             patch.object(expansion_service, '_expand_with_synonyms', new_callable=AsyncMock) as mock_synonyms, \
             patch.object(expansion_service, '_expand_with_semantics', new_callable=AsyncMock) as mock_semantics, \
             patch.object(expansion_service, '_expand_with_concepts', new_callable=AsyncMock) as mock_concepts:
            
            # Setup comprehensive mocks
            mock_terms.return_value = [
                QueryTerm(term="machine", pos_tag="NOUN", importance=0.8),
                QueryTerm(term="learning", pos_tag="NOUN", importance=0.9),
                QueryTerm(term="algorithms", pos_tag="NOUN", importance=0.7),
            ]
            
            mock_synonyms.return_value = {
                "machine": ["device", "apparatus"],
                "learning": ["training", "education"],
                "algorithms": ["methods", "techniques"],
            }
            
            mock_semantics.return_value = {
                "machine learning": [("artificial intelligence", 0.85), ("neural networks", 0.78)],
                "algorithms": [("optimization", 0.82), ("computation", 0.75)],
            }
            
            mock_concepts.return_value = {
                "machine_learning": {
                    "related_concepts": ["data_science", "pattern_recognition"],
                },
                "algorithms": {
                    "related_concepts": ["data_structures", "complexity"],
                },
            }
            
            # 拡張実行
            result = await expansion_service.expand_query(sample_expansion_request)
            
            # 結果検証
            assert result.success is True
            assert result.original_query == sample_expansion_request.query
            assert len(result.expanded_terms) > 0
            assert result.expansion_time > 0
            assert result.expanded_query != result.original_query
            
            # 拡張用語に期待する内容が含まれていることを確認
            expanded_terms_str = " ".join(result.expanded_terms)
            assert any(term in expanded_terms_str for term in ["device", "training", "methods"])


class TestSynonymExpander:
    """同義語拡張器のテストクラス"""

    @pytest.mark.unit
    def test_synonym_expander_initialization(self):
        """同義語拡張器初期化テスト"""
        config = QueryExpansionConfig(
            expansion_methods=[ExpansionMethod.SYNONYM],
            max_expansions=5,
        )
        
        expander = SynonymExpander(config)
        
        assert expander.config == config
        assert expander.max_expansions == 5

    @pytest.mark.unit
    async def test_english_synonym_expansion(self):
        """英語同義語拡張テスト"""
        config = QueryExpansionConfig()
        expander = SynonymExpander(config)
        
        with patch.object(expander, '_get_wordnet_synonyms') as mock_wordnet:
            mock_wordnet.return_value = ["device", "apparatus", "machine"]
            
            synonyms = await expander.expand_term("computer", "en")
            
            assert len(synonyms) > 0
            assert "device" in synonyms


class TestSemanticExpander:
    """セマンティック拡張器のテストクラス"""

    @pytest.mark.unit
    def test_semantic_expander_initialization(self):
        """セマンティック拡張器初期化テスト"""
        config = QueryExpansionConfig(
            expansion_methods=[ExpansionMethod.SEMANTIC],
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        expander = SemanticExpander(config)
        
        assert expander.config == config
        assert expander.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"

    @pytest.mark.unit
    async def test_semantic_similarity_calculation(self):
        """セマンティック類似度計算テスト"""
        config = QueryExpansionConfig()
        expander = SemanticExpander(config)
        
        with patch.object(expander, '_calculate_semantic_similarity') as mock_similarity:
            mock_similarity.return_value = 0.85
            
            similarity = expander._calculate_semantic_similarity("machine learning", "artificial intelligence")
            
            assert 0 <= similarity <= 1
            assert similarity == 0.85


class TestConceptualExpander:
    """概念拡張器のテストクラス"""

    @pytest.mark.unit
    def test_conceptual_expander_initialization(self):
        """概念拡張器初期化テスト"""
        config = QueryExpansionConfig(
            expansion_methods=[ExpansionMethod.CONCEPTUAL],
            enable_concept_expansion=True,
        )
        
        expander = ConceptualExpander(config)
        
        assert expander.config == config
        assert expander.enable_concept_expansion is True

    @pytest.mark.unit
    async def test_concept_hierarchy_expansion(self):
        """概念階層拡張テスト"""
        config = QueryExpansionConfig()
        expander = ConceptualExpander(config)
        
        with patch.object(expander, '_get_concept_hierarchy') as mock_hierarchy:
            mock_hierarchy.return_value = {
                "broader": ["computer_science", "artificial_intelligence"],
                "narrower": ["supervised_learning", "unsupervised_learning"],
                "related": ["data_mining", "pattern_recognition"],
            }
            
            concepts = await expander.expand_concept("machine_learning", "technology")
            
            assert "broader_concepts" in concepts
            assert "computer_science" in concepts["broader_concepts"]


class TestStatisticalExpander:
    """統計的拡張器のテストクラス"""

    @pytest.mark.unit
    def test_statistical_expander_initialization(self):
        """統計的拡張器初期化テスト"""
        config = QueryExpansionConfig(
            expansion_methods=[ExpansionMethod.STATISTICAL],
            enable_statistical_expansion=True,
            min_term_frequency=10,
        )
        
        expander = StatisticalExpander(config)
        
        assert expander.config == config
        assert expander.min_term_frequency == 10

    @pytest.mark.unit
    async def test_co_occurrence_expansion(self):
        """共起語拡張テスト"""
        config = QueryExpansionConfig()
        expander = StatisticalExpander(config)
        
        with patch.object(expander, '_get_co_occurrence_terms') as mock_cooccur:
            mock_cooccur.return_value = [
                ("neural", 0.75, 1250),
                ("deep", 0.68, 980),
                ("supervised", 0.62, 750),
            ]
            
            cooccur_terms = await expander.get_co_occurrence_terms("learning", min_frequency=100)
            
            assert len(cooccur_terms) == 3
            assert cooccur_terms[0][0] == "neural"
            assert cooccur_terms[0][2] == 1250  # frequency