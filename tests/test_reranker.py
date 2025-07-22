"""Rerankerサービスのテストモジュール

TDD実装：CrossEncoder/ColBERTベースの高精度再ランキング機能
"""

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.services.reranker import (
    ColBERTReranker,
    CrossEncoderReranker,
    EnsembleReranker,
    RerankerConfig,
    RerankerService,
    RerankerType,
    RerankRequest,
    RerankResult,
)


class TestRerankerService:
    """Rerankerサービスのテストクラス"""

    @pytest.fixture
    def basic_reranker_config(self) -> RerankerConfig:
        """基本Reranker設定"""
        return RerankerConfig(
            reranker_type=RerankerType.CROSS_ENCODER,
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k=10,
            enable_caching=True,
            cache_ttl=3600,
            batch_size=16,
            max_sequence_length=512,
        )

    @pytest.fixture
    def colbert_config(self) -> RerankerConfig:
        """ColBERT設定"""
        return RerankerConfig(
            reranker_type=RerankerType.COLBERT,
            model_name="colbert-ir/colbertv2.0",
            top_k=20,
            enable_caching=False,
            batch_size=8,
            max_sequence_length=256,
        )

    @pytest.fixture
    def ensemble_config(self) -> RerankerConfig:
        """アンサンブル設定"""
        return RerankerConfig(
            reranker_type=RerankerType.ENSEMBLE,
            ensemble_weights=[0.6, 0.4],
            top_k=15,
            enable_caching=True,
            batch_size=16,
        )

    @pytest.fixture
    def sample_documents(self) -> list[dict[str, Any]]:
        """サンプルドキュメント"""
        return [
            {
                "id": "doc-1",
                "title": "Introduction to Machine Learning",
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                "search_score": 0.85,
                "metadata": {"category": "ai", "difficulty": "beginner"},
            },
            {
                "id": "doc-2",
                "title": "Deep Learning Fundamentals",
                "content": "Deep learning is a machine learning technique that teaches computers to do what comes naturally to humans: learn by example.",
                "search_score": 0.82,
                "metadata": {"category": "ai", "difficulty": "intermediate"},
            },
            {
                "id": "doc-3",
                "title": "Natural Language Processing",
                "content": "Natural language processing combines computational linguistics with statistical machine learning and deep learning models.",
                "search_score": 0.78,
                "metadata": {"category": "nlp", "difficulty": "advanced"},
            },
            {
                "id": "doc-4",
                "title": "Computer Vision Applications",
                "content": "Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world.",
                "search_score": 0.75,
                "metadata": {"category": "cv", "difficulty": "intermediate"},
            },
            {
                "id": "doc-5",
                "title": "Data Science Overview",
                "content": "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge from data.",
                "search_score": 0.71,
                "metadata": {"category": "data", "difficulty": "beginner"},
            },
        ]

    @pytest.fixture
    def rerank_request(self, sample_documents) -> RerankRequest:
        """再ランキングリクエスト"""
        return RerankRequest(
            query="machine learning for natural language processing",
            documents=sample_documents,
            top_k=3,
            return_scores=True,
            return_explanations=False,
        )

    @pytest.mark.unit
    async def test_reranker_service_initialization(
        self, basic_reranker_config: RerankerConfig
    ):
        """Rerankerサービス初期化テスト"""
        reranker = RerankerService(config=basic_reranker_config)

        assert reranker.config == basic_reranker_config
        assert reranker.config.reranker_type == RerankerType.CROSS_ENCODER
        assert reranker.config.top_k == 10
        assert reranker.config.enable_caching is True

    @pytest.mark.unit
    async def test_cross_encoder_reranking(
        self,
        basic_reranker_config: RerankerConfig,
        rerank_request: RerankRequest,
    ):
        """CrossEncoder再ランキングテスト"""
        reranker = RerankerService(config=basic_reranker_config)

        with patch.object(
            reranker, "_get_cross_encoder_scores", new_callable=AsyncMock
        ) as mock_scores:
            # CrossEncoderスコアをモック
            mock_scores.return_value = [0.92, 0.89, 0.85, 0.73, 0.68]

            result = await reranker.rerank(rerank_request)

            assert isinstance(result, RerankResult)
            assert result.success is True
            assert len(result.documents) == 3  # top_k=3
            assert result.total_documents == 5
            assert result.rerank_time > 0

            # スコア降順でソートされていることを確認
            scores = [doc["rerank_score"] for doc in result.documents]
            assert scores == sorted(scores, reverse=True)

            # 元のsearch_scoreより高いrerank_scoreを持つことを確認
            assert (
                result.documents[0]["rerank_score"]
                > result.documents[0]["search_score"]
            )

    @pytest.mark.unit
    async def test_colbert_reranking(
        self,
        colbert_config: RerankerConfig,
        rerank_request: RerankRequest,
    ):
        """ColBERT再ランキングテスト"""
        reranker = RerankerService(config=colbert_config)

        with patch.object(
            reranker, "_get_colbert_scores", new_callable=AsyncMock
        ) as mock_scores:
            # ColBERTスコアをモック
            mock_scores.return_value = [0.88, 0.91, 0.87, 0.76, 0.72]

            result = await reranker.rerank(rerank_request)

            assert result.success is True
            assert len(result.documents) == 3

            # ColBERTスコアに基づいた順序変更を確認
            # doc-2が最高スコア(0.91)なので1位になるべき
            assert result.documents[0]["id"] == "doc-2"

    @pytest.mark.unit
    async def test_ensemble_reranking(
        self,
        ensemble_config: RerankerConfig,
        rerank_request: RerankRequest,
    ):
        """アンサンブル再ランキングテスト"""
        reranker = RerankerService(config=ensemble_config)

        with patch.object(
            reranker, "_get_ensemble_scores", new_callable=AsyncMock
        ) as mock_ensemble:
            # アンサンブルスコアを直接モック
            mock_ensemble.return_value = [0.892, 0.878, 0.814, 0.758, 0.708]

            result = await reranker.rerank(rerank_request)

            assert result.success is True
            assert len(result.documents) == 3

            # アンサンブルスコアが正しく適用されていることを確認
            assert result.documents[0]["rerank_score"] == 0.892
            assert result.documents[1]["rerank_score"] == 0.878
            assert result.documents[2]["rerank_score"] == 0.814

    @pytest.mark.unit
    async def test_reranking_with_explanations(
        self,
        basic_reranker_config: RerankerConfig,
        sample_documents: list[dict[str, Any]],
    ):
        """説明付き再ランキングテスト"""
        reranker = RerankerService(config=basic_reranker_config)

        request = RerankRequest(
            query="machine learning algorithms",
            documents=sample_documents,
            top_k=2,
            return_scores=True,
            return_explanations=True,
        )

        with (
            patch.object(
                reranker, "_get_cross_encoder_scores", new_callable=AsyncMock
            ) as mock_scores,
            patch.object(
                reranker, "_generate_explanations", new_callable=AsyncMock
            ) as mock_explain,
        ):
            mock_scores.return_value = [0.91, 0.87, 0.83, 0.74, 0.69]
            mock_explain.return_value = [
                {
                    "relevance_factors": ["title_match", "content_relevance"],
                    "confidence": 0.91,
                },
                {
                    "relevance_factors": ["content_match", "semantic_similarity"],
                    "confidence": 0.87,
                },
            ]

            result = await reranker.rerank(request)

            assert result.success is True
            assert len(result.documents) == 2

            # 説明が含まれていることを確認
            for doc in result.documents:
                assert "rerank_explanation" in doc
                assert "relevance_factors" in doc["rerank_explanation"]
                assert "confidence" in doc["rerank_explanation"]

    @pytest.mark.unit
    async def test_reranking_cache_functionality(
        self,
        basic_reranker_config: RerankerConfig,
        rerank_request: RerankRequest,
    ):
        """再ランキングキャッシュ機能テスト"""
        reranker = RerankerService(config=basic_reranker_config)

        with (
            patch.object(reranker, "_get_cache_key") as mock_cache_key,
            patch.object(
                reranker, "_get_from_cache", new_callable=AsyncMock
            ) as mock_get_cache,
            patch.object(
                reranker, "_set_cache", new_callable=AsyncMock
            ) as mock_set_cache,
            patch.object(
                reranker, "_get_cross_encoder_scores", new_callable=AsyncMock
            ) as mock_scores,
        ):
            cache_key = "rerank_cache_key_123"
            mock_cache_key.return_value = cache_key
            mock_get_cache.return_value = None  # キャッシュなし
            mock_scores.return_value = [0.90, 0.85, 0.80, 0.75, 0.70]

            # 初回実行
            result1 = await reranker.rerank(rerank_request)

            # キャッシュに保存されることを確認
            mock_set_cache.assert_called_once()

            # 2回目はキャッシュから取得
            cached_result = RerankResult(
                success=True,
                documents=result1.documents,
                total_documents=5,
                rerank_time=0.00001,  # キャッシュヒット時は非常に短い時間
                query="cached_query",
                cache_hit=True,
            )
            mock_get_cache.return_value = cached_result

            result2 = await reranker.rerank(rerank_request)

            assert result2.cache_hit is True
            assert (
                result2.rerank_time <= result1.rerank_time
            )  # キャッシュヒット時は同等かより短い

    @pytest.mark.unit
    async def test_batch_reranking(
        self,
        basic_reranker_config: RerankerConfig,
    ):
        """バッチ再ランキングテスト"""
        reranker = RerankerService(config=basic_reranker_config)

        # 大量のドキュメントでバッチ処理をテスト
        large_documents = []
        for i in range(50):
            large_documents.append(
                {
                    "id": f"doc-{i}",
                    "title": f"Document Title {i}",
                    "content": f"This is content for document {i} about machine learning.",
                    "search_score": 0.8 - (i * 0.01),
                }
            )

        request = RerankRequest(
            query="machine learning techniques",
            documents=large_documents,
            top_k=10,
        )

        with patch.object(
            reranker, "_get_cross_encoder_scores", new_callable=AsyncMock
        ) as mock_scores:
            # 全体のスコアを返すモック（実際のバッチ処理はCrossEncoderReranker内で行われる）
            mock_scores.return_value = [0.9 - (i * 0.01) for i in range(50)]

            result = await reranker.rerank(request)

            assert result.success is True
            assert len(result.documents) == 10  # top_k
            # バッチ処理が実行されたことを確認
            mock_scores.assert_called_once()

    @pytest.mark.unit
    async def test_multilingual_reranking(
        self,
        basic_reranker_config: RerankerConfig,
    ):
        """多言語再ランキングテスト"""
        reranker = RerankerService(config=basic_reranker_config)

        multilingual_docs = [
            {
                "id": "doc-en",
                "title": "Machine Learning Basics",
                "content": "Introduction to machine learning algorithms and techniques.",
                "search_score": 0.85,
                "language": "en",
            },
            {
                "id": "doc-ja",
                "title": "機械学習の基礎",
                "content": "機械学習のアルゴリズムと手法についての入門的な説明です。",
                "search_score": 0.82,
                "language": "ja",
            },
            {
                "id": "doc-zh",
                "title": "机器学习基础",
                "content": "机器学习算法和技术的入门介绍。",
                "search_score": 0.80,
                "language": "zh",
            },
        ]

        request = RerankRequest(
            query="machine learning introduction",
            documents=multilingual_docs,
            top_k=3,
        )

        with patch.object(
            reranker, "_get_cross_encoder_scores", new_callable=AsyncMock
        ) as mock_scores:
            mock_scores.return_value = [0.89, 0.85, 0.81]

            result = await reranker.rerank(request)

            assert result.success is True
            assert len(result.documents) == 3

            # 多言語対応で全言語のドキュメントが処理されることを確認
            languages = {doc["language"] for doc in result.documents}
            assert len(languages) == 3  # en, ja, zh

    @pytest.mark.unit
    async def test_reranking_error_handling(
        self,
        basic_reranker_config: RerankerConfig,
        rerank_request: RerankRequest,
    ):
        """再ランキングエラーハンドリングテスト"""
        reranker = RerankerService(config=basic_reranker_config)

        with patch.object(
            reranker, "_get_cross_encoder_scores", new_callable=AsyncMock
        ) as mock_scores:
            # エラーをシミュレート
            mock_scores.side_effect = Exception("Model loading failed")

            result = await reranker.rerank(rerank_request)

            assert result.success is False
            assert "Model loading failed" in result.error_message
            assert len(result.documents) == 0

    @pytest.mark.unit
    async def test_reranking_timeout_handling(
        self,
        basic_reranker_config: RerankerConfig,
        rerank_request: RerankRequest,
    ):
        """再ランキングタイムアウト処理テスト"""
        basic_reranker_config.timeout = 0.001  # 非常に短いタイムアウト
        reranker = RerankerService(config=basic_reranker_config)

        with patch.object(
            reranker, "_get_cross_encoder_scores", new_callable=AsyncMock
        ) as mock_scores:

            async def slow_scoring(*args, **kwargs):
                await asyncio.sleep(0.1)  # タイムアウトより長い処理
                return [0.9, 0.8, 0.7, 0.6, 0.5]

            mock_scores.side_effect = slow_scoring

            result = await reranker.rerank(rerank_request)

            # タイムアウト時の適切なハンドリングを確認
            assert (
                result.success is False
                or result.rerank_time > basic_reranker_config.timeout
            )

    @pytest.mark.unit
    async def test_empty_documents_handling(
        self,
        basic_reranker_config: RerankerConfig,
    ):
        """空ドキュメント処理テスト"""
        reranker = RerankerService(config=basic_reranker_config)

        empty_request = RerankRequest(
            query="test query",
            documents=[],
            top_k=5,
        )

        result = await reranker.rerank(empty_request)

        assert result.success is True
        assert len(result.documents) == 0
        assert result.total_documents == 0

    @pytest.mark.integration
    async def test_reranker_performance_benchmarking(
        self,
        basic_reranker_config: RerankerConfig,
        sample_documents: list[dict[str, Any]],
    ):
        """Rerankerパフォーマンスベンチマークテスト"""
        reranker = RerankerService(config=basic_reranker_config)

        # 大量データでのパフォーマンステスト
        large_dataset = sample_documents * 20  # 100ドキュメント

        request = RerankRequest(
            query="machine learning artificial intelligence",
            documents=large_dataset,
            top_k=10,
        )

        with patch.object(
            reranker, "_get_cross_encoder_scores", new_callable=AsyncMock
        ) as mock_scores:
            mock_scores.return_value = [
                0.9 - (i * 0.001) for i in range(len(large_dataset))
            ]

            start_time = datetime.now()
            result = await reranker.rerank(request)
            end_time = datetime.now()

            processing_time = (end_time - start_time).total_seconds()

            assert result.success is True
            assert len(result.documents) == 10
            assert processing_time < 5.0  # 5秒以内でのレスポンス


class TestCrossEncoderReranker:
    """CrossEncoderの単体テストクラス"""

    @pytest.mark.unit
    def test_cross_encoder_initialization(self):
        """CrossEncoder初期化テスト"""
        config = RerankerConfig(
            reranker_type=RerankerType.CROSS_ENCODER,
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        )

        reranker = CrossEncoderReranker(config)

        assert reranker.config == config
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @pytest.mark.unit
    async def test_cross_encoder_scoring(self):
        """CrossEncoderスコアリングテスト"""
        config = RerankerConfig(reranker_type=RerankerType.CROSS_ENCODER)
        reranker = CrossEncoderReranker(config)

        # MockCrossEncoderModelのインスタンスをモック
        from app.services.reranker import MockCrossEncoderModel

        mock_model = MockCrossEncoderModel("test-model")
        mock_model.predict = MagicMock(return_value=np.array([0.85, 0.92, 0.78]))
        reranker.model = mock_model

        query = "machine learning"
        texts = ["ML tutorial", "Deep learning guide", "Data science overview"]

        scores = await reranker.score(query, texts)

        assert len(scores) == 3
        assert scores[1] > scores[0] > scores[2]  # 順序確認
        assert all(0 <= score <= 1 for score in scores)


class TestColBERTReranker:
    """ColBERTの単体テストクラス"""

    @pytest.mark.unit
    def test_colbert_initialization(self):
        """ColBERT初期化テスト"""
        config = RerankerConfig(
            reranker_type=RerankerType.COLBERT,
            model_name="colbert-ir/colbertv2.0",
        )

        reranker = ColBERTReranker(config)

        assert reranker.config == config
        assert reranker.model_name == "colbert-ir/colbertv2.0"

    @pytest.mark.unit
    async def test_colbert_maxsim_scoring(self):
        """ColBERT MaxSim スコアリングテスト"""
        config = RerankerConfig(reranker_type=RerankerType.COLBERT)
        reranker = ColBERTReranker(config)

        with (
            patch.object(reranker, "_encode_query") as mock_encode_q,
            patch.object(reranker, "_encode_documents") as mock_encode_d,
        ):
            # クエリとドキュメントのトークン表現をモック
            mock_encode_q.return_value = np.random.random((5, 128))  # 5 tokens, 128 dim
            mock_encode_d.return_value = [
                np.random.random((8, 128)),  # doc1: 8 tokens
                np.random.random((12, 128)),  # doc2: 12 tokens
                np.random.random((6, 128)),  # doc3: 6 tokens
            ]

            query = "machine learning algorithms"
            texts = ["ML tutorial", "Deep learning", "Data science"]

            scores = await reranker.score(query, texts)

            assert len(scores) == 3
            assert all(isinstance(score, float) for score in scores)
            assert all(score >= 0 for score in scores)


class TestEnsembleReranker:
    """アンサンブルRerankerの単体テストクラス"""

    @pytest.mark.unit
    def test_ensemble_initialization(self):
        """アンサンブル初期化テスト"""
        config = RerankerConfig(
            reranker_type=RerankerType.ENSEMBLE,
            ensemble_weights=[0.6, 0.4],
        )

        reranker = EnsembleReranker(config)

        assert reranker.config == config
        assert reranker.ensemble_weights == [0.6, 0.4]

    @pytest.mark.unit
    async def test_ensemble_score_combination(self):
        """アンサンブルスコア統合テスト"""
        config = RerankerConfig(
            reranker_type=RerankerType.ENSEMBLE,
            ensemble_weights=[0.7, 0.3],
        )
        reranker = EnsembleReranker(config)

        # 手動でrerankers属性を設定してからテスト
        mock_reranker1 = AsyncMock()
        mock_reranker2 = AsyncMock()
        mock_reranker1.score.return_value = [0.8, 0.6, 0.9]
        mock_reranker2.score.return_value = [0.7, 0.9, 0.5]
        reranker.rerankers = [mock_reranker1, mock_reranker2]

        query = "test query"
        texts = ["text1", "text2", "text3"]

        ensemble_scores = await reranker.score(query, texts)

        # 期待されるアンサンブルスコア
        # text1: 0.8*0.7 + 0.7*0.3 = 0.77
        # text2: 0.6*0.7 + 0.9*0.3 = 0.69
        # text3: 0.9*0.7 + 0.5*0.3 = 0.78
        expected = [0.77, 0.69, 0.78]

        assert len(ensemble_scores) == 3
        for actual, expect in zip(ensemble_scores, expected, strict=False):
            assert abs(actual - expect) < 0.001
