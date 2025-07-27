
"""BGE-M3 Embedding Serviceのテスト"""

import asyncio
from unittest.mock import patch

import numpy as np
import pytest

from app.services.embedding_service import (
    BatchEmbeddingRequest,
    EmbeddingConfig,
    EmbeddingResult,
    EmbeddingService,
)


class TestEmbeddingResult:
    """EmbeddingResultクラスのテスト"""

    def test_embedding_result_creation(self):
        """EmbeddingResultの作成テスト"""
        dense_vector = [0.1, 0.2, 0.3] * 341 + [0.4]  # 1024次元
        sparse_vector = {0: 0.5, 100: 0.8, 500: 0.3}
        multi_vector = np.array([[0.1, 0.2], [0.3, 0.4]])

        result = EmbeddingResult(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            multi_vector=multi_vector,
            processing_time=0.15,
            chunk_id=None,
            document_id=None,
        )

        assert len(result.dense_vector) == 1024
        assert result.sparse_vector == sparse_vector
        assert result.multi_vector.shape == (2, 2)
        assert result.processing_time == 0.15

    def test_embedding_result_validation(self):
        """EmbeddingResultのバリデーションテスト"""
        with pytest.raises(ValueError, match="Dense vector must be 1024 dimensions"):
            EmbeddingResult(
                dense_vector=[0.1, 0.2],  # 不正な次元数
                sparse_vector={},
                multi_vector=np.array([]),
                processing_time=0.1,
                chunk_id=None,
                document_id=None,
            )


class TestEmbeddingConfig:
    """EmbeddingConfigクラスのテスト"""

    def test_config_creation(self):
        """設定オブジェクトの作成テスト"""
        config = EmbeddingConfig(
            model_name="BAAI/BGE-M3",
            device="cuda",
            batch_size=32,
            max_length=8192,
        )

        assert config.model_name == "BAAI/BGE-M3"
        assert config.device == "cuda"
        assert config.batch_size == 32
        assert config.max_length == 8192

    def test_config_defaults(self):
        """デフォルト設定のテスト"""
        config = EmbeddingConfig()

        assert config.model_name == "BAAI/BGE-M3"
        assert config.device == "auto"
        assert config.batch_size == 16
        assert config.max_length == 8192


class TestBatchEmbeddingRequest:
    """BatchEmbeddingRequestクラスのテスト"""

    def test_batch_request_creation(self):
        """バッチリクエストの作成テスト"""
        request = BatchEmbeddingRequest(
            texts=["text1", "text2", "text3"],
            chunk_ids=["chunk1", "chunk2", "chunk3"],
            document_ids=["doc1", "doc1", "doc2"],
        )

        assert len(request.texts) == 3
        assert len(request.chunk_ids) == 3
        assert len(request.document_ids) == 3

    def test_batch_request_validation(self):
        """バッチリクエストのバリデーションテスト"""
        with pytest.raises(ValueError, match="All lists must have the same length"):
            BatchEmbeddingRequest(
                texts=["text1", "text2"],
                chunk_ids=["chunk1"],  # 長さが異なる
                document_ids=["doc1", "doc2"],
            )


@pytest.fixture
def embedding_service():
    """EmbeddingServiceのフィクスチャ"""
    config = EmbeddingConfig(device="cpu")  # テスト用にCPUを使用
    service = EmbeddingService(config)
    return service


class TestEmbeddingService:
    """EmbeddingServiceクラスのテスト"""

    def test_service_initialization(self):
        """サービスの初期化テスト"""
        config = EmbeddingConfig()

        with patch("app.services.embedding_service.FlagModel") as mock_flag_model:
            service = EmbeddingService(config)

            assert service.config == config
            assert not service.is_initialized
            mock_flag_model.assert_not_called()  # 遅延初期化

    @pytest.mark.asyncio
    async def test_initialize_model(self, embedding_service):
        """モデルの初期化テスト"""
        await embedding_service.initialize()

        assert embedding_service.is_initialized
        assert embedding_service.model is not None

    @pytest.mark.asyncio
    async def test_embed_single_text(self, embedding_service):
        """単一テキストの埋め込みテスト"""
        await embedding_service.initialize()

        result = await embedding_service.embed_text("これはテストテキストです。")

        assert isinstance(result, EmbeddingResult)
        assert len(result.dense_vector) == 1024
        assert isinstance(result.sparse_vector, dict)
        assert result.multi_vector is not None
        assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_embed_batch_texts(self, embedding_service):
        """バッチテキストの埋め込みテスト"""
        await embedding_service.initialize()

        texts = [
            "これは最初のテキストです。",
            "これは2番目のテキストです。",
            "これは3番目のテキストです。",
        ]

        results = await embedding_service.embed_batch(texts)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, EmbeddingResult)
            assert len(result.dense_vector) == 1024
            assert isinstance(result.sparse_vector, dict)

    @pytest.mark.asyncio
    async def test_embed_batch_request(self, embedding_service):
        """BatchEmbeddingRequestの処理テスト"""
        await embedding_service.initialize()

        request = BatchEmbeddingRequest(
            texts=["text1", "text2"],
            chunk_ids=["chunk1", "chunk2"],
            document_ids=["doc1", "doc1"],
        )

        results = await embedding_service.process_batch_request(request)

        assert len(results) == 2
        for i, result in enumerate(results):
            assert result.chunk_id == request.chunk_ids[i]
            assert result.document_id == request.document_ids[i]

    @pytest.mark.asyncio
    async def test_model_not_initialized_error(self, embedding_service):
        """モデル未初期化時のエラーテスト"""
        # 初期化せずに埋め込み処理を実行
        with pytest.raises(RuntimeError, match="Model not initialized"):
            await embedding_service.embed_text("test")

    @pytest.mark.asyncio
    async def test_empty_text_handling(self, embedding_service):
        """空テキストの処理テスト"""
        await embedding_service.initialize()

        with pytest.raises(ValueError, match="Text cannot be empty"):
            await embedding_service.embed_text("")

    @pytest.mark.asyncio
    async def test_gpu_optimization(self):
        """GPU最適化の設定テスト"""
        config = EmbeddingConfig(device="cuda", batch_size=64)

        with patch("app.services.embedding_service.FlagModel") as mock_flag_model:
            service = EmbeddingService(config)
            await service.initialize()

            # FlagModelが正しい設定で呼ばれたことを確認
            mock_flag_model.assert_called_once_with(
                "BAAI/BGE-M3", use_fp16=True, device="cuda"
            )

    @pytest.mark.asyncio
    async def test_error_handling_during_embedding(self):
        """埋め込み処理中のエラーハンドリングテスト"""

        # Create a mock that raises an exception
        class ErrorFlagModel:
            def __init__(self, *args, **kwargs):
                pass

            def encode(self, *args, **kwargs):
                raise Exception("Model error")

        with patch("app.services.embedding_service.FlagModel", ErrorFlagModel):
            config = EmbeddingConfig(device="cpu")
            service = EmbeddingService(config)
            await service.initialize()

            with pytest.raises(RuntimeError, match="Embedding failed"):
                await service.embed_text("test text")

    @pytest.mark.asyncio
    async def test_max_length_handling(self, embedding_service):
        """最大長制限の処理テスト"""
        await embedding_service.initialize()

        # 8192トークンを超える長いテキスト
        long_text = "これは非常に長いテキストです。" * 1000

        result = await embedding_service.embed_text(long_text)

        # 正常に処理されることを確認
        assert isinstance(result, EmbeddingResult)
        assert len(result.dense_vector) == 1024


class TestEmbeddingServiceIntegration:
    """EmbeddingServiceの統合テスト"""

    @pytest.mark.asyncio
    async def test_full_pipeline_processing(self, embedding_service):
        """完全なパイプライン処理のテスト"""
        await embedding_service.initialize()

        # バッチリクエストの作成
        request = BatchEmbeddingRequest(
            texts=[
                "システム設計書の概要です。",
                "API仕様の詳細情報です。",
                "データベーススキーマの定義です。",
            ],
            chunk_ids=["chunk_001", "chunk_002", "chunk_003"],
            document_ids=["doc_system", "doc_api", "doc_schema"],
        )

        # バッチ処理の実行
        results = await embedding_service.process_batch_request(request)

        # 結果の検証
        assert len(results) == 3

        for i, result in enumerate(results):
            # 基本的な構造の確認
            assert result.chunk_id == request.chunk_ids[i]
            assert result.document_id == request.document_ids[i]
            assert len(result.dense_vector) == 1024
            assert isinstance(result.sparse_vector, dict)
            assert result.multi_vector is not None

            # ベクトルの有効性確認
            assert all(isinstance(x, int | float) for x in result.dense_vector)
            assert all(
                isinstance(k, int) and isinstance(v, int | float)
                for k, v in result.sparse_vector.items()
            )

    @pytest.mark.asyncio
    async def test_concurrent_embedding_requests(self, embedding_service):
        """並行埋め込みリクエストのテスト"""
        await embedding_service.initialize()

        # 複数の並行リクエストを作成
        texts = [f"テストテキスト{i}です。" for i in range(10)]

        # 並行実行
        tasks = [embedding_service.embed_text(text) for text in texts]
        results = await asyncio.gather(*tasks)

        # 結果の検証
        assert len(results) == 10
        for result in results:
            assert isinstance(result, EmbeddingResult)
            assert len(result.dense_vector) == 1024
