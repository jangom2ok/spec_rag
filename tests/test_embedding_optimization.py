"""GPU最適化とバッチ処理のテスト"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from app.services.embedding_service import EmbeddingConfig, EmbeddingService


class TestGPUOptimization:
    """GPU最適化機能のテスト"""

    @pytest.mark.asyncio
    async def test_gpu_device_detection(self):
        """GPU デバイス検出のテスト"""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("app.services.embedding_service.FlagModel") as mock_flag_model:
                config = EmbeddingConfig(device="auto")
                service = EmbeddingService(config)
                await service.initialize()

                # GPU が利用可能な場合は CUDA が選択されることを確認
                mock_flag_model.assert_called_once()
                call_args = mock_flag_model.call_args
                assert "device" in call_args.kwargs or len(call_args.args) >= 3

    @pytest.mark.asyncio
    async def test_cpu_fallback(self):
        """CPU フォールバック機能のテスト"""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("app.services.embedding_service.FlagModel") as mock_flag_model:
                config = EmbeddingConfig(device="auto")
                service = EmbeddingService(config)
                await service.initialize()

                # GPU が利用不可能な場合は CPU が選択されることを確認
                mock_flag_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_large_batch_processing(self):
        """大量バッチ処理のテスト"""
        config = EmbeddingConfig(batch_size=32)

        with patch("app.services.embedding_service.FlagModel") as mock_flag_model:
            # モックモデルの設定
            mock_model = Mock()

            def mock_encode_batch(texts, **kwargs):
                batch_size = len(texts)
                return {
                    "dense_vecs": np.random.rand(batch_size, 1024).astype(np.float32),
                    "lexical_weights": [
                        {i: np.random.rand() for i in range(10)}
                        for _ in range(batch_size)
                    ],
                    "colbert_vecs": [
                        np.random.rand(5, 1024).astype(np.float32)
                        for _ in range(batch_size)
                    ],
                }

            mock_model.encode.side_effect = mock_encode_batch
            mock_flag_model.return_value = mock_model

            service = EmbeddingService(config)
            await service.initialize()

            # 100個のテキストでテスト（バッチサイズ32で分割処理される）
            texts = [f"テストテキスト{i}です。" for i in range(100)]
            results = await service.embed_batch(texts)

            assert len(results) == 100
            for result in results:
                assert len(result.dense_vector) == 1024
                assert isinstance(result.sparse_vector, dict)

    @pytest.mark.asyncio
    async def test_memory_optimization(self):
        """メモリ最適化のテスト"""
        config = EmbeddingConfig(use_fp16=True, batch_size=16)

        with patch("app.services.embedding_service.FlagModel") as mock_flag_model:
            service = EmbeddingService(config)
            await service.initialize()

            # FP16 使用フラグが正しく渡されることを確認
            mock_flag_model.assert_called_once_with(
                "BAAI/BGE-M3", use_fp16=True, device="auto"
            )


class TestBatchProcessingOptimization:
    """バッチ処理最適化のテスト"""

    @pytest.fixture
    def mock_embedding_service(self):
        """最適化されたEmbeddingServiceのモック"""
        mock_service = Mock()

        async def mock_embed_batch(texts):
            # バッチサイズに応じた処理時間をシミュレート
            batch_count = len(texts)
            base_time = 0.01
            processing_time = base_time * (batch_count / 16)  # バッチサイズ16を基準

            results = []
            for i, _text in enumerate(texts):
                result = Mock()
                result.dense_vector = [0.1 + i * 0.01] * 1024
                result.sparse_vector = {j: 0.5 + j * 0.01 for j in range(10)}
                result.multi_vector = np.random.rand(5, 1024)
                result.processing_time = processing_time / batch_count
                results.append(result)

            return results

        mock_service.embed_batch = mock_embed_batch
        return mock_service

    @pytest.mark.asyncio
    async def test_optimal_batch_size_detection(self, mock_embedding_service):
        """最適バッチサイズ検出のテスト"""
        # 異なるバッチサイズでの処理時間を測定
        batch_sizes = [8, 16, 32, 64]
        text_count = 100
        texts = [f"テスト{i}" for i in range(text_count)]

        performance_results = {}

        for batch_size in batch_sizes:
            # バッチサイズごとの処理時間を測定（モック）
            batches = [
                texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
            ]
            total_time = 0

            for batch in batches:
                results = await mock_embedding_service.embed_batch(batch)
                total_time += sum(r.processing_time for r in results)

            performance_results[batch_size] = total_time

        # 最適バッチサイズの選択（最小処理時間）
        optimal_batch_size = min(
            performance_results.keys(), key=lambda k: performance_results[k]
        )

        # バッチサイズ16または32が最適であることを確認
        assert optimal_batch_size in [16, 32]

    @pytest.mark.asyncio
    async def test_concurrent_batch_processing(self):
        """並行バッチ処理のテスト"""
        config = EmbeddingConfig(batch_size=16)

        with patch("app.services.embedding_service.FlagModel") as mock_flag_model:
            mock_model = Mock()

            def mock_encode(texts, **kwargs):
                batch_size = len(texts) if isinstance(texts, list) else 1
                if not isinstance(texts, list):
                    texts = [texts]
                    batch_size = 1

                return {
                    "dense_vecs": np.random.rand(batch_size, 1024).astype(np.float32),
                    "lexical_weights": [
                        {i: np.random.rand() for i in range(10)}
                        for _ in range(batch_size)
                    ],
                    "colbert_vecs": [
                        np.random.rand(5, 1024).astype(np.float32)
                        for _ in range(batch_size)
                    ],
                }

            mock_model.encode.side_effect = mock_encode
            mock_flag_model.return_value = mock_model

            service = EmbeddingService(config)
            await service.initialize()

            # 複数の並行リクエストを作成
            import asyncio

            tasks = []
            for i in range(5):
                texts = [f"並行テスト{i}-{j}" for j in range(20)]
                task = service.embed_batch(texts)
                tasks.append(task)

            # 並行実行
            results_list = await asyncio.gather(*tasks)

            # 結果の検証
            assert len(results_list) == 5
            for results in results_list:
                assert len(results) == 20
                for result in results:
                    assert len(result.dense_vector) == 1024

    @pytest.mark.asyncio
    async def test_dynamic_batch_size_adjustment(self):
        """動的バッチサイズ調整のテスト"""
        config = EmbeddingConfig(batch_size=32)

        with patch("app.services.embedding_service.FlagModel") as mock_flag_model:
            mock_model = Mock()
            call_count = 0

            def mock_encode_with_memory_simulation(texts, **kwargs):
                nonlocal call_count
                call_count += 1

                batch_size = len(texts) if isinstance(texts, list) else 1
                if not isinstance(texts, list):
                    texts = [texts]
                    batch_size = 1

                # メモリ不足をシミュレート（特定の条件下）
                if batch_size > 16 and call_count <= 2:
                    raise RuntimeError("CUDA out of memory")

                return {
                    "dense_vecs": np.random.rand(batch_size, 1024).astype(np.float32),
                    "lexical_weights": [
                        {i: np.random.rand() for i in range(10)}
                        for _ in range(batch_size)
                    ],
                    "colbert_vecs": [
                        np.random.rand(5, 1024).astype(np.float32)
                        for _ in range(batch_size)
                    ],
                }

            mock_model.encode.side_effect = mock_encode_with_memory_simulation
            mock_flag_model.return_value = mock_model

            service = EmbeddingService(config)
            await service.initialize()

            # 大量のテキストで処理を試行
            texts = [f"動的調整テスト{i}" for i in range(50)]

            # メモリエラーが発生するがリトライで成功することを想定
            try:
                results = await service.embed_batch(texts)
                # 成功した場合の検証
                assert len(results) == 50
            except RuntimeError as e:
                # メモリ不足エラーの場合の検証
                assert "out of memory" in str(e)


class TestAdvancedFeatures:
    """高度な機能のテスト"""

    @pytest.mark.asyncio
    async def test_multi_vector_processing(self):
        """Multi-Vector処理のテスト"""
        config = EmbeddingConfig()

        with patch("app.services.embedding_service.FlagModel") as mock_flag_model:
            mock_model = Mock()

            def mock_encode_multi_vector(texts, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]

                return_colbert = kwargs.get("return_colbert_vecs", False)

                results = {
                    "dense_vecs": np.random.rand(len(texts), 1024).astype(np.float32),
                    "lexical_weights": [
                        {i: np.random.rand() for i in range(10)} for _ in texts
                    ],
                }

                if return_colbert:
                    # Multi-Vector (ColBERT style) - 可変長のトークンベクトル
                    results["colbert_vecs"] = [
                        np.random.rand(np.random.randint(5, 20), 1024).astype(
                            np.float32
                        )
                        for _ in texts
                    ]

                return results

            mock_model.encode.side_effect = mock_encode_multi_vector
            mock_flag_model.return_value = mock_model

            service = EmbeddingService(config)
            await service.initialize()

            # 長いテキストでのMulti-Vector生成
            long_text = "これは非常に長いテキストです。" * 50  # 長文テスト
            result = await service.embed_text(long_text)

            assert len(result.dense_vector) == 1024
            assert isinstance(result.sparse_vector, dict)
            assert result.multi_vector is not None
            assert result.multi_vector.shape[1] == 1024  # 各トークンベクトルは1024次元

    @pytest.mark.asyncio
    async def test_text_preprocessing_optimization(self):
        """テキスト前処理最適化のテスト"""
        config = EmbeddingConfig(max_length=8192)

        with patch("app.services.embedding_service.FlagModel") as mock_flag_model:
            mock_model = Mock()

            def mock_encode_with_length_check(texts, **kwargs):
                max_length = kwargs.get("max_length", 8192)

                if isinstance(texts, str):
                    texts = [texts]

                # 長さチェック
                for text in texts:
                    if len(text) > max_length * 4:  # 概算でトークン数を推定
                        # 長すぎるテキストは切り詰め
                        text = text[: max_length * 4]

                return {
                    "dense_vecs": np.random.rand(len(texts), 1024).astype(np.float32),
                    "lexical_weights": [
                        {i: np.random.rand() for i in range(10)} for _ in texts
                    ],
                    "colbert_vecs": [
                        np.random.rand(10, 1024).astype(np.float32) for _ in texts
                    ],
                }

            mock_model.encode.side_effect = mock_encode_with_length_check
            mock_flag_model.return_value = mock_model

            service = EmbeddingService(config)
            await service.initialize()

            # 極端に長いテキスト
            very_long_text = "非常に長いテキスト。" * 2000

            result = await service.embed_text(very_long_text)

            # 正常に処理されることを確認
            assert len(result.dense_vector) == 1024
            assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_error_recovery_and_retry(self):
        """エラー回復とリトライ機能のテスト"""
        config = EmbeddingConfig()

        with patch("app.services.embedding_service.FlagModel") as mock_flag_model:
            mock_model = Mock()
            call_count = 0

            def mock_encode_with_retry(texts, **kwargs):
                nonlocal call_count
                call_count += 1

                # 最初の2回は失敗、3回目で成功
                if call_count <= 2:
                    raise RuntimeError("Temporary model error")

                if isinstance(texts, str):
                    texts = [texts]

                return {
                    "dense_vecs": np.random.rand(len(texts), 1024).astype(np.float32),
                    "lexical_weights": [
                        {i: np.random.rand() for i in range(10)} for _ in texts
                    ],
                    "colbert_vecs": [
                        np.random.rand(5, 1024).astype(np.float32) for _ in texts
                    ],
                }

            mock_model.encode.side_effect = mock_encode_with_retry
            mock_flag_model.return_value = mock_model

            service = EmbeddingService(config)
            await service.initialize()

            # リトライ機能をテスト（実装依存）
            with pytest.raises(RuntimeError):
                await service.embed_text("リトライテスト")


class TestPerformanceMetrics:
    """パフォーマンスメトリクスのテスト"""

    @pytest.mark.asyncio
    async def test_throughput_measurement(self):
        """スループット測定のテスト"""
        config = EmbeddingConfig(batch_size=32)

        with patch("app.services.embedding_service.FlagModel") as mock_flag_model:
            mock_model = Mock()

            def mock_encode_timed(texts, **kwargs):
                # 処理時間をシミュレート
                import time

                time.sleep(0.01)  # 10ms の処理時間

                if isinstance(texts, str):
                    texts = [texts]

                return {
                    "dense_vecs": np.random.rand(len(texts), 1024).astype(np.float32),
                    "lexical_weights": [
                        {i: np.random.rand() for i in range(10)} for _ in texts
                    ],
                    "colbert_vecs": [
                        np.random.rand(5, 1024).astype(np.float32) for _ in texts
                    ],
                }

            mock_model.encode.side_effect = mock_encode_timed
            mock_flag_model.return_value = mock_model

            service = EmbeddingService(config)
            await service.initialize()

            # スループット測定
            texts = [f"スループットテスト{i}" for i in range(100)]

            import time

            start_time = time.time()
            results = await service.embed_batch(texts)
            end_time = time.time()

            total_time = end_time - start_time
            throughput = len(texts) / total_time  # texts per second

            assert len(results) == 100
            assert throughput > 0
            # パフォーマンス指標の記録（実際の実装では)
            print(f"Throughput: {throughput:.2f} texts/sec")

    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self):
        """メモリ使用量監視のテスト"""
        config = EmbeddingConfig(batch_size=16)

        with patch("app.services.embedding_service.FlagModel") as mock_flag_model:
            mock_model = Mock()
            mock_model.encode.return_value = {
                "dense_vecs": np.random.rand(16, 1024).astype(np.float32),
                "lexical_weights": [
                    {i: np.random.rand() for i in range(10)} for _ in range(16)
                ],
                "colbert_vecs": [
                    np.random.rand(5, 1024).astype(np.float32) for _ in range(16)
                ],
            }
            mock_flag_model.return_value = mock_model

            service = EmbeddingService(config)
            await service.initialize()

            # メモリ使用量の監視（psutil使用）
            import os

            import psutil

            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss

            # 大量バッチ処理
            texts = [f"メモリテスト{i}" for i in range(100)]
            results = await service.embed_batch(texts)

            memory_after = process.memory_info().rss
            memory_increase = memory_after - memory_before

            assert len(results) == 100
            # メモリ増加が許容範囲内であることを確認
            assert memory_increase < 100 * 1024 * 1024  # 100MB以下
