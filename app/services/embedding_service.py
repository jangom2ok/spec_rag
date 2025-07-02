"""BGE-M3 Embedding Service

BGE-M3モデルを使用したテキスト埋め込みサービス。
Dense、Sparse、Multi-Vectorの3種類のベクトルを同時生成。
"""

import asyncio
import logging
import time
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, validator

try:
    import torch as torch_module
    from FlagEmbedding import FlagModel

    HAS_EMBEDDING_LIBS = True
except ImportError:
    # テスト環境やモジュール未インストール時のダミークラス
    HAS_EMBEDDING_LIBS = False
    torch_module = None

    class FlagModel:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, *args, **kwargs):
            # ダミー実装
            import numpy as np

            if isinstance(args[0], str):
                batch_size = 1
            else:
                batch_size = len(args[0])

            return {
                "dense_vecs": np.random.rand(batch_size, 1024).astype(np.float32),
                "lexical_weights": [
                    dict.fromkeys(range(10), 0.1) for _ in range(batch_size)
                ],
                "colbert_vecs": [
                    np.random.rand(5, 1024).astype(np.float32)
                    for _ in range(batch_size)
                ],
            }


logger = logging.getLogger(__name__)


def _detect_device(preferred_device: str) -> str:
    """最適なデバイスを検出"""
    if preferred_device == "auto":
        if HAS_EMBEDDING_LIBS and torch_module and torch_module.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    return preferred_device


class EmbeddingConfig(BaseModel):
    """埋め込みサービスの設定"""

    model_name: str = Field(default="BAAI/BGE-M3", description="BGE-M3モデル名")
    device: str = Field(default="auto", description="実行デバイス (auto, cpu, cuda)")
    batch_size: int = Field(default=16, description="バッチサイズ")
    max_length: int = Field(default=8192, description="最大トークン長")
    use_fp16: bool = Field(default=True, description="FP16使用フラグ")
    normalize_embeddings: bool = Field(default=True, description="ベクトル正規化フラグ")


class EmbeddingResult(BaseModel):
    """埋め込み処理の結果"""

    dense_vector: list[float] = Field(..., description="Dense vector (1024次元)")
    sparse_vector: dict[int, float] = Field(..., description="Sparse vector")
    multi_vector: np.ndarray | None = Field(
        None, description="Multi-vector (ColBERT style)"
    )
    processing_time: float = Field(..., description="処理時間(秒)")
    chunk_id: str | None = Field(None, description="チャンクID")
    document_id: str | None = Field(None, description="ドキュメントID")

    class Config:
        arbitrary_types_allowed = True

    @validator("dense_vector")
    def validate_dense_vector_dimension(self, v):
        """Dense vectorの次元数を検証"""
        if len(v) != 1024:
            raise ValueError("Dense vector must be 1024 dimensions")
        return v


class BatchEmbeddingRequest(BaseModel):
    """バッチ埋め込みリクエスト"""

    texts: list[str] = Field(..., description="埋め込み対象のテキストリスト")
    chunk_ids: list[str] = Field(..., description="チャンクIDリスト")
    document_ids: list[str] = Field(..., description="ドキュメントIDリスト")

    @validator("chunk_ids", "document_ids")
    def validate_list_lengths(self, v, values):
        """リストの長さを検証"""
        if "texts" in values and len(v) != len(values["texts"]):
            raise ValueError("All lists must have the same length")
        return v


class EmbeddingService:
    """BGE-M3埋め込みサービス"""

    def __init__(self, config: EmbeddingConfig):
        """サービスの初期化

        Args:
            config: 埋め込みサービスの設定
        """
        self.config = config
        self.model: FlagModel | None = None
        self.is_initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """モデルの初期化（遅延初期化）"""
        if self.is_initialized:
            return

        async with self._lock:
            if self.is_initialized:
                return

            try:
                # デバイスの自動検出
                device = _detect_device(self.config.device)
                logger.info(
                    f"Initializing BGE-M3 model: {self.config.model_name} on {device}"
                )

                # FlagModelの初期化
                self.model = FlagModel(
                    self.config.model_name, use_fp16=self.config.use_fp16, device=device
                )

                self.is_initialized = True
                logger.info(f"BGE-M3 model initialized successfully on {device}")

            except Exception as e:
                logger.error(f"Failed to initialize BGE-M3 model: {e}")
                raise RuntimeError(f"Model initialization failed: {e}") from e

    async def embed_text(self, text: str) -> EmbeddingResult:
        """単一テキストの埋め込み処理

        Args:
            text: 埋め込み対象のテキスト

        Returns:
            EmbeddingResult: 埋め込み結果

        Raises:
            ValueError: テキストが空の場合
            RuntimeError: モデル未初期化またはエラー発生時
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")

        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        start_time = time.time()

        try:
            # BGE-M3での埋め込み生成
            results = await self._encode_text(text)

            processing_time = time.time() - start_time

            return EmbeddingResult(
                dense_vector=results["dense_vector"],
                sparse_vector=results["sparse_vector"],
                multi_vector=results["multi_vector"],
                processing_time=processing_time,
                chunk_id=None,
                document_id=None,
            )

        except Exception as e:
            logger.error(f"Embedding failed for text: {e}")
            raise RuntimeError(f"Embedding failed: {e}") from e

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """バッチテキストの埋め込み処理

        Args:
            texts: 埋め込み対象のテキストリスト

        Returns:
            List[EmbeddingResult]: 埋め込み結果リスト
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        start_time = time.time()

        try:
            # バッチ処理での埋め込み生成
            batch_results = await self._encode_batch(texts)

            processing_time = time.time() - start_time
            per_text_time = processing_time / len(texts) if texts else 0

            results = []
            for i, _text in enumerate(texts):
                result = EmbeddingResult(
                    dense_vector=batch_results["dense_vectors"][i],
                    sparse_vector=batch_results["sparse_vectors"][i],
                    multi_vector=batch_results["multi_vectors"][i],
                    processing_time=per_text_time,
                    chunk_id=None,
                    document_id=None,
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            raise RuntimeError(f"Batch embedding failed: {e}") from e

    async def process_batch_request(
        self, request: BatchEmbeddingRequest
    ) -> list[EmbeddingResult]:
        """BatchEmbeddingRequestの処理

        Args:
            request: バッチ埋め込みリクエスト

        Returns:
            List[EmbeddingResult]: 埋め込み結果リスト（メタデータ付き）
        """
        results = await self.embed_batch(request.texts)

        # メタデータを追加
        for i, result in enumerate(results):
            result.chunk_id = request.chunk_ids[i]
            result.document_id = request.document_ids[i]

        return results

    async def _encode_text(self, text: str) -> dict[str, Any]:
        """単一テキストのエンコード（内部メソッド）"""
        loop = asyncio.get_event_loop()

        # CPUバウンドなタスクを別スレッドで実行
        def encode_sync():
            results = self.model.encode(
                text,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True,
                max_length=self.config.max_length,
            )

            return {
                "dense_vector": results["dense_vecs"][0].tolist(),
                "sparse_vector": results["lexical_weights"][0],
                "multi_vector": results["colbert_vecs"][0],
            }

        return await loop.run_in_executor(None, encode_sync)

    async def _encode_batch(self, texts: list[str]) -> dict[str, list[Any]]:
        """バッチテキストのエンコード（内部メソッド）"""
        loop = asyncio.get_event_loop()

        # バッチサイズに分割して処理
        batch_size = self.config.batch_size
        all_dense_vectors = []
        all_sparse_vectors = []
        all_multi_vectors = []

        def encode_batch_sync(batch_texts):
            results = self.model.encode(
                batch_texts,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True,
                max_length=self.config.max_length,
            )

            return {
                "dense_vecs": results["dense_vecs"],
                "sparse_vectors": results["lexical_weights"],
                "multi_vectors": results["colbert_vecs"],
            }

        # バッチごとに処理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_results = await loop.run_in_executor(
                None, encode_batch_sync, batch_texts
            )

            # Dense vectors
            all_dense_vectors.extend(
                [vec.tolist() for vec in batch_results["dense_vecs"]]
            )

            # Sparse vectors
            all_sparse_vectors.extend(batch_results["sparse_vectors"])

            # Multi vectors
            all_multi_vectors.extend(batch_results["multi_vectors"])

        return {
            "dense_vectors": all_dense_vectors,
            "sparse_vectors": all_sparse_vectors,
            "multi_vectors": all_multi_vectors,
        }

    async def get_model_info(self) -> dict[str, Any]:
        """モデル情報の取得"""
        return {
            "model_name": self.config.model_name,
            "device": self.config.device,
            "batch_size": self.config.batch_size,
            "max_length": self.config.max_length,
            "is_initialized": self.is_initialized,
            "use_fp16": self.config.use_fp16,
        }

    async def health_check(self) -> dict[str, Any]:
        """ヘルスチェック"""
        try:
            if not self.is_initialized:
                return {"status": "unhealthy", "reason": "Model not initialized"}

            # 簡単なテスト埋め込みを実行
            test_result = await self.embed_text("Health check test.")

            return {
                "status": "healthy",
                "model_info": await self.get_model_info(),
                "test_embedding_time": test_result.processing_time,
            }

        except Exception as e:
            return {"status": "unhealthy", "reason": f"Health check failed: {str(e)}"}
