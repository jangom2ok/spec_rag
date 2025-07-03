"""Rerankerサービス

TDD実装：CrossEncoder/ColBERTベースの高精度再ランキング機能
- CrossEncoder: BERTベースのクエリ-ドキュメントペア分類
- ColBERT: 効率的なColBERTアーキテクチャ
- アンサンブル: 複数Rerankerの統合
"""

import asyncio
import hashlib
import logging
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


class RerankerType(str, Enum):
    """Rerankerタイプ"""
    
    CROSS_ENCODER = "cross_encoder"
    COLBERT = "colbert"
    ENSEMBLE = "ensemble"


@dataclass
class RerankerConfig:
    """Reranker設定"""
    
    reranker_type: RerankerType
    model_name: Optional[str] = None
    top_k: int = 10
    batch_size: int = 16
    max_sequence_length: int = 512
    enable_caching: bool = True
    cache_ttl: int = 3600
    timeout: float = 30.0
    
    # アンサンブル用設定
    ensemble_weights: Optional[List[float]] = None
    ensemble_models: Optional[List[str]] = None
    
    # デバイス設定
    device: str = "cpu"
    fp16: bool = False
    
    def __post_init__(self):
        """設定値のバリデーション"""
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be greater than 0")


@dataclass
class RerankRequest:
    """再ランキングリクエスト"""
    
    query: str
    documents: List[Dict[str, Any]]
    top_k: Optional[int] = None
    return_scores: bool = True
    return_explanations: bool = False


@dataclass
class RerankResult:
    """再ランキング結果"""
    
    success: bool
    documents: List[Dict[str, Any]]
    total_documents: int
    rerank_time: float
    query: str
    error_message: Optional[str] = None
    cache_hit: bool = False
    
    def get_summary(self) -> Dict[str, Any]:
        """再ランキング結果のサマリーを取得"""
        return {
            "success": self.success,
            "total_documents": self.total_documents,
            "returned_count": len(self.documents),
            "rerank_time": self.rerank_time,
            "cache_hit": self.cache_hit,
        }


class BaseReranker:
    """Rerankerベースクラス"""
    
    def __init__(self, config: RerankerConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    async def load_model(self) -> None:
        """モデル読み込み（オーバーライド必須）"""
        raise NotImplementedError
    
    async def score(self, query: str, texts: List[str]) -> List[float]:
        """テキストのスコアリング（オーバーライド必須）"""
        raise NotImplementedError


class CrossEncoderReranker(BaseReranker):
    """CrossEncoderベースのReranker"""
    
    def __init__(self, config: RerankerConfig):
        super().__init__(config)
        self.model_name = config.model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    async def load_model(self) -> None:
        """CrossEncoderモデル読み込み"""
        try:
            # 実際の実装では sentence-transformers.CrossEncoder を使用
            # ここではモック実装
            logger.info(f"Loading CrossEncoder model: {self.model_name}")
            self.model = MockCrossEncoderModel(self.model_name)
            logger.info("CrossEncoder model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model: {e}")
            raise
    
    async def score(self, query: str, texts: List[str]) -> List[float]:
        """CrossEncoderスコアリング"""
        if not self.model:
            await self.load_model()
        
        # クエリとテキストのペアを作成
        pairs = [[query, text] for text in texts]
        
        # バッチ処理でスコア計算
        scores = []
        for i in range(0, len(pairs), self.config.batch_size):
            batch = pairs[i:i + self.config.batch_size]
            batch_scores = self.model.predict(batch)
            scores.extend(batch_scores)
        
        return scores


class ColBERTReranker(BaseReranker):
    """ColBERTベースのReranker"""
    
    def __init__(self, config: RerankerConfig):
        super().__init__(config)
        self.model_name = config.model_name or "colbert-ir/colbertv2.0"
    
    async def load_model(self) -> None:
        """ColBERTモデル読み込み"""
        try:
            logger.info(f"Loading ColBERT model: {self.model_name}")
            self.model = MockColBERTModel(self.model_name)
            logger.info("ColBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ColBERT model: {e}")
            raise
    
    async def score(self, query: str, texts: List[str]) -> List[float]:
        """ColBERTスコアリング（MaxSim）"""
        if not self.model:
            await self.load_model()
        
        # クエリエンコーディング
        query_embeddings = self._encode_query(query)
        
        # ドキュメントエンコーディング
        doc_embeddings = self._encode_documents(texts)
        
        # MaxSimスコア計算
        scores = []
        for doc_emb in doc_embeddings:
            maxsim_score = self._compute_maxsim(query_embeddings, doc_emb)
            scores.append(maxsim_score)
        
        return scores
    
    def _encode_query(self, query: str) -> np.ndarray:
        """クエリエンコーディング"""
        # モック実装
        tokens = query.split()[:32]  # 最大32トークン
        return np.random.random((len(tokens), 128))
    
    def _encode_documents(self, texts: List[str]) -> List[np.ndarray]:
        """ドキュメントエンコーディング"""
        # モック実装
        doc_embeddings = []
        for text in texts:
            tokens = text.split()[:180]  # 最大180トークン
            doc_embeddings.append(np.random.random((len(tokens), 128)))
        return doc_embeddings
    
    def _compute_maxsim(self, query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
        """MaxSimスコア計算"""
        # コサイン類似度の最大値を計算
        similarities = np.dot(query_emb, doc_emb.T)
        maxsim = np.mean(np.max(similarities, axis=1))
        return float(maxsim)


class EnsembleReranker(BaseReranker):
    """アンサンブルReranker"""
    
    def __init__(self, config: RerankerConfig):
        super().__init__(config)
        self.ensemble_weights = config.ensemble_weights or [0.5, 0.5]
        self.rerankers = []
    
    async def load_model(self) -> None:
        """アンサンブルモデル読み込み"""
        # CrossEncoderとColBERTを組み合わせ
        ce_config = RerankerConfig(
            reranker_type=RerankerType.CROSS_ENCODER,
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            batch_size=self.config.batch_size,
        )
        colbert_config = RerankerConfig(
            reranker_type=RerankerType.COLBERT,
            model_name="colbert-ir/colbertv2.0", 
            batch_size=self.config.batch_size,
        )
        
        self.rerankers = [
            CrossEncoderReranker(ce_config),
            ColBERTReranker(colbert_config),
        ]
        
        # 各Rerankerを初期化
        for reranker in self.rerankers:
            await reranker.load_model()
    
    async def score(self, query: str, texts: List[str]) -> List[float]:
        """アンサンブルスコアリング"""
        if not self.rerankers:
            await self.load_model()
        
        # 各Rerankerからスコアを取得
        all_scores = []
        for reranker in self.rerankers:
            scores = await reranker.score(query, texts)
            all_scores.append(scores)
        
        # 重み付き平均でアンサンブル
        ensemble_scores = []
        for i in range(len(texts)):
            weighted_score = sum(
                scores[i] * weight 
                for scores, weight in zip(all_scores, self.ensemble_weights)
            )
            ensemble_scores.append(weighted_score)
        
        return ensemble_scores


class RerankerService:
    """Rerankerメインサービス"""
    
    def __init__(self, config: RerankerConfig):
        self.config = config
        self.reranker = self._create_reranker()
        self.cache = {}  # 簡易キャッシュ実装
    
    def _create_reranker(self) -> BaseReranker:
        """Rerankerインスタンス作成"""
        if self.config.reranker_type == RerankerType.CROSS_ENCODER:
            return CrossEncoderReranker(self.config)
        elif self.config.reranker_type == RerankerType.COLBERT:
            return ColBERTReranker(self.config)
        elif self.config.reranker_type == RerankerType.ENSEMBLE:
            return EnsembleReranker(self.config)
        else:
            raise ValueError(f"Unsupported reranker type: {self.config.reranker_type}")
    
    async def rerank(self, request: RerankRequest) -> RerankResult:
        """再ランキング実行"""
        start_time = datetime.now()
        
        try:
            # 入力バリデーション
            if not request.documents:
                return RerankResult(
                    success=True,
                    documents=[],
                    total_documents=0,
                    rerank_time=0.0,
                    query=request.query,
                )
            
            # キャッシュチェック
            if self.config.enable_caching:
                cache_key = self._get_cache_key(request)
                cached_result = await self._get_from_cache(cache_key)
                if cached_result:
                    return cached_result
            
            # top_k設定
            top_k = request.top_k or self.config.top_k
            
            # ドキュメントテキスト抽出
            texts = []
            for doc in request.documents:
                title = doc.get("title", "")
                content = doc.get("content", "")
                combined_text = f"{title} {content}".strip()
                texts.append(combined_text)
            
            # タイムアウト処理付きでスコアリング実行
            try:
                scores = await asyncio.wait_for(
                    self._get_reranker_scores(request.query, texts),
                    timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                return RerankResult(
                    success=False,
                    documents=[],
                    total_documents=len(request.documents),
                    rerank_time=(datetime.now() - start_time).total_seconds(),
                    query=request.query,
                    error_message="Reranking timeout",
                )
            
            # ドキュメントにスコアを追加
            scored_docs = []
            for doc, score in zip(request.documents, scores):
                doc_copy = doc.copy()
                if request.return_scores:
                    doc_copy["rerank_score"] = score
                
                # 説明を追加
                if request.return_explanations:
                    explanation = await self._generate_explanation(
                        request.query, doc, score
                    )
                    doc_copy["rerank_explanation"] = explanation
                
                scored_docs.append(doc_copy)
            
            # スコア順でソート
            scored_docs.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            # top_kで切り取り
            result_docs = scored_docs[:top_k]
            
            end_time = datetime.now()
            rerank_time = (end_time - start_time).total_seconds()
            
            result = RerankResult(
                success=True,
                documents=result_docs,
                total_documents=len(request.documents),
                rerank_time=rerank_time,
                query=request.query,
            )
            
            # キャッシュに保存
            if self.config.enable_caching:
                await self._set_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            end_time = datetime.now()
            rerank_time = (end_time - start_time).total_seconds()
            
            return RerankResult(
                success=False,
                documents=[],
                total_documents=len(request.documents),
                rerank_time=rerank_time,
                query=request.query,
                error_message=str(e),
            )
    
    async def _get_reranker_scores(self, query: str, texts: List[str]) -> List[float]:
        """Rerankerスコア取得の統一インターフェース"""
        if self.config.reranker_type == RerankerType.CROSS_ENCODER:
            return await self._get_cross_encoder_scores(query, texts)
        elif self.config.reranker_type == RerankerType.COLBERT:
            return await self._get_colbert_scores(query, texts)
        elif self.config.reranker_type == RerankerType.ENSEMBLE:
            return await self._get_ensemble_scores(query, texts)
        else:
            raise ValueError(f"Unknown reranker type: {self.config.reranker_type}")
    
    async def _get_cross_encoder_scores(self, query: str, texts: List[str]) -> List[float]:
        """CrossEncoderスコア取得"""
        return await self.reranker.score(query, texts)
    
    async def _get_colbert_scores(self, query: str, texts: List[str]) -> List[float]:
        """ColBERTスコア取得"""
        return await self.reranker.score(query, texts)
    
    async def _get_ensemble_scores(self, query: str, texts: List[str]) -> List[float]:
        """アンサンブルスコア取得"""
        return await self.reranker.score(query, texts)
    
    async def _process_batch(self, query: str, texts: List[str]) -> List[float]:
        """バッチ処理"""
        return await self.reranker.score(query, texts)
    
    def _get_cache_key(self, request: RerankRequest) -> str:
        """キャッシュキー生成"""
        # クエリとドキュメントIDからハッシュ生成
        content = {
            "query": request.query,
            "doc_ids": [doc.get("id", "") for doc in request.documents],
            "top_k": request.top_k or self.config.top_k,
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()
    
    async def _get_from_cache(self, cache_key: str) -> Optional[RerankResult]:
        """キャッシュから取得"""
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.config.cache_ttl):
                # RerankResultオブジェクトを正しくコピー
                result = RerankResult(
                    success=cached_data.success,
                    documents=cached_data.documents.copy(),
                    total_documents=cached_data.total_documents,
                    rerank_time=cached_data.rerank_time,
                    query=cached_data.query,
                    error_message=cached_data.error_message,
                    cache_hit=True,
                )
                return result
            else:
                # 期限切れキャッシュを削除
                del self.cache[cache_key]
        return None
    
    async def _set_cache(self, cache_key: str, result: RerankResult) -> None:
        """キャッシュに保存"""
        self.cache[cache_key] = (result, datetime.now())
    
    async def _generate_explanation(
        self, 
        query: str, 
        document: Dict[str, Any], 
        score: float
    ) -> Dict[str, Any]:
        """再ランキング説明生成"""
        explanation = await self._generate_explanations([document])
        return explanation[0] if explanation else {}
    
    async def _generate_explanations(
        self, 
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """複数ドキュメントの説明生成"""
        explanations = []
        for doc in documents:
            # 簡易的な説明生成（実際はより詳細な分析）
            explanation = {
                "relevance_factors": ["content_match", "semantic_similarity"],
                "confidence": doc.get("rerank_score", 0.5),
                "key_phrases": ["machine learning", "algorithms"],
            }
            explanations.append(explanation)
        return explanations


# モッククラス（テスト用）
class MockCrossEncoderModel:
    """モックCrossEncoderモデル"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def predict(self, pairs: List[List[str]]) -> List[float]:
        """予測（モック実装）"""
        # クエリとテキストの単語重複率ベースの簡易スコア
        scores = []
        for query, text in pairs:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            overlap = len(query_words.intersection(text_words))
            total = len(query_words.union(text_words))
            score = overlap / total if total > 0 else 0
            # より現実的なスコア範囲に調整
            adjusted_score = 0.5 + (score * 0.4)  # 0.5-0.9の範囲
            scores.append(adjusted_score)
        return scores


class MockColBERTModel:
    """モックColBERTモデル"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def encode(self, texts: List[str]) -> List[np.ndarray]:
        """エンコーディング（モック実装）"""
        embeddings = []
        for text in texts:
            # テキスト長に基づいたランダムな埋め込み
            tokens = len(text.split())
            embedding = np.random.random((min(tokens, 180), 128))
            embeddings.append(embedding)
        return embeddings