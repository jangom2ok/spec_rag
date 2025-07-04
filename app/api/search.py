"""検索API"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from app.core.auth import validate_api_key
from app.services.hybrid_search_engine import (
    HybridSearchEngine,
    RankingAlgorithm,
    SearchConfig,
    SearchFilter,
    SearchMode,
    SearchQuery,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/search", tags=["search"])


class SearchFilterRequest(BaseModel):
    """検索フィルターリクエスト"""

    field: str
    value: Any
    operator: str = "eq"


class FacetRequest(BaseModel):
    """ファセットリクエスト"""

    field: str
    limit: int = 10


class SearchRequest(BaseModel):
    """検索リクエスト用のモデル"""

    query: str = Field(..., description="検索クエリ")
    max_results: int = Field(10, description="最大取得件数", ge=1, le=100)
    offset: int = Field(0, description="オフセット", ge=0)
    search_mode: SearchMode = Field(SearchMode.HYBRID, description="検索モード")
    filters: list[SearchFilterRequest] = Field(
        default_factory=list, description="フィルター条件"
    )
    facets: list[str] = Field(
        default_factory=list, description="ファセット計算フィールド"
    )

    # ハイブリッド検索設定
    dense_weight: float | None = Field(None, description="Dense検索重み", ge=0, le=1)
    sparse_weight: float | None = Field(None, description="Sparse検索重み", ge=0, le=1)
    similarity_threshold: float | None = Field(
        None, description="類似度閾値", ge=0, le=1
    )
    enable_reranking: bool | None = Field(None, description="リランキング有効化")


class SearchResultDocument(BaseModel):
    """検索結果ドキュメント"""

    id: str
    title: str
    content: str
    search_score: float
    source_type: str | None = None
    language: str | None = None
    document_type: str | None = None
    metadata: dict[str, Any] | None = None
    rerank_score: float | None = None
    ranking_explanation: dict[str, Any] | None = None


class FacetValue(BaseModel):
    """ファセット値"""

    value: str
    count: int


class SearchFacet(BaseModel):
    """検索ファセット"""

    field: str
    values: list[FacetValue]


class SearchResponse(BaseModel):
    """検索レスポンス用のモデル"""

    success: bool
    query: str
    total_hits: int
    search_time: float
    documents: list[SearchResultDocument]
    facets: list[SearchFacet] | None = None
    error_message: str | None = None


class SearchSuggestionsResponse(BaseModel):
    """検索サジェストレスポンス"""

    suggestions: list[str]
    query: str


# 認証依存性
async def get_current_user_or_api_key(
    authorization: str | None = Header(None), x_api_key: str | None = Header(None)
) -> dict[str, Any]:
    """JWT認証またはAPI Key認証を試行"""
    # API Key認証を先に試行
    if x_api_key:
        api_key_info = validate_api_key(x_api_key)
        if api_key_info:
            return {
                "user_id": api_key_info["user_id"],
                "permissions": api_key_info["permissions"],
                "auth_type": "api_key",
            }

    # JWT認証を試行
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        try:
            from app.core.auth import is_token_blacklisted, users_storage, verify_token

            if is_token_blacklisted(token):
                raise HTTPException(status_code=401, detail="Token has been revoked")

            payload = verify_token(token)
            email = payload.get("sub")
            if email:
                user = users_storage.get(email)
                if user:
                    user_info = user.copy()
                    user_info["email"] = email
                    user_info["auth_type"] = "jwt"
                    return user_info
        except Exception as e:
            logging.debug(f"JWT認証に失敗、API Key認証にフォールバック: {e}")
            pass

    raise HTTPException(status_code=401, detail="Authentication required")


# 検索エンジン依存性注入
async def get_hybrid_search_engine() -> HybridSearchEngine:
    """ハイブリッド検索エンジンの依存性注入"""
    # 実際の実装では、DIコンテナやファクトリを使用
    search_config = SearchConfig(
        dense_weight=0.7,
        sparse_weight=0.3,
        top_k=20,
        search_mode=SearchMode.HYBRID,
        ranking_algorithm=RankingAlgorithm.RRF,
        enable_reranking=True,
        similarity_threshold=0.0,
    )

    # 実際のサービスを注入
    from app.services.embedding_service import EmbeddingService, EmbeddingConfig
    from app.repositories.document_repository import DocumentRepository
    from app.repositories.chunk_repository import DocumentChunkRepository
    
    # サービスの初期化
    embedding_config = EmbeddingConfig()
    embedding_service = EmbeddingService(embedding_config)
    await embedding_service.initialize()
    
    document_repository = DocumentRepository()
    chunk_repository = DocumentChunkRepository()

    return HybridSearchEngine(
        config=search_config,
        embedding_service=embedding_service,
        document_repository=document_repository,
        chunk_repository=chunk_repository,
    )


@router.post("/", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    current_user: dict = Depends(get_current_user_or_api_key),
    search_engine: HybridSearchEngine = Depends(get_hybrid_search_engine),
):
    """ハイブリッドドキュメント検索

    BGE-M3を使用したdense/sparse vectorsハイブリッド検索を実行します。
    RRF（Reciprocal Rank Fusion）で結果を統合し、関連性スコアリングで最適化します。
    """
    try:
        # 読み取り権限をチェック
        if "read" not in current_user.get("permissions", []):
            raise HTTPException(status_code=403, detail="Read permission required")

        # 検索設定の動的更新
        if request.dense_weight is not None and request.sparse_weight is not None:
            if abs(request.dense_weight + request.sparse_weight - 1.0) > 0.001:
                raise HTTPException(
                    status_code=400,
                    detail="dense_weight + sparse_weight must equal 1.0",
                )
            search_engine.config.dense_weight = request.dense_weight
            search_engine.config.sparse_weight = request.sparse_weight

        if request.similarity_threshold is not None:
            search_engine.config.similarity_threshold = request.similarity_threshold

        if request.enable_reranking is not None:
            search_engine.config.enable_reranking = request.enable_reranking

        # 検索フィルターの変換
        search_filters = [
            SearchFilter(field=f.field, value=f.value, operator=f.operator)
            for f in request.filters
        ]

        # 検索クエリの構築
        search_query = SearchQuery(
            text=request.query,
            filters=search_filters,
            facets=request.facets,
            search_mode=request.search_mode,
            max_results=request.max_results,
            offset=request.offset,
        )

        # ハイブリッド検索実行
        search_result = await search_engine.search(search_query)

        # レスポンスの構築
        if search_result.success:
            # ドキュメント変換
            result_documents = [
                SearchResultDocument(
                    id=doc["id"],
                    title=doc.get("title", ""),
                    content=doc.get("content", ""),
                    search_score=doc.get("search_score", 0.0),
                    source_type=doc.get("source_type"),
                    language=doc.get("language"),
                    document_type=doc.get("document_type"),
                    metadata=doc.get("metadata"),
                    rerank_score=doc.get("rerank_score"),
                    ranking_explanation=doc.get("ranking_explanation"),
                )
                for doc in search_result.documents
            ]

            # ファセット変換
            response_facets = None
            if search_result.facets:
                response_facets = [
                    SearchFacet(
                        field=field,
                        values=[
                            FacetValue(value=facet.value, count=facet.count)
                            for facet in facets
                        ],
                    )
                    for field, facets in search_result.facets.items()
                ]

            return SearchResponse(
                success=True,
                query=search_result.query,
                total_hits=search_result.total_hits,
                search_time=search_result.search_time,
                documents=result_documents,
                facets=response_facets,
            )
        else:
            return SearchResponse(
                success=False,
                query=search_result.query,
                total_hits=0,
                search_time=search_result.search_time,
                documents=[],
                error_message=search_result.error_message,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}") from e


@router.get("/suggestions", response_model=SearchSuggestionsResponse)
async def get_search_suggestions(
    q: str,
    limit: int = 5,
    current_user: dict = Depends(get_current_user_or_api_key),
):
    """検索サジェスト取得

    入力されたクエリに基づいて検索候補を提供します。
    """
    try:
        # 読み取り権限をチェック
        if "read" not in current_user.get("permissions", []):
            raise HTTPException(status_code=403, detail="Read permission required")

        # 簡易的なサジェスト実装（実際は検索履歴やインデックスから生成）
        suggestions = []
        if q:
            base_suggestions = [
                "machine learning algorithms",
                "natural language processing",
                "deep learning networks",
                "data processing pipeline",
                "search engine optimization",
                "database design patterns",
                "web api best practices",
                "software architecture",
                "cloud computing",
                "artificial intelligence",
            ]

            # 部分一致でフィルタリング
            suggestions = [s for s in base_suggestions if q.lower() in s.lower()][
                :limit
            ]

        return SearchSuggestionsResponse(
            suggestions=suggestions,
            query=q,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Suggestions failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Suggestions failed: {str(e)}"
        ) from e


@router.get("/config", response_model=dict[str, Any])
async def get_search_config(
    current_user: dict = Depends(get_current_user_or_api_key),
):
    """検索設定取得

    現在の検索エンジン設定を返します。
    """
    try:
        # 読み取り権限をチェック
        if "read" not in current_user.get("permissions", []):
            raise HTTPException(status_code=403, detail="Read permission required")

        # デフォルト設定を返す
        return {
            "search_modes": [mode.value for mode in SearchMode],
            "ranking_algorithms": [algo.value for algo in RankingAlgorithm],
            "default_config": {
                "dense_weight": 0.7,
                "sparse_weight": 0.3,
                "top_k": 20,
                "search_mode": SearchMode.HYBRID.value,
                "ranking_algorithm": RankingAlgorithm.RRF.value,
                "enable_reranking": True,
                "similarity_threshold": 0.0,
                "search_timeout": 30.0,
            },
            "available_filters": [
                "source_type",
                "language",
                "document_type",
                "metadata.category",
                "metadata.author",
                "metadata.tags",
            ],
            "available_facets": [
                "source_type",
                "language",
                "document_type",
                "metadata.category",
                "metadata.difficulty",
                "metadata.tags",
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Config retrieval failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Config retrieval failed: {str(e)}"
        ) from e


@router.post("/semantic", response_model=SearchResponse)
async def search_semantic(
    request: SearchRequest,
    current_user: dict = Depends(get_current_user_or_api_key),
    search_engine: HybridSearchEngine = Depends(get_hybrid_search_engine),
):
    """セマンティック検索（Dense Vector重視）
    
    BGE-M3のdense vectorを使用した意味的検索を実行します。
    """
    # SemanticモードでRequestを上書き
    request.search_mode = SearchMode.SEMANTIC
    
    # 通常の検索エンドポイントを呼び出し
    return await search_documents(request, current_user, search_engine)


@router.post("/keyword", response_model=SearchResponse)
async def search_keyword(
    request: SearchRequest,
    current_user: dict = Depends(get_current_user_or_api_key),
    search_engine: HybridSearchEngine = Depends(get_hybrid_search_engine),
):
    """キーワード検索（Sparse Vector重視）
    
    BGE-M3のsparse vectorを使用したキーワードベース検索を実行します。
    """
    # KeywordモードでRequestを上書き
    request.search_mode = SearchMode.KEYWORD
    
    # 通常の検索エンドポイントを呼び出し
    return await search_documents(request, current_user, search_engine)
