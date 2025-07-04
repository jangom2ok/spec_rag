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


class DateRange(BaseModel):
    """日付範囲フィルター"""
    
    from_date: str = Field(alias="from", description="開始日 (YYYY-MM-DD)")
    to_date: str = Field(alias="to", description="終了日 (YYYY-MM-DD)")


class EnhancedFilters(BaseModel):
    """拡張フィルター"""
    
    source_types: list[str] | None = Field(None, description="ソースタイプフィルター")
    languages: list[str] | None = Field(None, description="言語フィルター")
    date_range: DateRange | None = Field(None, description="日付範囲フィルター")
    tags: list[str] | None = Field(None, description="タグフィルター")


class SearchOptions(BaseModel):
    """検索オプション"""
    
    search_type: str = Field("hybrid", description="検索タイプ: dense, sparse, hybrid")
    max_results: int = Field(10, description="最大結果数", ge=1, le=100)
    min_score: float = Field(0.0, description="最小スコア", ge=0.0, le=1.0)
    include_metadata: bool = Field(True, description="メタデータを含める")
    highlight: bool = Field(False, description="ハイライト機能を有効にする")


class RankingOptions(BaseModel):
    """ランキングオプション"""
    
    dense_weight: float = Field(0.7, description="Dense検索重み", ge=0.0, le=1.0)
    sparse_weight: float = Field(0.3, description="Sparse検索重み", ge=0.0, le=1.0)
    rerank: bool = Field(True, description="リランキングを有効にする")
    diversity: bool = Field(False, description="多様性を考慮する")


class SearchRequest(BaseModel):
    """検索リクエスト用のモデル（仕様書準拠版）"""

    query: str = Field(..., description="検索クエリ")
    filters: EnhancedFilters | None = Field(None, description="拡張フィルター条件")
    search_options: SearchOptions | None = Field(None, description="検索オプション")
    ranking_options: RankingOptions | None = Field(None, description="ランキングオプション")
    
    # レガシー互換性のためのフィールド（既存APIとの後方互換性）
    max_results: int | None = Field(None, description="最大取得件数（レガシー）", ge=1, le=100)
    offset: int = Field(0, description="オフセット", ge=0)
    search_mode: SearchMode | None = Field(None, description="検索モード（レガシー）")
    legacy_filters: list[SearchFilterRequest] = Field(
        default_factory=list, description="従来のフィルター条件", alias="legacy_filters"
    )
    facets: list[str] = Field(
        default_factory=list, description="ファセット計算フィールド"
    )

    # レガシー互換性フィールド
    dense_weight: float | None = Field(None, description="Dense検索重み（レガシー）", ge=0, le=1)
    sparse_weight: float | None = Field(None, description="Sparse検索重み（レガシー）", ge=0, le=1)
    similarity_threshold: float | None = Field(
        None, description="類似度閾値（レガシー）", ge=0, le=1
    )
    enable_reranking: bool | None = Field(None, description="リランキング有効化（レガシー）")


class SourceInfo(BaseModel):
    """ソース情報"""
    
    type: str = Field(..., description="ソースタイプ")
    url: str | None = Field(None, description="ソースURL")
    author: str | None = Field(None, description="作成者")
    last_updated: str | None = Field(None, description="最終更新日時")


class ContextInfo(BaseModel):
    """コンテキスト情報"""
    
    hierarchy_path: str | None = Field(None, description="階層パス")
    parent_sections: list[str] = Field(default_factory=list, description="親セクション")
    related_chunks: list[str] = Field(default_factory=list, description="関連チャンク")


class SearchResultDocument(BaseModel):
    """検索結果ドキュメント（仕様書準拠版）"""

    document_id: str = Field(..., description="ドキュメントID")
    chunk_id: str | None = Field(None, description="チャンクID")
    score: float = Field(..., description="検索スコア")
    chunk_type: str | None = Field(None, description="チャンクタイプ")
    title: str = Field(..., description="タイトル")
    content: str = Field(..., description="コンテンツ")
    highlighted_content: str | None = Field(None, description="ハイライト済みコンテンツ")
    source: SourceInfo | None = Field(None, description="ソース情報")
    metadata: dict[str, Any] | None = Field(None, description="メタデータ")
    context: ContextInfo | None = Field(None, description="コンテキスト情報")
    
    # レガシー互換性フィールド
    id: str | None = Field(None, description="ID（レガシー）")
    search_score: float | None = Field(None, description="検索スコア（レガシー）")
    source_type: str | None = Field(None, description="ソースタイプ（レガシー）")
    language: str | None = Field(None, description="言語（レガシー）")
    document_type: str | None = Field(None, description="ドキュメントタイプ（レガシー）")
    rerank_score: float | None = Field(None, description="リランクスコア（レガシー）")
    ranking_explanation: dict[str, Any] | None = Field(None, description="ランキング説明（レガシー）")


class FacetValue(BaseModel):
    """ファセット値"""

    value: str
    count: int


class SearchFacet(BaseModel):
    """検索ファセット"""

    field: str
    values: list[FacetValue]


class SearchResponse(BaseModel):
    """検索レスポンス用のモデル（仕様書準拠版）"""

    query: str = Field(..., description="検索クエリ")
    total_results: int = Field(..., description="総結果数")
    returned_results: int = Field(..., description="返却結果数")
    search_time_ms: float = Field(..., description="検索時間（ミリ秒）")
    results: list[SearchResultDocument] = Field(..., description="検索結果")
    facets: dict[str, dict[str, int]] | None = Field(None, description="ファセット")
    suggestions: list[str] | None = Field(None, description="検索候補")
    
    # レガシー互換性フィールド
    success: bool | None = Field(None, description="成功フラグ（レガシー）")
    total_hits: int | None = Field(None, description="総ヒット数（レガシー）")
    search_time: float | None = Field(None, description="検索時間秒（レガシー）")
    documents: list[SearchResultDocument] | None = Field(None, description="ドキュメント（レガシー）")
    legacy_facets: list[SearchFacet] | None = Field(None, description="ファセット（レガシー）", alias="legacy_facets")
    error_message: str | None = Field(None, description="エラーメッセージ")


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


# ユーティリティ関数
def highlight_content(content: str, query: str) -> str:
    """コンテンツにハイライトを追加"""
    if not query or not content:
        return content
    
    import re
    # クエリの単語を分割
    query_words = query.split()
    highlighted_content = content
    
    for word in query_words:
        if len(word) > 2:  # 短すぎる単語はスキップ
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted_content = pattern.sub(f"**{word}**", highlighted_content)
    
    return highlighted_content


def convert_enhanced_filters_to_legacy(enhanced_filters: EnhancedFilters | None) -> list[SearchFilter]:
    """拡張フィルターを従来のフィルター形式に変換"""
    legacy_filters = []
    
    if not enhanced_filters:
        return legacy_filters
    
    if enhanced_filters.source_types:
        legacy_filters.append(
            SearchFilter(field="source_type", value=enhanced_filters.source_types, operator="in")
        )
    
    if enhanced_filters.languages:
        legacy_filters.append(
            SearchFilter(field="language", value=enhanced_filters.languages, operator="in")
        )
    
    if enhanced_filters.tags:
        legacy_filters.append(
            SearchFilter(field="metadata.tags", value=enhanced_filters.tags, operator="contains_any")
        )
    
    if enhanced_filters.date_range:
        legacy_filters.extend([
            SearchFilter(field="updated_at", value=enhanced_filters.date_range.from_date, operator="gte"),
            SearchFilter(field="updated_at", value=enhanced_filters.date_range.to_date, operator="lte")
        ])
    
    return legacy_filters


def generate_search_suggestions(query: str, results: list[dict]) -> list[str]:
    """検索結果に基づいて候補を生成"""
    suggestions = []
    
    # 基本的な候補リスト
    base_suggestions = [
        f"{query} 実装",
        f"{query} 設定",
        f"{query} エラー",
        f"{query} 使い方",
        f"{query} ガイド",
    ]
    
    # 結果から関連タグを抽出して候補を生成
    tags = set()
    for result in results[:5]:  # 上位5件から抽出
        metadata = result.get("metadata", {})
        if isinstance(metadata, dict) and "tags" in metadata:
            if isinstance(metadata["tags"], list):
                tags.update(metadata["tags"][:3])  # 各結果から最大3個のタグ
    
    # タグベースの候補を追加
    for tag in list(tags)[:3]:  # 最大3個のタグベース候補
        suggestions.append(f"{tag} {query}")
    
    # 重複除去とフィルタリング
    unique_suggestions = list(dict.fromkeys(suggestions))
    return unique_suggestions[:5]  # 最大5個の候補


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
    """ハイブリッドドキュメント検索（仕様書準拠版）

    BGE-M3を使用したdense/sparse vectorsハイブリッド検索を実行します。
    RRF（Reciprocal Rank Fusion）で結果を統合し、関連性スコアリングで最適化します。
    
    新しい検索API仕様（filters, search_options, ranking_options）をサポートします。
    """
    import time
    start_time = time.time()
    
    try:
        # 読み取り権限をチェック
        if "read" not in current_user.get("permissions", []):
            raise HTTPException(status_code=403, detail="Read permission required")

        # 検索オプションの処理
        search_options = request.search_options or SearchOptions()
        ranking_options = request.ranking_options or RankingOptions()
        
        # レガシー互換性: 古いフィールドがある場合は優先
        max_results = request.max_results or search_options.max_results
        
        # 検索モードの決定
        search_mode = SearchMode.HYBRID
        if search_options.search_type == "dense":
            search_mode = SearchMode.DENSE
        elif search_options.search_type == "sparse":
            search_mode = SearchMode.SPARSE
        elif request.search_mode:  # レガシー互換性
            search_mode = request.search_mode

        # 検索設定の動的更新
        if ranking_options.dense_weight + ranking_options.sparse_weight != 1.0:
            # 重みを正規化
            total_weight = ranking_options.dense_weight + ranking_options.sparse_weight
            if total_weight > 0:
                ranking_options.dense_weight /= total_weight
                ranking_options.sparse_weight /= total_weight
            else:
                ranking_options.dense_weight = 0.7
                ranking_options.sparse_weight = 0.3

        search_engine.config.dense_weight = ranking_options.dense_weight
        search_engine.config.sparse_weight = ranking_options.sparse_weight
        search_engine.config.enable_reranking = ranking_options.rerank
        search_engine.config.similarity_threshold = search_options.min_score

        # レガシー設定があれば優先
        if request.dense_weight is not None:
            search_engine.config.dense_weight = request.dense_weight
        if request.sparse_weight is not None:
            search_engine.config.sparse_weight = request.sparse_weight
        if request.similarity_threshold is not None:
            search_engine.config.similarity_threshold = request.similarity_threshold
        if request.enable_reranking is not None:
            search_engine.config.enable_reranking = request.enable_reranking

        # フィルターの変換
        search_filters = []
        
        # 新しいフィルター形式
        if request.filters:
            search_filters.extend(convert_enhanced_filters_to_legacy(request.filters))
        
        # レガシーフィルター
        if request.legacy_filters:
            search_filters.extend([
                SearchFilter(field=f.field, value=f.value, operator=f.operator)
                for f in request.legacy_filters
            ])

        # 検索クエリの構築
        search_query = SearchQuery(
            text=request.query,
            filters=search_filters,
            facets=request.facets,
            search_mode=search_mode,
            max_results=max_results,
            offset=request.offset,
        )

        # ハイブリッド検索実行
        search_result = await search_engine.search(search_query)

        # 検索時間の計算
        search_time_ms = (time.time() - start_time) * 1000

        # レスポンスの構築
        if search_result.success:
            # ドキュメント変換
            result_documents = []
            for doc in search_result.documents:
                # ハイライト処理
                highlighted_content = None
                if search_options.highlight:
                    highlighted_content = highlight_content(doc.get("content", ""), request.query)
                
                # ソース情報の構築
                source_info = None
                if search_options.include_metadata:
                    source_info = SourceInfo(
                        type=doc.get("source_type", "unknown"),
                        url=doc.get("metadata", {}).get("url"),
                        author=doc.get("metadata", {}).get("author"),
                        last_updated=doc.get("updated_at")
                    )
                
                # コンテキスト情報の構築
                context_info = None
                if search_options.include_metadata:
                    context_info = ContextInfo(
                        hierarchy_path=doc.get("hierarchy_path"),
                        parent_sections=doc.get("metadata", {}).get("parent_sections", []),
                        related_chunks=doc.get("metadata", {}).get("related_chunks", [])
                    )
                
                # 結果ドキュメントの作成
                result_doc = SearchResultDocument(
                    document_id=doc.get("id", doc.get("document_id", "")),
                    chunk_id=doc.get("chunk_id"),
                    score=doc.get("search_score", 0.0),
                    chunk_type=doc.get("chunk_type"),
                    title=doc.get("title", ""),
                    content=doc.get("content", ""),
                    highlighted_content=highlighted_content,
                    source=source_info,
                    metadata=doc.get("metadata") if search_options.include_metadata else None,
                    context=context_info,
                    # レガシー互換性
                    id=doc.get("id"),
                    search_score=doc.get("search_score", 0.0),
                    source_type=doc.get("source_type"),
                    language=doc.get("language"),
                    document_type=doc.get("document_type"),
                    rerank_score=doc.get("rerank_score"),
                    ranking_explanation=doc.get("ranking_explanation"),
                )
                result_documents.append(result_doc)

            # ファセットの変換（新形式）
            facets_dict = {}
            if search_result.facets:
                for field, facet_list in search_result.facets.items():
                    facets_dict[field] = {
                        facet.value: facet.count for facet in facet_list
                    }

            # 検索候補の生成
            suggestions = generate_search_suggestions(request.query, search_result.documents)

            return SearchResponse(
                query=request.query,
                total_results=search_result.total_hits,
                returned_results=len(result_documents),
                search_time_ms=search_time_ms,
                results=result_documents,
                facets=facets_dict if facets_dict else None,
                suggestions=suggestions,
                # レガシー互換性
                success=True,
                total_hits=search_result.total_hits,
                search_time=search_time_ms / 1000,
                documents=result_documents,
                legacy_facets=None,
                error_message=None,
            )
        else:
            return SearchResponse(
                query=request.query,
                total_results=0,
                returned_results=0,
                search_time_ms=search_time_ms,
                results=[],
                facets=None,
                suggestions=None,
                # レガシー互換性
                success=False,
                total_hits=0,
                search_time=search_time_ms / 1000,
                documents=[],
                error_message=search_result.error_message,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}") from e


@router.post("/semantic", response_model=SearchResponse)
async def semantic_search(
    request: SearchRequest,
    current_user: dict = Depends(get_current_user_or_api_key),
    search_engine: HybridSearchEngine = Depends(get_hybrid_search_engine),
):
    """セマンティック検索
    
    Dense vectorのみを使用したセマンティック検索を実行します。
    文脈的な意味の類似性に基づいて検索結果を返します。
    """
    try:
        # 読み取り権限をチェック
        if "read" not in current_user.get("permissions", []):
            raise HTTPException(status_code=403, detail="Read permission required")

        # セマンティック検索用にモードを強制設定
        search_engine.config.search_mode = SearchMode.DENSE
        search_engine.config.dense_weight = 1.0
        search_engine.config.sparse_weight = 0.0

        # その他の設定は通常の検索と同様に適用
        if request.similarity_threshold is not None:
            search_engine.config.similarity_threshold = request.similarity_threshold
        if request.enable_reranking is not None:
            search_engine.config.enable_reranking = request.enable_reranking

        # 検索フィルターの変換
        search_filters = [
            SearchFilter(field=f.field, value=f.value, operator=f.operator)
            for f in request.filters
        ]

        # 検索クエリの構築（セマンティック検索用）
        search_query = SearchQuery(
            text=request.query,
            filters=search_filters,
            facets=request.facets,
            search_mode=SearchMode.DENSE,
            max_results=request.max_results,
            offset=request.offset,
        )

        # セマンティック検索実行
        search_result = await search_engine.search(search_query)

        # レスポンスの構築（通常の検索と同じロジック）
        if search_result.success:
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
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}") from e


@router.post("/keyword", response_model=SearchResponse)
async def keyword_search(
    request: SearchRequest,
    current_user: dict = Depends(get_current_user_or_api_key),
    search_engine: HybridSearchEngine = Depends(get_hybrid_search_engine),
):
    """キーワード検索
    
    Sparse vectorのみを使用したキーワードベースの検索を実行します。
    完全一致や部分一致に基づいて検索結果を返します。
    """
    try:
        # 読み取り権限をチェック
        if "read" not in current_user.get("permissions", []):
            raise HTTPException(status_code=403, detail="Read permission required")

        # キーワード検索用にモードを強制設定
        search_engine.config.search_mode = SearchMode.SPARSE
        search_engine.config.dense_weight = 0.0
        search_engine.config.sparse_weight = 1.0

        # その他の設定は通常の検索と同様に適用
        if request.similarity_threshold is not None:
            search_engine.config.similarity_threshold = request.similarity_threshold
        if request.enable_reranking is not None:
            search_engine.config.enable_reranking = request.enable_reranking

        # 検索フィルターの変換
        search_filters = [
            SearchFilter(field=f.field, value=f.value, operator=f.operator)
            for f in request.filters
        ]

        # 検索クエリの構築（キーワード検索用）
        search_query = SearchQuery(
            text=request.query,
            filters=search_filters,
            facets=request.facets,
            search_mode=SearchMode.SPARSE,
            max_results=request.max_results,
            offset=request.offset,
        )

        # キーワード検索実行
        search_result = await search_engine.search(search_query)

        # レスポンスの構築（通常の検索と同じロジック）
        if search_result.success:
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
        logger.error(f"Keyword search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Keyword search failed: {str(e)}") from e


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
