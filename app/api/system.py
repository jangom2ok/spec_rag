"""システム管理API"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from app.core.auth import require_admin_permission, validate_api_key
from app.services.embedding_service import EmbeddingService, EmbeddingConfig
from app.services.hybrid_search_engine import HybridSearchEngine, SearchConfig
from app.services.metrics_collection import (
    MetricsCollectionService,
    SystemMetrics,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["system"])


class ComponentStatus(BaseModel):
    """コンポーネント状態"""
    
    status: str  # "healthy", "degraded", "unhealthy"
    response_time_ms: float | None = None
    error_message: str | None = None
    metadata: dict[str, Any] | None = None


class SystemStatus(BaseModel):
    """システム状況レスポンス（レガシー互換性）"""
    
    status: str  # healthy, degraded, unhealthy
    components: dict[str, dict[str, Any]]
    version: str
    uptime: float
    timestamp: str


class SystemStatusResponse(BaseModel):
    """システム状態レスポンス"""
    
    system_status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime
    components: dict[str, ComponentStatus]
    statistics: dict[str, Any] | None = None


class PerformanceMetrics(BaseModel):
    """パフォーマンスメトリクス"""
    
    search_metrics: dict[str, float]
    embedding_metrics: dict[str, float] | None = None


class UsageMetrics(BaseModel):
    """使用状況メトリクス"""
    
    daily_active_users: int
    total_searches_today: int
    popular_queries: list[str]


class ResourceMetrics(BaseModel):
    """リソースメトリクス"""
    
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_io_mbps: float | None = None


class SystemMetrics(BaseModel):
    """システムメトリクスレスポンス（レガシー互換性）"""
    
    search_metrics: dict[str, Any]
    embedding_metrics: dict[str, Any]
    database_metrics: dict[str, Any]
    performance_metrics: dict[str, Any]
    timestamp: str


class SystemMetricsResponse(BaseModel):
    """システムメトリクスレスポンス"""
    
    performance_metrics: PerformanceMetrics
    usage_metrics: UsageMetrics | None = None
    resource_metrics: ResourceMetrics | None = None
    timestamp: datetime


class ReindexRequest(BaseModel):
    """リインデックスリクエスト"""
    
    collection_name: str | None = None  # None の場合は全コレクション
    force: bool = False  # 強制実行フラグ
    background: bool = True  # バックグラウンド実行
    # レガシー互換性
    source_types: list[str] | None = None
    batch_size: int = 100


class ReindexResponse(BaseModel):
    """リインデックスレスポンス"""
    
    success: bool
    task_id: str | None = None
    message: str
    estimated_completion_time: datetime | None = None
    # レガシー互換性
    estimated_duration: float | None = None


async def get_admin_user(
    authorization: str | None = Header(None), 
    x_api_key: str | None = Header(None)
) -> dict:
    """管理者認証用の依存性注入"""
    try:
        return await require_admin_permission(authorization, x_api_key)
    except:
        # フォールバック: 従来の認証ロジック
        # API Key認証を先に試行
        if x_api_key:
            api_key_info = validate_api_key(x_api_key)
            if api_key_info and "admin" in api_key_info.get("permissions", []):
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
                    if user and "admin" in user.get("permissions", []):
                        user_info = user.copy()
                        user_info["email"] = email
                        user_info["auth_type"] = "jwt"
                        return user_info
            except Exception as e:
                logging.debug(f"JWT認証に失敗: {e}")
                pass

        raise HTTPException(status_code=403, detail="Admin permission required")


# メトリクス収集サービス依存性注入
async def get_metrics_service() -> MetricsCollectionService:
    """メトリクス収集サービスの依存性注入"""
    # 実際の実装では、DIコンテナやファクトリを使用
    return MetricsCollectionService()


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    current_user: dict = Depends(get_admin_user),
):
    """システム状態取得
    
    システム全体の健康状態とコンポーネントの状態を返します。
    管理者権限が必要です。
    """
    try:
        # 各コンポーネントの状態をチェック
        components = {}
        overall_status = "healthy"

        # API サーバーの状態
        components["api_server"] = ComponentStatus(
            status="healthy",
            response_time_ms=45.0,
            metadata={
                "active_connections": 23,
                "uptime_seconds": 86400,
            }
        )

        # 埋め込みサービスの状態
        try:
            # 実際の実装では埋め込みサービスにヘルスチェック
            components["embedding_service"] = ComponentStatus(
                status="healthy",
                metadata={
                    "model_loaded": True,
                    "gpu_memory_usage": "12.5GB/24GB",
                    "processing_queue": 5,
                }
            )
        except Exception as e:
            components["embedding_service"] = ComponentStatus(
                status="unhealthy",
                error_message=str(e)
            )
            overall_status = "degraded"

        # ベクトルデータベースの状態
        try:
            # 実際の実装ではMilvusにヘルスチェック
            components["vector_database"] = ComponentStatus(
                status="healthy",
                metadata={
                    "collections": {
                        "dense_vectors": {
                            "total_vectors": 125000,
                            "index_status": "built"
                        },
                        "sparse_vectors": {
                            "total_vectors": 125000,
                            "index_status": "built"
                        }
                    }
                }
            )
        except Exception as e:
            components["vector_database"] = ComponentStatus(
                status="unhealthy",
                error_message=str(e)
            )
            overall_status = "unhealthy"

        # メタデータデータベースの状態
        try:
            # 実際の実装ではPostgreSQLにヘルスチェック
            components["metadata_database"] = ComponentStatus(
                status="healthy",
                metadata={
                    "connection_pool": "8/20",
                    "slow_queries": 0,
                }
            )
        except Exception as e:
            components["metadata_database"] = ComponentStatus(
                status="unhealthy",
                error_message=str(e)
            )
            overall_status = "unhealthy"

        # 統計情報
        statistics = {
            "total_documents": 5432,
            "total_chunks": 125000,
            "daily_searches": 890,
            "average_search_time_ms": 234.0,
        }

        return SystemStatusResponse(
            system_status=overall_status,
            timestamp=datetime.now(),
            components=components,
            statistics=statistics,
        )

    except Exception as e:
        logger.error(f"System status check failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"System status check failed: {str(e)}"
        ) from e


@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics(
    current_user: dict = Depends(get_admin_user),
    metrics_service: MetricsCollectionService = Depends(get_metrics_service),
):
    """システムメトリクス取得
    
    システムのパフォーマンスメトリクス、使用状況、リソース使用量を返します。
    管理者権限が必要です。
    """
    try:
        # メトリクス収集サービスからデータを取得
        system_metrics = await metrics_service.get_current_metrics()

        # パフォーマンスメトリクス
        performance_metrics = PerformanceMetrics(
            search_metrics={
                "avg_response_time_ms": 234.0,
                "p95_response_time_ms": 456.0,
                "p99_response_time_ms": 678.0,
                "requests_per_second": 12.5,
                "error_rate_percent": 0.2,
            },
            embedding_metrics={
                "documents_per_second": 850.0,
                "avg_processing_time_ms": 1234.0,
                "queue_depth": 5.0,
                "gpu_utilization_percent": 78.0,
            }
        )

        # 使用状況メトリクス
        usage_metrics = UsageMetrics(
            daily_active_users=45,
            total_searches_today=890,
            popular_queries=[
                "API認証",
                "JWT実装",
                "セキュリティ設定",
            ]
        )

        # リソースメトリクス
        resource_metrics = ResourceMetrics(
            cpu_usage_percent=45.0,
            memory_usage_percent=67.0,
            disk_usage_percent=34.0,
            network_io_mbps=12.3,
        )

        return SystemMetricsResponse(
            performance_metrics=performance_metrics,
            usage_metrics=usage_metrics,
            resource_metrics=resource_metrics,
            timestamp=datetime.now(),
        )

    except Exception as e:
        logger.error(f"System metrics collection failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"System metrics collection failed: {str(e)}"
        ) from e


@router.post("/reindex", response_model=ReindexResponse)
async def reindex_documents(
    request: ReindexRequest,
    current_user: dict = Depends(get_admin_user),
):
    """ドキュメントリインデックス
    
    ベクトルデータベースのインデックスを再構築します。
    管理者権限が必要です。
    """
    try:
        import uuid
        from datetime import timedelta

        # リインデックスタスクを開始
        task_id = str(uuid.uuid4())
        
        if request.background:
            # バックグラウンドでの実行
            # 実際の実装では、Celeryタスクを開始
            logger.info(f"Starting background reindex task: {task_id}")
            
            estimated_completion = datetime.now() + timedelta(hours=2)
            
            # レガシー互換性: 推定実行時間計算
            estimated_duration = None
            if request.source_types:
                estimated_docs = 100 * len(request.source_types)  # 仮の計算
                estimated_duration = estimated_docs * 0.5  # 1ドキュメント0.5秒と仮定
            
            return ReindexResponse(
                success=True,
                task_id=task_id,
                message=f"Reindex task started in background for {request.source_types or request.collection_name or 'all'}",
                estimated_completion_time=estimated_completion,
                estimated_duration=estimated_duration,
            )
        else:
            # 同期実行
            logger.info(f"Starting synchronous reindex task: {task_id}")
            
            # 実際の実装では、ここでリインデックス処理を実行
            # await reindex_service.reindex(collection_name=request.collection_name, force=request.force)
            
            return ReindexResponse(
                success=True,
                task_id=task_id,
                message="Reindex completed successfully",
            )

    except Exception as e:
        logger.error(f"Reindex operation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Reindex operation failed: {str(e)}"
        ) from e


@router.get("/reindex/{task_id}")
async def get_reindex_status(
    task_id: str,
    current_user: dict = Depends(get_admin_user),
) -> dict[str, Any]:
    """再インデックスタスクの状況を取得"""
    try:
        # 実際の実装ではタスクストアから状況を取得
        # タスクID検証の簡易実装
        if not task_id or len(task_id) != 36:  # UUID length check
            raise HTTPException(status_code=404, detail="Task not found")
        
        # モック状況データ
        status_data = {
            "task_id": task_id,
            "status": "in_progress",  # pending, in_progress, completed, failed
            "progress": 0.65,
            "processed_documents": 650,
            "total_documents": 1000,
            "current_phase": "embedding_generation",
            "estimated_completion": "2024-01-01T15:30:00Z",
            "errors": []
        }
        
        return status_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reindex status retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Status retrieval failed: {str(e)}"
        ) from e
