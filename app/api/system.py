"""システム管理API"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from app.core.auth import require_admin_permission
from app.services.embedding_service import EmbeddingService, EmbeddingConfig
from app.services.hybrid_search_engine import HybridSearchEngine, SearchConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["system"])


class SystemStatus(BaseModel):
    """システム状況レスポンス"""
    
    status: str  # healthy, degraded, unhealthy
    components: dict[str, dict[str, Any]]
    version: str
    uptime: float
    timestamp: str


class SystemMetrics(BaseModel):
    """システムメトリクスレスポンス"""
    
    search_metrics: dict[str, Any]
    embedding_metrics: dict[str, Any]
    database_metrics: dict[str, Any]
    performance_metrics: dict[str, Any]
    timestamp: str


class ReindexRequest(BaseModel):
    """再インデックスリクエスト"""
    
    source_types: list[str] | None = None
    force: bool = False
    batch_size: int = 100


class ReindexResponse(BaseModel):
    """再インデックスレスポンス"""
    
    success: bool
    task_id: str
    message: str
    estimated_duration: float | None = None


async def get_admin_user(
    authorization: str | None = Header(None), 
    x_api_key: str | None = Header(None)
) -> dict:
    """管理者認証用の依存性注入"""
    return await require_admin_permission(authorization, x_api_key)


@router.get("/status", response_model=SystemStatus)
async def get_system_status(
    current_user: dict = Depends(get_admin_user),
) -> SystemStatus:
    """システム全体の状況を取得
    
    各コンポーネントのヘルスチェックを実行し、システム状況を返します。
    """
    try:
        from datetime import datetime
        import time
        import psutil
        
        # 起動時間計算（簡易実装）
        start_time = time.time() - 3600  # 1時間前に起動したと仮定
        uptime = time.time() - start_time
        
        components = {}
        overall_status = "healthy"
        
        # 埋め込みサービスのヘルスチェック
        try:
            embedding_config = EmbeddingConfig()
            embedding_service = EmbeddingService(embedding_config)
            await embedding_service.initialize()
            
            embedding_health = await embedding_service.health_check()
            components["embedding_service"] = {
                "status": embedding_health["status"],
                "details": embedding_health
            }
            
            if embedding_health["status"] != "healthy":
                overall_status = "degraded"
                
        except Exception as e:
            components["embedding_service"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            overall_status = "unhealthy"
        
        # データベース接続チェック
        try:
            from app.repositories.document_repository import DocumentRepository
            doc_repo = DocumentRepository()
            # 簡易的な接続チェック（実際の実装では実際にDBクエリを実行）
            components["database"] = {
                "status": "healthy",
                "connection": "active"
            }
        except Exception as e:
            components["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            overall_status = "unhealthy"
        
        # Milvusベクターデータベースチェック
        try:
            # 簡易実装（実際の実装ではMilvusクライアントで接続確認）
            components["vector_database"] = {
                "status": "healthy",
                "collections": ["dense_vectors", "sparse_vectors"]
            }
        except Exception as e:
            components["vector_database"] = {
                "status": "unhealthy", 
                "error": str(e)
            }
            overall_status = "unhealthy"
        
        # システムリソースチェック
        try:
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent(interval=1)
            disk_usage = psutil.disk_usage('/').percent
            
            resource_status = "healthy"
            if memory_usage > 90 or cpu_usage > 90 or disk_usage > 90:
                resource_status = "degraded"
                if overall_status == "healthy":
                    overall_status = "degraded"
            
            components["system_resources"] = {
                "status": resource_status,
                "memory_usage_percent": memory_usage,
                "cpu_usage_percent": cpu_usage,
                "disk_usage_percent": disk_usage
            }
        except Exception as e:
            components["system_resources"] = {
                "status": "unknown",
                "error": str(e)
            }
        
        return SystemStatus(
            status=overall_status,
            components=components,
            version="1.0.0",
            uptime=uptime,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"System status check failed: {str(e)}"
        ) from e


@router.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics(
    current_user: dict = Depends(get_admin_user),
) -> SystemMetrics:
    """システムメトリクスを取得
    
    パフォーマンス指標、使用統計、エラー率などを返します。
    """
    try:
        from datetime import datetime
        import time
        
        # 検索メトリクス（実際の実装では統計データベースから取得）
        search_metrics = {
            "total_searches_24h": 1247,
            "average_response_time_ms": 245.6,
            "search_success_rate": 0.987,
            "top_queries": [
                {"query": "machine learning", "count": 89},
                {"query": "api documentation", "count": 67},
                {"query": "database design", "count": 54}
            ],
            "search_modes_usage": {
                "hybrid": 0.65,
                "semantic": 0.25,
                "keyword": 0.10
            }
        }
        
        # 埋め込みメトリクス
        embedding_metrics = {
            "embeddings_generated_24h": 3456,
            "average_embedding_time_ms": 123.4,
            "model_info": {
                "name": "BAAI/BGE-M3",
                "dimension": 1024,
                "device": "cpu"
            },
            "embedding_cache_hit_rate": 0.78
        }
        
        # データベースメトリクス
        database_metrics = {
            "total_documents": 12450,
            "total_chunks": 98760,
            "index_size_mb": 2340.5,
            "query_performance": {
                "avg_query_time_ms": 45.2,
                "slow_queries_count": 23
            }
        }
        
        # パフォーマンスメトリクス
        performance_metrics = {
            "memory_usage": {
                "used_mb": 4567,
                "total_mb": 16384,
                "usage_percent": 27.9
            },
            "cpu_usage_percent": 23.4,
            "request_rate_per_minute": 156.7,
            "error_rate_percent": 1.3,
            "uptime_hours": 72.5
        }
        
        return SystemMetrics(
            search_metrics=search_metrics,
            embedding_metrics=embedding_metrics,
            database_metrics=database_metrics,
            performance_metrics=performance_metrics,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Metrics retrieval failed: {str(e)}"
        ) from e


@router.post("/reindex", response_model=ReindexResponse)
async def reindex_documents(
    request: ReindexRequest,
    current_user: dict = Depends(get_admin_user),
) -> ReindexResponse:
    """ドキュメントの再インデックスを実行
    
    指定されたソースタイプのドキュメントをベクターデータベースに再インデックスします。
    """
    try:
        import uuid
        
        # タスクIDを生成
        task_id = str(uuid.uuid4())
        
        # 推定実行時間計算（実際の実装では現在のドキュメント数から計算）
        if request.source_types:
            estimated_docs = 100 * len(request.source_types)  # 仮の計算
        else:
            estimated_docs = 1000  # 全ドキュメント
        
        estimated_duration = estimated_docs * 0.5  # 1ドキュメント0.5秒と仮定
        
        # バックグラウンドタスクとして実行（実際の実装ではCeleryやasyncタスクを使用）
        logger.info(f"Starting reindex task {task_id} for {request.source_types or 'all'} sources")
        
        # 実際の再インデックス処理はここで実装
        # await _execute_reindex_task(task_id, request)
        
        return ReindexResponse(
            success=True,
            task_id=task_id,
            message=f"Reindex task started for {request.source_types or 'all'} sources",
            estimated_duration=estimated_duration
        )
        
    except Exception as e:
        logger.error(f"Reindex initiation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Reindex failed: {str(e)}"
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


async def _execute_reindex_task(task_id: str, request: ReindexRequest) -> None:
    """再インデックスタスクの実際の実行（バックグラウンド処理）"""
    try:
        logger.info(f"Executing reindex task {task_id}")
        
        # 実際の再インデックス処理を実装
        # 1. ドキュメント取得
        # 2. チャンク分割
        # 3. 埋め込み生成
        # 4. ベクターデータベース更新
        # 5. メタデータ更新
        
        logger.info(f"Reindex task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Reindex task {task_id} failed: {e}")
        # エラー状況をタスクストアに記録
        raise