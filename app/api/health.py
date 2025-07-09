"""ヘルスチェックAPI"""

import time
from datetime import datetime
from typing import Any

import psutil
try:
    from aperturedb import DBException
except ImportError:
    from app.models.aperturedb_mock import DBException
from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import SQLAlchemyError

router = APIRouter(prefix="/v1/health", tags=["health"])


async def check_postgresql_connection() -> dict[str, Any]:
    """PostgreSQL接続チェック"""
    try:
        # 実際のDB接続チェックのモック
        # 実装時にはSQLAlchemyのエンジンを使用する
        start_time = time.time()
        # await database.engine.execute("SELECT 1")
        response_time = time.time() - start_time

        return {"status": "healthy", "response_time": response_time}
    except SQLAlchemyError as e:
        return {"status": "unhealthy", "error": str(e)}


async def check_aperturedb_connection() -> dict[str, Any]:
    """ApertureDB接続チェック"""
    try:
        # 実際のApertureDB接続チェックのモック
        # 実装時にはApertureDBクライアントを使用する
        return {"status": "healthy", "descriptor_sets": 2}  # モック値
    except DBException as e:
        return {"status": "unhealthy", "error": str(e)}


async def get_system_metrics() -> dict[str, Any]:
    """システムメトリクス取得"""
    try:
        return {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent,
            "uptime": time.time() - psutil.boot_time(),
        }
    except Exception:
        return {"cpu_usage": 0.0, "memory_usage": 0.0, "disk_usage": 0.0, "uptime": 0.0}


async def get_health_status() -> dict[str, Any]:
    """基本ヘルスステータス取得"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "environment": "development",
    }


@router.get("")
async def health_check():
    """基本ヘルスチェック"""
    return await get_health_status()


@router.get("/detailed")
async def detailed_health_check():
    """詳細ヘルスチェック"""
    try:
        postgresql_status = await check_postgresql_connection()
        aperturedb_status = await check_aperturedb_connection()
        system_metrics = await get_system_metrics()

        # 全体のステータス判定
        services_healthy = (
            postgresql_status["status"] == "healthy"
            and aperturedb_status["status"] == "healthy"
        )

        overall_status = "healthy" if services_healthy else "unhealthy"

        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "postgresql": postgresql_status,
                "aperturedb": aperturedb_status,
            },
            "system": system_metrics,
            "version": "1.0.0",
        }
    except Exception as e:
        # エラー時でもservicesフィールドを含める
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "postgresql": {"status": "unhealthy", "error": str(e)},
                "aperturedb": {"status": "unhealthy", "error": str(e)},
            },
            "system": await get_system_metrics(),
            "error": str(e),
            "version": "1.0.0",
        }


@router.get("/ready")
async def readiness_probe():
    """Readiness Probe - アプリケーションが要求を処理する準備ができているかチェック"""
    try:
        # データベース接続チェック
        postgresql_status = await check_postgresql_connection()
        aperturedb_status = await check_aperturedb_connection()

        ready = (
            postgresql_status["status"] == "healthy"
            and aperturedb_status["status"] == "healthy"
        )

        if ready:
            return {"ready": True, "timestamp": datetime.utcnow().isoformat()}
        else:
            raise HTTPException(
                status_code=503,
                detail={"ready": False, "timestamp": datetime.utcnow().isoformat()},
            )
    except Exception as err:
        raise HTTPException(
            status_code=503,
            detail={"ready": False, "timestamp": datetime.utcnow().isoformat()},
        ) from err


@router.get("/live")
async def liveness_probe():
    """Liveness Probe - アプリケーションが生存しているかチェック"""
    return {"alive": True, "timestamp": datetime.utcnow().isoformat()}
