import uuid
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from aperturedb import DBException  # type: ignore
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.api.auth import AuthHTTPError, admin_router
from app.api.auth import router as auth_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.search import router as search_router
from app.api.system import router as system_router
from app.core.exceptions import (
    RAGSystemError,
)


def create_app() -> FastAPI:
    """FastAPIアプリケーションを作成する関数

    Returns:
        FastAPI: 設定済みのFastAPIアプリケーション
    """
    app = FastAPI(
        title="RAG System API",
        version="1.0.0",
        description="RAG (Retrieval-Augmented Generation) System API",
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORSミドルウェアの設定
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 本番環境では適切に制限する
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # デバッグモードの設定（開発環境）
    app.debug = True

    # テスト環境では認証ミドルウェアをスキップ
    import os

    if os.getenv("TESTING") != "true":
        # 本番環境では認証ミドルウェアを追加
        pass  # ここに認証ミドルウェアを追加

    # ルーターの登録
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(search_router)
    app.include_router(system_router)
    app.include_router(auth_router)
    app.include_router(admin_router)

    # エラーハンドラーの登録
    setup_error_handlers(app)

    return app


def setup_error_handlers(app: FastAPI) -> None:
    """エラーハンドラーを設定する"""

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "NOT_FOUND",
                    "message": "Requested resource not found",
                    "type": "not_found",
                },
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": str(uuid.uuid4()),
            },
        )

    @app.exception_handler(405)
    async def method_not_allowed_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=405,
            content={
                "error": {
                    "code": "METHOD_NOT_ALLOWED",
                    "message": "Method not allowed for this endpoint",
                    "type": "method_not_allowed",
                },
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": str(uuid.uuid4()),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Validation error",
                    "type": "validation_error",
                    "details": exc.errors(),
                },
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": str(uuid.uuid4()),
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        # 認証関連のHTTPExceptionの場合
        if exc.status_code == 401:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "AUTHENTICATION_ERROR",
                        "message": str(exc.detail),
                        "type": "authentication",
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": str(uuid.uuid4()),
                },
            )
        # 認可関連のHTTPExceptionの場合
        elif exc.status_code == 403:
            return JSONResponse(
                status_code=403,
                content={
                    "error": {
                        "code": "AUTHORIZATION_ERROR",
                        "message": str(exc.detail),
                        "type": "authorization",
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": str(uuid.uuid4()),
                },
            )
        # その他のHTTPExceptionはデフォルト処理
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "HTTP_ERROR",
                    "message": str(exc.detail),
                    "type": "http_error",
                },
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": str(uuid.uuid4()),
            },
        )

    @app.exception_handler(SQLAlchemyError)
    async def database_exception_handler(request: Request, exc: SQLAlchemyError):
        if isinstance(exc, IntegrityError):
            return JSONResponse(
                status_code=409,
                content={
                    "error": {
                        "code": "CONSTRAINT_VIOLATION",
                        "message": "Database constraint violation",
                        "type": "constraint_violation",
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": str(uuid.uuid4()),
                },
            )

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "DATABASE_ERROR",
                    "message": "Database operation failed",
                    "type": "database_error",
                },
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": str(uuid.uuid4()),
            },
        )

    @app.exception_handler(DBException)
    async def vector_database_exception_handler(request: Request, exc: DBException):
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "VECTOR_DATABASE_ERROR",
                    "message": "Vector database operation failed",
                    "type": "vector_database_error",
                },
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": str(uuid.uuid4()),
            },
        )

    @app.exception_handler(AuthHTTPError)
    async def authentication_exception_handler(request: Request, exc: AuthHTTPError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": str(exc.detail),
                    "type": exc.error_type,
                },
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": str(uuid.uuid4()),
            },
        )

    @app.exception_handler(RAGSystemError)
    async def rag_system_exception_handler(request: Request, exc: RAGSystemError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": str(exc),
                    "type": "rag_system_error",
                },
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": str(uuid.uuid4()),
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "Internal server error",
                    "type": "internal_server_error",
                },
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": str(uuid.uuid4()),
            },
        )


app = create_app()


@app.get("/")
def read_root() -> dict[str, str]:
    """ルートエンドポイントのハンドラー。

    Returns:
        dict[str, str]: 'Hello, World!'メッセージを含む辞書。
    """
    return {"message": "Hello, World!"}


@app.options("/")
def options_root():
    """ルートエンドポイントのOPTIONSメソッド（CORS対応）"""
    return {"message": "OK"}
