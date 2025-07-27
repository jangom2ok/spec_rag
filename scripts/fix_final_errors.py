#!/usr/bin/env python3
"""
Final fix for Cursor errors - fix main.py structure
"""

from pathlib import Path


def fix_main_py_structure():
    """Fix main.py by restructuring exception classes and handlers"""
    content = '''import uuid
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.admin import router as admin_router
from app.api.auth import router as auth_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.search import router as search_router
from app.api.system import router as system_router
from app.core.config import settings
from app.core.middleware import (
    ErrorHandlingMiddleware,
    add_security_headers,
    log_requests,
)

# Handle DBException import
try:
    from aperturedb import DBException
except ImportError:
    class DBException(Exception):
        """Mock ApertureDB exception."""
        pass


# Custom exception classes
class DatabaseException(Exception):
    """Database-related exception."""
    pass


class VectorDatabaseException(Exception):
    """Vector database exception."""
    pass


class AuthenticationException(Exception):
    """Authentication exception."""
    pass


class RAGSystemException(Exception):
    """RAG system exception."""
    pass


# Exception handlers
async def database_exception_handler(request: Request, exc: DatabaseException) -> JSONResponse:
    """Handle database exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "database_error",
                "message": str(exc),
                "code": "DB_ERROR",
            },
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4()),
        }
    )


async def vector_database_exception_handler(request: Request, exc: VectorDatabaseException) -> JSONResponse:
    """Handle vector database exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "vector_db_error",
                "message": str(exc),
                "code": "VECTOR_DB_ERROR",
            },
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4()),
        }
    )


async def authentication_exception_handler(request: Request, exc: AuthenticationException) -> JSONResponse:
    """Handle authentication exceptions."""
    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "type": "authentication_error",
                "message": str(exc),
                "code": "AUTH_ERROR",
            },
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4()),
        }
    )


async def rag_system_exception_handler(request: Request, exc: RAGSystemException) -> JSONResponse:
    """Handle RAG system exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "rag_system_error",
                "message": str(exc),
                "code": "RAG_ERROR",
            },
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4()),
        }
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "internal_error",
                "message": "An unexpected error occurred",
                "code": "INTERNAL_ERROR",
            },
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4()),
        }
    )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        openapi_url="/api/v1/openapi.json" if not settings.PRODUCTION else None,
        docs_url="/docs" if not settings.PRODUCTION else None,
        redoc_url="/redoc" if not settings.PRODUCTION else None,
    )

    # CORS設定
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # カスタムミドルウェア
    app.add_middleware(ErrorHandlingMiddleware)
    app.middleware("http")(add_security_headers)
    app.middleware("http")(log_requests)

    # ルーターの登録
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(search_router)
    app.include_router(system_router)
    app.include_router(auth_router)
    app.include_router(admin_router)

    # エラーハンドラーの登録
    setup_error_handlers(app)

    # Register custom exception handlers
    app.add_exception_handler(DatabaseException, database_exception_handler)
    app.add_exception_handler(VectorDatabaseException, vector_database_exception_handler)
    app.add_exception_handler(AuthenticationException, authentication_exception_handler)
    app.add_exception_handler(RAGSystemException, rag_system_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    return app


def setup_error_handlers(app: FastAPI) -> None:
    """エラーハンドラーを設定する"""

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: HTTPException) -> JSONResponse:
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
    async def method_not_allowed_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
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
    ) -> JSONResponse:
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
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
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


# Create the app instance
app = create_app()
'''

    file_path = Path(__file__).parent.parent / "app" / "main.py"
    file_path.write_text(content, encoding="utf-8")
    print(f"Restructured {file_path}")


def main():
    """Main function"""
    fix_main_py_structure()
    print("Fixed main.py structure!")


if __name__ == "__main__":
    main()
