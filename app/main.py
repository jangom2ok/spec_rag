import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.api.admin import router as admin_router
from app.api.auth import router as auth_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.search import router as search_router
from app.api.system import router as system_router
from app.core.config import settings
from app.core.exceptions import RAGSystemError
from app.core.middleware import (
    ErrorHandlingMiddleware,
    add_security_headers,
    log_requests,
)

# Handle DBException import
if TYPE_CHECKING:
    # For type checking, just use Exception as the base type
    DBException = Exception
else:
    try:
        from aperturedb import DBException  # type: ignore
    except ImportError:
        # Use the mock DBException when aperturedb is not available
        from app.models.aperturedb_mock import DBException  # type: ignore


# Custom exception classes
class DatabaseException(Exception):  # noqa: N818
    """Database-related exception."""

    pass


class VectorDatabaseException(Exception):  # noqa: N818
    """Vector database exception."""

    pass


class AuthenticationException(Exception):  # noqa: N818
    """Authentication exception."""

    pass


class RAGSystemException(Exception):  # noqa: N818
    """RAG system exception."""

    pass


# Exception handlers
async def database_exception_handler(request: Request, exc: Exception) -> JSONResponse:
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
        },
    )


async def vector_database_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
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
        },
    )


async def authentication_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
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
        },
    )


async def rag_system_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
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
        },
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "internal_server_error",
                "message": "Internal server error",
                "code": "INTERNAL_SERVER_ERROR",
            },
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4()),
        },
    )


async def integrity_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle database integrity errors."""
    return JSONResponse(
        status_code=409,
        content={
            "error": {
                "type": "constraint_violation",
                "message": "Database constraint violation",
                "code": "CONSTRAINT_VIOLATION",
            },
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4()),
        },
    )


async def sqlalchemy_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general SQLAlchemy errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "database_error",
                "message": "Database operation failed",
                "code": "DATABASE_ERROR",
            },
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4()),
        },
    )


async def db_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle ApertureDB exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "vector_database_error",
                "message": "Vector database operation failed",
                "code": "VECTOR_DATABASE_ERROR",
            },
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4()),
        },
    )


async def rag_system_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle RAG system errors."""
    return JSONResponse(
        status_code=getattr(exc, "status_code", 503),
        content={
            "error": {
                "type": "rag_system_error",
                "message": str(exc),
                "code": getattr(exc, "error_code", "RAG_SYSTEM_ERROR"),
            },
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4()),
        },
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

    # ルートエンドポイント
    @app.get("/")
    async def root():
        """Root endpoint for health check."""
        return {"message": "Hello, World!"}

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
    app.add_exception_handler(
        VectorDatabaseException, vector_database_exception_handler
    )
    app.add_exception_handler(AuthenticationException, authentication_exception_handler)
    app.add_exception_handler(RAGSystemException, rag_system_exception_handler)

    # Add handlers for SQLAlchemy exceptions
    app.add_exception_handler(IntegrityError, integrity_error_handler)
    app.add_exception_handler(SQLAlchemyError, sqlalchemy_error_handler)

    # Add handler for DBException
    app.add_exception_handler(DBException, db_exception_handler)

    # Add handler for RAGSystemError
    app.add_exception_handler(RAGSystemError, rag_system_error_handler)

    app.add_exception_handler(Exception, general_exception_handler)

    return app


def setup_error_handlers(app: FastAPI) -> None:
    """エラーハンドラーを設定する"""

    @app.exception_handler(404)
    async def not_found_handler(  # type: ignore[reportUnusedFunction]
        request: Request, exc: HTTPException
    ) -> JSONResponse:  # noqa: F841
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
    async def method_not_allowed_handler(  # type: ignore[reportUnusedFunction]  # noqa: F841
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
    async def validation_exception_handler(  # type: ignore[reportUnusedFunction]  # noqa: F841
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
    async def http_exception_handler(  # type: ignore[reportUnusedFunction]  # noqa: F841
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
