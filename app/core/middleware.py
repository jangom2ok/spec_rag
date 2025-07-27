"""Custom middleware"""

import time
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Error handling middleware."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle errors in requests."""
        # Just pass through - let exception handlers deal with errors
        response = await call_next(request)
        return response


async def add_security_headers(request: Request, call_next: Callable) -> Response:
    """Add security headers to responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


async def log_requests(request: Request, call_next: Callable) -> Response:
    """Log all requests."""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Add request ID to request state
    request.state.request_id = request_id

    response = await call_next(request)

    process_time = time.time() - start_time
    print(
        f"Request {request_id} - {request.method} {request.url.path} - {process_time:.3f}s"
    )

    return response
