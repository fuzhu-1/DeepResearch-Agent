"""FastAPI middleware for logging, CORS, error handling, and request ID tracing."""

import logging
import time
import uuid
from typing import Callable, Awaitable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.utils.logger import get_request_id, set_request_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that injects a unique request_id into every request.

    The request_id is propagated via contextvars so all downstream
    async tasks (including background research) can log with it.
    Responds with X-Request-ID header for client-side tracing.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable]
    ):
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        set_request_id(request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs each request with method, path, status, and duration."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable]
    ):
        start_time = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response


# ---------------------------------------------------------------------------
# Error handling middleware
# ---------------------------------------------------------------------------


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware that catches unhandled exceptions and returns 500 JSON."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable]
    ):
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            logger.exception("Unhandled exception processing %s %s", request.method, request.url.path)
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "error": str(exc),
                },
            )


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def register_middleware(app: FastAPI) -> None:
    """Register all middleware on a FastAPI application.

    Order matters: error handling wraps everything, request logging
    wraps the app logic, and CORS is applied at the ASGI level.

    Args:
        app: The FastAPI application instance.
    """
    # CORS – allow frontend dev server on port 5173
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID (outermost after error handling, so even error responses get IDs)
    app.add_middleware(RequestIDMiddleware)

    # Request logging
    app.add_middleware(RequestLoggingMiddleware)

    # Error handling (outermost, catches everything)
    app.add_middleware(ErrorHandlingMiddleware)
