from __future__ import annotations

import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter (in-process, per client IP).

    No external dependencies — uses only Starlette (already a FastAPI dependency).

    Args:
        max_requests: Maximum requests allowed within the window.
        window:       Window duration in seconds.
        exempt_paths: Set of exact paths that bypass rate limiting.
                      Defaults to health/docs endpoints.

    Usage:
        app.add_middleware(RateLimitMiddleware, max_requests=20, window=60)
    """

    _DEFAULT_EXEMPT = frozenset({
        "/health", "/docs", "/redoc", "/openapi.json",
    })

    def __init__(
        self,
        app,
        max_requests: int = 20,
        window: int = 60,
        exempt_paths: frozenset[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._max = max_requests
        self._window = window
        self._exempt = exempt_paths if exempt_paths is not None else self._DEFAULT_EXEMPT
        self._hits: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if path in self._exempt:
            return await call_next(request)

        ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Evict timestamps that have fallen outside the current window
        self._hits[ip] = [t for t in self._hits[ip] if now - t < self._window]

        if len(self._hits[ip]) >= self._max:
            retry_after = int(self._window - (now - self._hits[ip][0]))
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": str(max(retry_after, 1))},
            )

        self._hits[ip].append(now)
        return await call_next(request)
