"""FastAPI application entry point for agenteval."""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))  # F:\Sample root

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles

from agenteval.api import agent_routes, eval_routes, health
from agenteval.config import get_settings
from rate_limit import RateLimitMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _configure_langsmith() -> None:
    """Set LangSmith env vars so the SDK picks them up at import time."""
    settings = get_settings()
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    if settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
        logger.info("LangSmith tracing enabled → project: %s", settings.langsmith_project)
    else:
        os.environ.setdefault("LANGSMITH_TRACING", "false")


_configure_langsmith()

# Auth
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    settings = get_settings()
    if api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown."""
    settings = get_settings()
    logger.info("agenteval starting on port %d (LLM: %s)", settings.port, settings.llm_provider)

    # In-memory store for eval run reports (bounded — eviction handled in eval_routes)
    app.state.report_store = {}

    # Validate DB path is accessible
    import sqlite3
    from pathlib import Path

    db_path = Path(settings.db_path)
    if db_path.exists():
        try:
            with sqlite3.connect(str(db_path)) as conn:
                tables = conn.execute(
                    "SELECT count(*) FROM sqlite_master WHERE type='table'"
                ).fetchone()[0]
            logger.info("Tamil Songs DB ready: %s (%d tables)", db_path, tables)
        except Exception as e:
            logger.warning("DB check failed: %s", e)
    else:
        logger.warning("DB not found at %s — agent tools will fail until DB is present", db_path)

    yield
    logger.info("agenteval shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AgentEval",
        description="AI agent evaluation framework for a LangGraph SQL agent",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(RateLimitMiddleware, max_requests=20, window=60)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Auth dependency applied to all non-health routes
    app.include_router(health.router)
    app.include_router(
        agent_routes.router,
        dependencies=[Depends(verify_api_key)],
    )
    app.include_router(
        eval_routes.router,
        dependencies=[Depends(verify_api_key)],
    )

    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/ui")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("agenteval.main:app", host="0.0.0.0", port=settings.port, reload=True)
