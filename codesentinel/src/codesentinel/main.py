from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))  # GenAI_Projects root → rate_limit.py

from pathlib import Path as _Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from rate_limit import RateLimitMiddleware
from codesentinel.api.health import router as health_router
from codesentinel.api.review import router as review_router
from codesentinel.api.webhook import router as webhook_router
from codesentinel.services.review_store import close_db, init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("CodeSentinel starting up...")
    await init_db()
    logger.info("CodeSentinel ready")
    yield
    # Shutdown
    logger.info("CodeSentinel shutting down...")
    await close_db()
    logger.info("CodeSentinel shutdown complete")


app = FastAPI(
    title="CodeSentinel",
    description=(
        "AI-powered GitHub PR reviewer that detects security vulnerabilities and code quality issues. "
        "Triggered via GitHub webhook or REST API. Posts structured review comments to PRs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(RateLimitMiddleware, max_requests=30, window=60)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(review_router)
app.include_router(webhook_router)

_static_dir = _Path(__file__).parent / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/ui", StaticFiles(directory=str(_static_dir), html=True), name="ui")
