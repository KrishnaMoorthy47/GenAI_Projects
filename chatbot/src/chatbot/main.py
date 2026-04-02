from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))  # F:\Sample root

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chatbot.adapters import cache_adapter, vector_adapter
from rate_limit import RateLimitMiddleware
from chatbot.config import get_settings
from chatbot.routers.query_router import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    settings = get_settings()
    logger.info("Personal RAG Chatbot starting up (env=%s)...", settings.app_env)

    # Connect to Redis (falls back to in-memory if unavailable)
    cache_adapter.init_redis()

    # Pre-load the default FAISS index into memory
    loaded = vector_adapter.load_index(settings.data_indexed_dir, settings.default_collection)
    if not loaded:
        logger.warning(
            "FAISS index not found. Start the app then run: python scripts/ingest.py"
        )

    logger.info("Personal RAG Chatbot ready on port %d", settings.app_port)
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Personal RAG Chatbot shutting down...")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Personal RAG Chatbot",
        description=(
            "Answer questions over your own documents using Retrieval-Augmented Generation. "
            "Upload PDFs/text/HTML to data/input/, run the ingestion script, then query via API."
        ),
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(RateLimitMiddleware, max_requests=30, window=60)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
