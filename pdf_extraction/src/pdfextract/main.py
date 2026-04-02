from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))  # GenAI_Projects root

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rate_limit import RateLimitMiddleware
from pdfextract.api.extract import router as extract_router
from pdfextract.api.health import router as health_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("PDF Extraction service starting up...")
    logger.info("PDF Extraction service ready on port 8008")
    yield
    logger.info("PDF Extraction service shutdown complete")


app = FastAPI(
    title="PDF Extraction",
    description=(
        "AI-powered document extraction using LLM Whisperer OCR and structured LLM parsing. "
        "Upload invoices or purchase orders to extract header info and product line items "
        "into a structured JSON schema with spell correction and metadata generation."
    ),
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

app.include_router(health_router)
app.include_router(extract_router)
