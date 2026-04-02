from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))  # GenAI_Projects root

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rate_limit import RateLimitMiddleware
from jobsearch.api.health import router as health_router
from jobsearch.api.jobs import router as jobs_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Job Search Agent starting up...")
    logger.info("Job Search Agent ready on port 8007")
    yield
    logger.info("Job Search Agent shutdown complete")


app = FastAPI(
    title="Job Search Agent",
    description=(
        "AI-powered job search using browser automation (browser-use). "
        "Upload your CV and select target companies — parallel browser agents search "
        "each company's career site, extract job listings, and score each against your CV profile "
        "using an LLM. Results ranked by fit score."
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
app.include_router(jobs_router)
