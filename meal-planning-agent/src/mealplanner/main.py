from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))  # GenAI_Projects root

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rate_limit import RateLimitMiddleware
from mealplanner.api.health import router as health_router
from mealplanner.api.meals import router as meals_router
from mealplanner.services.mongodb import close_mongodb, init_mongodb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Meal Planner starting up...")
    await init_mongodb()
    logger.info("Meal Planner ready on port 8006")
    yield
    logger.info("Meal Planner shutting down...")
    await close_mongodb()
    logger.info("Meal Planner shutdown complete")


app = FastAPI(
    title="Meal Planning Agent",
    description=(
        "AI-powered family meal planner using a 4-agent LangGraph pipeline. "
        "Generates weekly dinner plans tailored for children and adults, "
        "with a categorised grocery list. Backed by MongoDB Atlas."
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
app.include_router(meals_router)
