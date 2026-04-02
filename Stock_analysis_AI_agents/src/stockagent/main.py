from __future__ import annotations

import asyncio
import logging
import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

# psycopg3 async requires SelectorEventLoop on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.insert(0, str(Path(__file__).parents[3]))                    # GenAI_Projects/ (rate_limit)
sys.path.insert(0, str(Path(__file__).parents[3] / "finagent"))       # finagent/ (app.* imports)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.checkpointer import close_checkpointer, init_checkpointer
from rate_limit import RateLimitMiddleware
from stockagent.api.analyze import router as analyze_router
from stockagent.api.health import router as health_router

warnings.filterwarnings(
    "ignore",
    message=".*PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Stock Intelligence starting up...")
    await init_checkpointer()
    logger.info("Stock Intelligence ready on port 8005 (LangGraph + CrewAI)")
    yield
    logger.info("Stock Intelligence shutting down...")
    await close_checkpointer()
    logger.info("Stock Intelligence shutdown complete")


app = FastAPI(
    title="Stock Intelligence",
    description=(
        "Dual-engine stock analysis service. "
        "Choose LangGraph for human-in-the-loop interactive research with approval gate, "
        "or CrewAI for fully automated 4-agent analysis with BUY/HOLD/SELL recommendation. "
        "Both engines use yfinance for real market data and Groq for LLM reasoning."
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
app.include_router(analyze_router)
