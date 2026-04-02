from __future__ import annotations

import asyncio
import logging
import sys
import warnings
from contextlib import asynccontextmanager

# psycopg3 async requires SelectorEventLoop on Windows (not the default ProactorEventLoop)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))  # GenAI_Projects root → rate_limit.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rate_limit import RateLimitMiddleware
from finagent.api.health import router as health_router
from finagent.api.research import router as research_router
from finagent.services.checkpointer import close_checkpointer, init_checkpointer

# LangChain sets AIMessage.parsed to a Pydantic model instance after structured output,
# but Pydantic's serializer expects None there. The warning is harmless — suppress it.
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
    # Startup
    logger.info("FinAgent starting up...")
    await init_checkpointer()
    logger.info("FinAgent ready")
    yield
    # Shutdown
    logger.info("FinAgent shutting down...")
    await close_checkpointer()
    logger.info("FinAgent shutdown complete")


app = FastAPI(
    title="FinAgent",
    description=(
        "AI-powered investment research using a LangGraph multi-agent system. "
        "Autonomously researches stocks and generates structured investment briefs "
        "with human-in-the-loop approval."
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
app.include_router(research_router)
