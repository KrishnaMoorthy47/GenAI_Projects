from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Union

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from finagent.config import get_settings

logger = logging.getLogger(__name__)

_pool: AsyncConnectionPool | None = None
_checkpointer: Union[AsyncPostgresSaver, MemorySaver, None] = None


async def init_checkpointer() -> Union[AsyncPostgresSaver, MemorySaver]:
    """Initialize the checkpointer. Call once at startup.

    When USE_MEMORY_SAVER=true (local testing), uses an in-process MemorySaver.
    Otherwise, opens a Postgres connection pool and returns AsyncPostgresSaver.
    """
    global _pool, _checkpointer

    settings = get_settings()

    if settings.use_memory_saver:
        # Local testing mode — no Postgres required, state lives in-process
        _checkpointer = MemorySaver()
        logger.info("Using MemorySaver (local testing mode — state not persisted across restarts)")
        return _checkpointer

    if not settings.database_url:
        raise ValueError("DATABASE_URL is not set. Add it to .env or set USE_MEMORY_SAVER=true.")

    # autocommit=True is required by AsyncPostgresSaver — omitting causes InFailedSqlTransaction
    _pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        max_size=10,
        kwargs={"autocommit": True},
        open=False,
    )
    await _pool.open()
    logger.info("Postgres connection pool opened (Supabase)")

    _checkpointer = AsyncPostgresSaver(_pool)
    await _checkpointer.setup()
    logger.info("LangGraph AsyncPostgresSaver initialized")

    return _checkpointer


async def close_checkpointer() -> None:
    """Close the Postgres connection pool. Call at shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        logger.info("Postgres connection pool closed")


def get_checkpointer() -> Union[AsyncPostgresSaver, MemorySaver]:
    if _checkpointer is None:
        raise RuntimeError("Checkpointer not initialized. Call init_checkpointer() first.")
    return _checkpointer
