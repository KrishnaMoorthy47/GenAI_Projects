from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

import redis as redis_lib

from chatbot.config import get_settings

logger = logging.getLogger(__name__)

# Module-level Redis client (None = not connected)
_redis: Optional[redis_lib.Redis] = None
_redis_available: bool = False

# In-memory fallback: session_id → list of message dicts
_memory_store: Dict[str, List[dict]] = {}


def init_redis() -> bool:
    """
    Attempt to connect to Redis. Returns True if successful.
    Falls back to in-memory dict silently if Redis is unavailable.
    """
    global _redis, _redis_available
    settings = get_settings()

    try:
        pool = redis_lib.ConnectionPool(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password or None,
            max_connections=10,
            socket_connect_timeout=3,
            decode_responses=True,
        )
        _redis = redis_lib.Redis(connection_pool=pool)
        _redis.ping()
        _redis_available = True
        logger.info("Redis connected: %s:%d", settings.redis_host, settings.redis_port)
        return True
    except Exception as exc:
        logger.warning("Redis unavailable (%s). Using in-memory history (not persistent).", exc)
        _redis_available = False
        return False


def is_redis_connected() -> bool:
    return _redis_available


def _make_key(session_id: str) -> str:
    return f"chat_history:{session_id}"


def get_history(session_id: str) -> List[dict]:
    """Retrieve the last MAX_HISTORY_TURNS*2 messages for a session."""
    settings = get_settings()
    max_msgs = settings.max_history_turns * 2

    if _redis_available and _redis:
        try:
            raw = _redis.get(_make_key(session_id))
            if raw:
                messages = json.loads(raw)
                return messages[-max_msgs:]
            return []
        except Exception as exc:
            logger.warning("Redis get_history failed: %s", exc)

    # In-memory fallback
    messages = _memory_store.get(session_id, [])
    return messages[-max_msgs:]


def save_history(session_id: str, question: str, answer: str) -> None:
    """Append a new user+assistant turn to the session history."""
    settings = get_settings()
    max_msgs = settings.max_history_turns * 2

    current = get_history(session_id)
    current.append({"role": "user", "content": question})
    current.append({"role": "assistant", "content": answer})

    # Trim to max
    trimmed = current[-max_msgs:]

    if _redis_available and _redis:
        try:
            _redis.setex(
                _make_key(session_id),
                settings.cache_ttl,
                json.dumps(trimmed),
            )
            return
        except Exception as exc:
            logger.warning("Redis save_history failed: %s", exc)

    # In-memory fallback
    _memory_store[session_id] = trimmed


def clear_history(session_id: str) -> None:
    """Delete all history for a session."""
    if _redis_available and _redis:
        try:
            _redis.delete(_make_key(session_id))
        except Exception as exc:
            logger.warning("Redis clear_history failed: %s", exc)
    _memory_store.pop(session_id, None)
