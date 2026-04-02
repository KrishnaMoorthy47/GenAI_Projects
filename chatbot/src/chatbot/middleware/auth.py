from __future__ import annotations

from fastapi import Header, HTTPException

from chatbot.config import get_settings


async def verify_api_key(x_api_key: str = Header(...)) -> str:
    """FastAPI dependency — validate the x-api-key header."""
    settings = get_settings()
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key
