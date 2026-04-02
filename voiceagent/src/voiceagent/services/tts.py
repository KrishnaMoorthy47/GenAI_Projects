# Copyright (c) 2024 ValGenesis Inc. All rights reserved.
"""Text-to-speech via OpenAI TTS API."""

from __future__ import annotations

import base64
from functools import lru_cache

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from voiceagent.config import get_settings


@lru_cache
def get_client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(api_key=settings.openai_api_key or None)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4), reraise=True)
async def synthesize(text: str, voice: str | None = None, model: str | None = None) -> bytes:
    """
    Convert text to speech using OpenAI TTS.

    Args:
        text: The text to synthesize.
        voice: Voice ID (alloy, echo, fable, onyx, nova, shimmer). Defaults to config value.
        model: TTS model (tts-1 for speed, tts-1-hd for quality). Defaults to config value.

    Returns:
        Raw MP3 audio bytes.
    """
    settings = get_settings()
    client = get_client()
    response = await client.audio.speech.create(
        model=model or settings.tts_model,
        voice=voice or settings.tts_voice,  # type: ignore[arg-type]
        input=text,
        response_format="mp3",
    )
    return response.content


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4), reraise=True)
async def synthesize_base64(
    text: str, voice: str | None = None, model: str | None = None
) -> str:
    """Synthesize speech and return base64-encoded MP3."""
    audio_bytes = await synthesize(text, voice=voice, model=model)
    return base64.b64encode(audio_bytes).decode("utf-8")
