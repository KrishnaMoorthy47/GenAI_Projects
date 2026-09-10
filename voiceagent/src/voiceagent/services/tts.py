"""Text-to-speech via Groq's Orpheus API (OpenAI-compatible endpoint)."""

from __future__ import annotations

import base64
from functools import lru_cache

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from voiceagent.config import get_settings


@lru_cache
def get_client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(api_key=settings.groq_api_key or None, base_url="https://api.groq.com/openai/v1")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4), reraise=True)
async def synthesize(text: str, voice: str | None = None, model: str | None = None) -> bytes:
    """
    Convert text to speech using Groq's Orpheus TTS.

    Args:
        text: The text to synthesize.
        voice: Voice ID (autumn, diana, hannah, austin, daniel, troy). Defaults to config value.
        model: TTS model. Defaults to config value.

    Returns:
        Raw WAV audio bytes — Orpheus only supports "wav", unlike OpenAI's "mp3".
    """
    settings = get_settings()
    client = get_client()
    response = await client.audio.speech.create(
        model=model or settings.tts_model,
        voice=voice or settings.tts_voice,  # type: ignore[arg-type]
        input=text,
        response_format="wav",
    )
    return response.content


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4), reraise=True)
async def synthesize_base64(
    text: str, voice: str | None = None, model: str | None = None
) -> str:
    """Synthesize speech and return base64-encoded WAV."""
    audio_bytes = await synthesize(text, voice=voice, model=model)
    return base64.b64encode(audio_bytes).decode("utf-8")
