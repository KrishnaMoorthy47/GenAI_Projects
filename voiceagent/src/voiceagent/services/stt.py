# Copyright (c) 2024 ValGenesis Inc. All rights reserved.
"""Speech-to-text via OpenAI Whisper API."""

from __future__ import annotations

import io
import struct
import wave
from functools import lru_cache

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from voiceagent.config import get_settings


@lru_cache
def get_client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(api_key=settings.openai_api_key or None)


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw PCM bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4), reraise=True)
async def transcribe(audio_bytes: bytes, is_wav: bool = False) -> str:
    """
    Transcribe audio bytes using Whisper.

    Args:
        audio_bytes: Raw PCM bytes (16kHz, 16-bit, mono) OR a WAV file.
        is_wav: If True, treat audio_bytes as a complete WAV file.

    Returns:
        Transcribed text, or empty string if nothing detected.
    """
    if not is_wav:
        wav_bytes = pcm_to_wav(audio_bytes)
    else:
        wav_bytes = audio_bytes

    file_tuple = ("audio.wav", io.BytesIO(wav_bytes), "audio/wav")
    client = get_client()

    response = await client.audio.transcriptions.create(
        model="whisper-1",
        file=file_tuple,
        response_format="text",
    )
    # response is a plain string when response_format="text"
    return response.strip() if response else ""
