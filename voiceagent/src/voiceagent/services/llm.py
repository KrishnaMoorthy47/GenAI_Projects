# Copyright (c) 2024 ValGenesis Inc. All rights reserved.
"""LLM generation via Groq (llama-3.1-8b-instant) with sentence-level streaming."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from functools import lru_cache

from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_exponential

from voiceagent.config import get_settings

# Sentence-boundary regex: split on . ! ? followed by space or end-of-string
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@lru_cache
def get_client() -> AsyncGroq:
    settings = get_settings()
    return AsyncGroq(api_key=settings.groq_api_key or None)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4), reraise=True)
async def stream_sentences(
    messages: list[dict[str, str]],
    model: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> AsyncIterator[str]:
    """
    Stream LLM response as complete sentences.

    Buffers token deltas and yields each time a sentence boundary is detected.
    This minimises TTS latency compared to waiting for the full response.

    Yields:
        Complete sentences (strings) one at a time.
    """
    settings = get_settings()
    effective_model = model or settings.groq_model
    client = get_client()
    stream = await client.chat.completions.create(
        model=effective_model,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    buffer = ""
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta is None:
            continue
        buffer += delta

        # Yield complete sentences as they form
        parts = _SENTENCE_RE.split(buffer)
        # Keep the last (potentially incomplete) fragment in the buffer
        for sentence in parts[:-1]:
            sentence = sentence.strip()
            if sentence:
                yield sentence
        buffer = parts[-1]

    # Yield any remaining text
    remainder = buffer.strip()
    if remainder:
        yield remainder


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4), reraise=True)
async def generate(
    messages: list[dict[str, str]],
    model: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Generate a full response (non-streaming). Used in tests."""
    settings = get_settings()
    effective_model = model or settings.groq_model
    client = get_client()
    response = await client.chat.completions.create(
        model=effective_model,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )
    return response.choices[0].message.content or ""
