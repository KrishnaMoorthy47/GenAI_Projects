"""Tests for the LLM service (Groq streaming)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voiceagent.services.llm import generate, stream_sentences


def _make_chunk(content: str | None):
    """Build a mock Groq streaming chunk."""
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock()
    chunk.choices[0].delta.content = content
    return chunk


async def _async_iter(items):
    """Helper to create an async iterable from a list."""
    for item in items:
        yield item


# ── stream_sentences ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_sentences_single_sentence():
    """Single complete sentence should be yielded as-is."""
    mock_client = MagicMock()
    mock_stream = _async_iter([
        _make_chunk("Hello world."),
        _make_chunk(None),
    ])
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    with patch("voiceagent.services.llm.get_client", return_value=mock_client):
        sentences = []
        async for s in stream_sentences([{"role": "user", "content": "Hi"}]):
            sentences.append(s)

    assert len(sentences) == 1
    assert sentences[0] == "Hello world."


@pytest.mark.asyncio
async def test_stream_sentences_multiple_sentences():
    """Multiple sentences in the stream should each be yielded separately."""
    mock_client = MagicMock()
    # Deliver as separate chunks to simulate real streaming
    mock_stream = _async_iter([
        _make_chunk("Hello world. "),
        _make_chunk("How are you? "),
        _make_chunk("I'm fine."),
    ])
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    with patch("voiceagent.services.llm.get_client", return_value=mock_client):
        sentences = []
        async for s in stream_sentences([{"role": "user", "content": "Hi"}]):
            sentences.append(s)

    # Should have at least 2 sentences split on boundaries
    assert len(sentences) >= 2
    full = " ".join(sentences)
    assert "Hello world" in full
    assert "How are you" in full


@pytest.mark.asyncio
async def test_stream_sentences_yields_remainder():
    """Trailing text without sentence boundary should still be yielded."""
    mock_client = MagicMock()
    mock_stream = _async_iter([
        _make_chunk("This has no punctuation"),
    ])
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    with patch("voiceagent.services.llm.get_client", return_value=mock_client):
        sentences = []
        async for s in stream_sentences([{"role": "user", "content": "Tell me"}]):
            sentences.append(s)

    assert len(sentences) == 1
    assert sentences[0] == "This has no punctuation"


@pytest.mark.asyncio
async def test_stream_sentences_skips_none_chunks():
    """None content in delta should be silently skipped."""
    mock_client = MagicMock()
    mock_stream = _async_iter([
        _make_chunk(None),
        _make_chunk("Real content."),
        _make_chunk(None),
    ])
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    with patch("voiceagent.services.llm.get_client", return_value=mock_client):
        sentences = []
        async for s in stream_sentences([{"role": "user", "content": "Hi"}]):
            sentences.append(s)

    assert sentences == ["Real content."]


# ── generate ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_returns_content():
    """generate() should return the full message content."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "The answer is 42."
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    with patch("voiceagent.services.llm.get_client", return_value=mock_client):
        result = await generate([{"role": "user", "content": "What is 6x7?"}])

    assert result == "The answer is 42."


@pytest.mark.asyncio
async def test_generate_handles_none_content():
    """generate() should return empty string when content is None."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = None
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    with patch("voiceagent.services.llm.get_client", return_value=mock_client):
        result = await generate([{"role": "user", "content": "Hi"}])

    assert result == ""
