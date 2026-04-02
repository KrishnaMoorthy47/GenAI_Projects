"""Tests for the TTS service (OpenAI speech)."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voiceagent.services.tts import synthesize, synthesize_base64


# ── synthesize ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_synthesize_returns_bytes():
    """synthesize() should return audio bytes from the OpenAI response."""
    fake_audio = b"fake_mp3_bytes"
    mock_response = MagicMock()
    mock_response.content = fake_audio

    mock_client = MagicMock()
    mock_client.audio.speech.create = AsyncMock(return_value=mock_response)

    with patch("voiceagent.services.tts.get_client", return_value=mock_client):
        result = await synthesize("Hello world")

    assert result == fake_audio


@pytest.mark.asyncio
async def test_synthesize_passes_text_and_voice():
    """synthesize() should forward text and voice parameters to the API."""
    fake_audio = b"audio"
    mock_response = MagicMock()
    mock_response.content = fake_audio

    mock_client = MagicMock()
    mock_client.audio.speech.create = AsyncMock(return_value=mock_response)

    with patch("voiceagent.services.tts.get_client", return_value=mock_client):
        await synthesize("Test sentence", voice="nova", model="tts-1")

    call_kwargs = mock_client.audio.speech.create.call_args.kwargs
    assert call_kwargs["input"] == "Test sentence"
    assert call_kwargs["voice"] == "nova"
    assert call_kwargs["model"] == "tts-1"


@pytest.mark.asyncio
async def test_synthesize_uses_mp3_format():
    """synthesize() should request MP3 format from the API."""
    mock_response = MagicMock()
    mock_response.content = b"mp3"

    mock_client = MagicMock()
    mock_client.audio.speech.create = AsyncMock(return_value=mock_response)

    with patch("voiceagent.services.tts.get_client", return_value=mock_client):
        await synthesize("Hi")

    call_kwargs = mock_client.audio.speech.create.call_args.kwargs
    assert call_kwargs["response_format"] == "mp3"


# ── synthesize_base64 ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_synthesize_base64_encodes_correctly():
    """synthesize_base64() should return valid base64-encoded audio."""
    fake_audio = b"hello_audio_bytes"
    expected_b64 = base64.b64encode(fake_audio).decode("utf-8")

    mock_response = MagicMock()
    mock_response.content = fake_audio

    mock_client = MagicMock()
    mock_client.audio.speech.create = AsyncMock(return_value=mock_response)

    with patch("voiceagent.services.tts.get_client", return_value=mock_client):
        result = await synthesize_base64("Hello")

    assert result == expected_b64


@pytest.mark.asyncio
async def test_synthesize_base64_returns_string():
    """synthesize_base64() should return a string, not bytes."""
    mock_response = MagicMock()
    mock_response.content = b"audio"

    mock_client = MagicMock()
    mock_client.audio.speech.create = AsyncMock(return_value=mock_response)

    with patch("voiceagent.services.tts.get_client", return_value=mock_client):
        result = await synthesize_base64("Test")

    assert isinstance(result, str)
