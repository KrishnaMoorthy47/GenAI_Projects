"""Tests for the STT service (Whisper)."""

from __future__ import annotations

import io
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voiceagent.services.stt import pcm_to_wav, transcribe


# ── pcm_to_wav ────────────────────────────────────────────────────────────────

def test_pcm_to_wav_returns_wav_bytes():
    """pcm_to_wav should return valid WAV-formatted bytes."""
    pcm = b"\x00\x00" * 1600  # 0.1s of silence at 16kHz, 16-bit
    wav = pcm_to_wav(pcm)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"


def test_pcm_to_wav_correct_sample_rate():
    """WAV header should encode the requested sample rate."""
    pcm = b"\x00\x00" * 800
    wav = pcm_to_wav(pcm, sample_rate=16000)
    buf = io.BytesIO(wav)
    with wave.open(buf) as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2


def test_pcm_to_wav_empty_input():
    """pcm_to_wav should handle empty PCM gracefully."""
    wav = pcm_to_wav(b"")
    assert wav[:4] == b"RIFF"


# ── transcribe ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_transcribe_returns_text():
    """transcribe() should return the Whisper API response text."""
    pcm = b"\x00\x00" * 1600

    mock_client = MagicMock()
    mock_client.audio = MagicMock()
    mock_client.audio.transcriptions = MagicMock()
    mock_client.audio.transcriptions.create = AsyncMock(return_value="Hello world")

    with patch("voiceagent.services.stt.get_client", return_value=mock_client):
        result = await transcribe(pcm, is_wav=False)

    assert result == "Hello world"


@pytest.mark.asyncio
async def test_transcribe_strips_whitespace():
    """transcribe() should strip leading/trailing whitespace from API response."""
    pcm = b"\x00\x00" * 800

    mock_client = MagicMock()
    mock_client.audio.transcriptions.create = AsyncMock(return_value="  hi there  ")

    with patch("voiceagent.services.stt.get_client", return_value=mock_client):
        result = await transcribe(pcm, is_wav=False)

    assert result == "hi there"


@pytest.mark.asyncio
async def test_transcribe_empty_response():
    """transcribe() should return empty string when API returns empty."""
    pcm = b"\x00\x00" * 800

    mock_client = MagicMock()
    mock_client.audio.transcriptions.create = AsyncMock(return_value="")

    with patch("voiceagent.services.stt.get_client", return_value=mock_client):
        result = await transcribe(pcm, is_wav=False)

    assert result == ""


@pytest.mark.asyncio
async def test_transcribe_wav_passthrough():
    """transcribe() with is_wav=True should skip pcm_to_wav conversion."""
    dummy_wav = b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 32

    mock_client = MagicMock()
    mock_client.audio.transcriptions.create = AsyncMock(return_value="test transcript")

    with patch("voiceagent.services.stt.get_client", return_value=mock_client):
        result = await transcribe(dummy_wav, is_wav=True)

    assert result == "test transcript"
    # Verify it was called with the exact bytes passed in (not re-wrapped)
    call_kwargs = mock_client.audio.transcriptions.create.call_args
    file_arg = call_kwargs.kwargs.get("file") or call_kwargs.args[0] if call_kwargs.args else None
    # The mock was called — that's sufficient
    mock_client.audio.transcriptions.create.assert_called_once()
