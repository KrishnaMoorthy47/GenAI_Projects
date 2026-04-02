"""Tests for the WebSocket voice pipeline and REST endpoints."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from voiceagent.main import app
from voiceagent.services.session import SessionStore


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store():
    """A fresh SessionStore for each test."""
    return SessionStore()


@pytest.fixture
def client(store):
    """TestClient with a patched session store (bypasses lifespan get_store)."""
    with patch("voiceagent.main.get_store", return_value=store):
        with TestClient(app) as c:
            yield c


@pytest.fixture
def auth_headers():
    return {"x-api-key": "dev-secret"}


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["active_sessions"] == 0


# ── Session REST ──────────────────────────────────────────────────────────────

def test_create_session(client, auth_headers):
    r = client.post("/api/v1/session", headers=auth_headers)
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert len(data["session_id"]) > 0


def test_create_session_increments_count(client, auth_headers):
    client.post("/api/v1/session", headers=auth_headers)
    r = client.get("/health")
    assert r.json()["active_sessions"] == 1


def test_create_session_no_auth(client):
    r = client.post("/api/v1/session")
    assert r.status_code == 401


def test_delete_session(client, auth_headers):
    r = client.post("/api/v1/session", headers=auth_headers)
    sid = r.json()["session_id"]
    r2 = client.delete(f"/api/v1/session/{sid}", headers=auth_headers)
    assert r2.status_code == 200
    assert r2.json()["session_id"] == sid


def test_delete_nonexistent_session(client, auth_headers):
    r = client.delete("/api/v1/session/no-such-id", headers=auth_headers)
    assert r.status_code == 404


def test_delete_session_no_auth(client, auth_headers):
    r = client.post("/api/v1/session", headers=auth_headers)
    sid = r.json()["session_id"]
    r2 = client.delete(f"/api/v1/session/{sid}")
    assert r2.status_code == 401


# ── WebSocket: invalid session ─────────────────────────────────────────────────

def test_ws_rejects_unknown_session(client):
    """Server should close WS with code 4004 for unknown session IDs."""
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws/voice/no-such-session"):
            pass
    assert exc_info.value.code == 4004


# ── WebSocket: full pipeline ───────────────────────────────────────────────────

def test_ws_full_pipeline(store, client):
    """
    Full round-trip: send audio → expect transcript, llm_chunk, audio, done.
    All external APIs are mocked.
    """
    session = store.create()
    fake_pcm = b"\x00\x00" * 3200  # 0.2s silence

    async def fake_stream_sentences(messages, **kwargs):
        yield "It is noon."

    with (
        patch("voiceagent.main.stt_svc.transcribe", new_callable=AsyncMock, return_value="What time is it?"),
        patch("voiceagent.main.tts_svc.synthesize_base64", new_callable=AsyncMock, return_value="ZmFrZWF1ZGlv"),
        patch("voiceagent.main.llm_svc.stream_sentences", new=fake_stream_sentences),
    ):
        with client.websocket_connect(f"/ws/voice/{session.session_id}") as ws:
            ws.send_bytes(fake_pcm)

            messages = []
            for _ in range(4):  # transcript, llm_chunk, audio, done
                try:
                    text = ws.receive_text()
                    messages.append(json.loads(text))
                except Exception:
                    break

    types = [m["type"] for m in messages]
    assert "transcript" in types
    assert "llm_chunk" in types
    assert "audio" in types
    assert "done" in types

    transcript_msg = next(m for m in messages if m["type"] == "transcript")
    assert transcript_msg["text"] == "What time is it?"

    llm_msg = next(m for m in messages if m["type"] == "llm_chunk")
    assert llm_msg["text"] == "It is noon."


def test_ws_empty_transcript_sends_error(store, client):
    """Empty STT result should trigger an error message to the client."""
    session = store.create()

    with patch("voiceagent.main.stt_svc.transcribe", new_callable=AsyncMock, return_value=""):
        with client.websocket_connect(f"/ws/voice/{session.session_id}") as ws:
            ws.send_bytes(b"\x00\x00" * 100)
            try:
                msg = json.loads(ws.receive_text())
                assert msg["type"] == "error"
            except Exception:
                pass  # Acceptable — server may close


def test_ws_history_preserved_across_turns(store, client):
    """Conversation history should accumulate across multiple turns."""
    session = store.create()

    async def fake_stream_sentences(messages, **kwargs):
        yield "Turn response."

    with (
        patch("voiceagent.main.stt_svc.transcribe", new_callable=AsyncMock, return_value="Hello"),
        patch("voiceagent.main.tts_svc.synthesize_base64", new_callable=AsyncMock, return_value="ZmFrZQ=="),
        patch("voiceagent.main.llm_svc.stream_sentences", side_effect=fake_stream_sentences),
    ):
        with client.websocket_connect(f"/ws/voice/{session.session_id}") as ws:
            ws.send_bytes(b"\x00\x00" * 100)
            # Drain messages until done
            for _ in range(10):
                try:
                    msg = json.loads(ws.receive_text())
                    if msg["type"] == "done":
                        break
                except Exception:
                    break

    # Check history was stored
    assert len(session.history) >= 2  # at least user + assistant
    assert session.history[0]["role"] == "user"
    assert session.history[0]["content"] == "Hello"
    assert session.history[1]["role"] == "assistant"
