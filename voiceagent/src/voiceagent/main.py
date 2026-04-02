# Copyright (c) 2024 ValGenesis Inc. All rights reserved.
"""VoiceAgent — Real-time voice AI agent.

WebSocket flow:
  1. Client sends raw PCM audio bytes (16kHz, 16-bit, mono)
  2. Whisper STT → transcript sent back
  3. Groq LLM streams sentence by sentence
  4. Each sentence → OpenAI TTS → base64 audio chunk sent back
  5. Done message signals end of turn
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))  # F:\Sample root

from fastapi import Depends, FastAPI, HTTPException, Security, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from rate_limit import RateLimitMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles

from voiceagent.config import get_settings
from voiceagent.models import (
    AudioMessage,
    DoneMessage,
    ErrorMessage,
    HealthResponse,
    LLMChunkMessage,
    SessionCreateResponse,
    SessionDeleteResponse,
    TranscriptMessage,
)
from voiceagent.services import llm as llm_svc
from voiceagent.services import stt as stt_svc
from voiceagent.services import tts as tts_svc
from voiceagent.services.session import SessionStore, get_store

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

# Inactivity timeout for WebSocket connections (seconds)
_WS_RECEIVE_TIMEOUT = 120.0


def _require_api_key(key: str | None = Security(_api_key_header)) -> str:
    if key != get_settings().api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


# ── App Lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.sessions = get_store()
    logger.info("VoiceAgent started — session store ready")
    yield
    logger.info("VoiceAgent shutting down")


app = FastAPI(
    title="VoiceAgent",
    description="Real-time voice AI agent: STT → LLM → TTS over WebSocket",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(RateLimitMiddleware, max_requests=10, window=60)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the browser frontend
_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    store: SessionStore = app.state.sessions
    return HealthResponse(active_sessions=store.active_count)


@app.post("/api/v1/session", response_model=SessionCreateResponse, dependencies=[Depends(_require_api_key)])
async def create_session() -> SessionCreateResponse:
    store: SessionStore = app.state.sessions
    session = store.create()
    return SessionCreateResponse(session_id=session.session_id)


@app.delete(
    "/api/v1/session/{session_id}",
    response_model=SessionDeleteResponse,
    dependencies=[Depends(_require_api_key)],
)
async def delete_session(session_id: str) -> SessionDeleteResponse:
    store: SessionStore = app.state.sessions
    deleted = store.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionDeleteResponse(session_id=session_id)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    static_index = os.path.join(_static_dir, "index.html")
    with open(static_index) as f:
        return HTMLResponse(f.read())


# ── WebSocket Endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/voice/{session_id}")
async def voice_websocket(websocket: WebSocket, session_id: str) -> None:
    """
    WebSocket voice pipeline.

    Client sends: raw PCM bytes (16kHz, 16-bit, mono) or a WAV file.
    Server sends JSON messages:
      { "type": "transcript", "text": "..." }
      { "type": "llm_chunk",  "text": "..." }
      { "type": "audio",      "data": "<base64-mp3>" }
      { "type": "done" }
      { "type": "error",      "message": "..." }
    """
    store: SessionStore = app.state.sessions
    session = store.get(session_id)
    if session is None:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()
    logger.info("WebSocket connected: session=%s", session_id)

    try:
        while True:
            # Receive audio bytes from client with inactivity timeout
            try:
                audio_bytes = await asyncio.wait_for(
                    websocket.receive_bytes(), timeout=_WS_RECEIVE_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.info("WebSocket idle timeout: session=%s", session_id)
                await websocket.close(code=1001, reason="Inactivity timeout")
                return

            if not audio_bytes:
                continue

            # 1. Speech-to-Text
            try:
                # Detect WAV by magic bytes
                is_wav = audio_bytes[:4] == b"RIFF"
                transcript = await stt_svc.transcribe(audio_bytes, is_wav=is_wav)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("STT error")
                await _send(websocket, ErrorMessage(message=f"STT error: {e}"))
                continue

            if not transcript:
                await _send(websocket, ErrorMessage(message="No speech detected"))
                continue

            # 2. Send transcript back to client
            await _send(websocket, TranscriptMessage(text=transcript))

            # 3. Add to conversation history
            session.add_user(transcript)

            # 4. Stream LLM response sentence by sentence → TTS each sentence
            full_response = []
            try:
                async for sentence in llm_svc.stream_sentences(session.messages_for_llm()):
                    full_response.append(sentence)

                    # Send LLM text chunk
                    await _send(websocket, LLMChunkMessage(text=sentence))

                    # Synthesize and send audio
                    try:
                        audio_b64 = await tts_svc.synthesize_base64(sentence)
                        await _send(websocket, AudioMessage(data=audio_b64))
                    except asyncio.CancelledError:
                        raise
                    except Exception as tts_err:
                        logger.exception("TTS error for sentence: %s", sentence)
                        await _send(websocket, ErrorMessage(message=f"TTS error: {tts_err}"))

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("LLM streaming error")
                await _send(websocket, ErrorMessage(message=f"LLM error: {e}"))
                continue

            # 5. Save assistant response to history
            if full_response:
                session.add_assistant(" ".join(full_response))

            # 6. Signal end of turn
            await _send(websocket, DoneMessage())

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session=%s", session_id)
    except asyncio.CancelledError:
        logger.info("WebSocket task cancelled: session=%s", session_id)
        raise
    except Exception:
        logger.exception("Unexpected WebSocket error: session=%s", session_id)


async def _send(ws: WebSocket, msg: object) -> None:
    """Serialize a Pydantic model and send as JSON text."""
    if hasattr(msg, "model_dump"):
        await ws.send_text(json.dumps(msg.model_dump()))
    else:
        await ws.send_text(json.dumps(msg))
