# Copyright (c) 2024 ValGenesis Inc. All rights reserved.
"""Pydantic models for the VoiceAgent API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


# ── REST Models ──────────────────────────────────────────────────────────────

class SessionCreateResponse(BaseModel):
    session_id: str
    message: str = "Session created"


class SessionDeleteResponse(BaseModel):
    session_id: str
    message: str = "Session deleted"


class HealthResponse(BaseModel):
    status: str = "ok"
    active_sessions: int = 0


# ── WebSocket Message Models ──────────────────────────────────────────────────

class TranscriptMessage(BaseModel):
    type: Literal["transcript"] = "transcript"
    text: str


class LLMChunkMessage(BaseModel):
    type: Literal["llm_chunk"] = "llm_chunk"
    text: str


class AudioMessage(BaseModel):
    type: Literal["audio"] = "audio"
    data: str  # base64-encoded MP3


class DoneMessage(BaseModel):
    type: Literal["done"] = "done"


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    message: str
