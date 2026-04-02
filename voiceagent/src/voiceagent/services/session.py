# Copyright (c) 2024 ValGenesis Inc. All rights reserved.
"""In-memory session and conversation history store."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Session:
    session_id: str
    history: list[dict[str, str]] = field(default_factory=list)
    system_prompt: str = (
        "You are a helpful voice assistant. Keep responses concise and conversational "
        "— ideally 1-3 sentences — since they will be spoken aloud."
    )

    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})

    def messages_for_llm(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt}] + self.history


class SessionStore:
    """Thread-safe in-memory session registry."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        sid = str(uuid.uuid4())
        session = Session(session_id=sid)
        self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    @property
    def active_count(self) -> int:
        return len(self._sessions)


# Global singleton — initialized once at startup
_store: SessionStore | None = None


def get_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
