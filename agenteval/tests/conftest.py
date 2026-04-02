"""Pytest fixtures for agenteval tests."""

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, ToolMessage

from agenteval.eval.scorers.judge_model import JudgeResult


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """Inject required env vars and clear the lru_cache so every test gets a clean Settings."""
    monkeypatch.setenv("API_KEY", "test-secret")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

    # Clear cached settings so the monkeypatched env is picked up
    from agenteval.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ──────────────────────────────────────────────
# In-memory SQLite fixture (Tamil Songs schema)
# ──────────────────────────────────────────────
TAMIL_DDL = """
CREATE TABLE Composer (ComposerId INTEGER PRIMARY KEY, Name TEXT NOT NULL);
CREATE TABLE Movie (MovieId INTEGER PRIMARY KEY, Title TEXT NOT NULL, Year INTEGER, Director TEXT);
CREATE TABLE Genre (GenreId INTEGER PRIMARY KEY, Name TEXT NOT NULL);
CREATE TABLE Singer (SingerId INTEGER PRIMARY KEY, Name TEXT NOT NULL, Gender TEXT NOT NULL);
CREATE TABLE Song (
    SongId INTEGER PRIMARY KEY,
    Title TEXT NOT NULL,
    MovieId INTEGER,
    GenreId INTEGER,
    ComposerId INTEGER,
    Lyricist TEXT,
    DurationSeconds INTEGER
);
CREATE TABLE SongSinger (SongId INTEGER, SingerId INTEGER, PRIMARY KEY(SongId, SingerId));
CREATE TABLE Playlist (PlaylistId INTEGER PRIMARY KEY, Name TEXT NOT NULL);
CREATE TABLE PlaylistSong (PlaylistId INTEGER, SongId INTEGER, PRIMARY KEY(PlaylistId, SongId));
"""

TAMIL_DATA = """
INSERT INTO Composer VALUES (1, 'AR Rahman'), (2, 'Ilaiyaraaja'), (3, 'Anirudh Ravichander');
INSERT INTO Movie VALUES (1, 'Roja', 1992, 'Mani Ratnam'), (2, 'Bombay', 1995, 'Mani Ratnam'), (3, '3', 2012, 'Aishwarya R. Dhanush');
INSERT INTO Genre VALUES (1, 'Melody'), (2, 'Sad'), (3, 'Folk'), (4, 'Dance'), (5, 'Romantic');
INSERT INTO Singer VALUES (1, 'SP Balasubrahmanyam', 'Male'), (2, 'KS Chithra', 'Female'), (3, 'Hariharan', 'Male'), (4, 'Benny Dayal', 'Male');
INSERT INTO Song VALUES
  (1, 'Roja Janeman', 1, 1, 1, 'Vairamuthu', 248),
  (2, 'Chinna Chinna Aasai', 1, 1, 1, 'Vairamuthu', 312),
  (3, 'Uyire', 2, 5, 1, 'Vairamuthu', 339),
  (4, 'Humma Humma', 2, 4, 1, 'Vairamuthu', 277),
  (5, 'Why This Kolaveri Di', 3, 2, 3, 'Dhanush', 236),
  (6, 'Kannazhaga', 3, 1, 3, 'Dhanush', 342);
INSERT INTO SongSinger VALUES (1, 1), (1, 2), (2, 2), (3, 1), (4, 1), (5, 4), (6, 4);
INSERT INTO Playlist VALUES (1, 'AR Rahman Hits'), (2, 'All Time Classics');
INSERT INTO PlaylistSong VALUES (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 6);
"""


@pytest.fixture
def mock_db(tmp_path):
    """Create a SQLite DB with Tamil Songs schema and return its path."""
    db_path = tmp_path / "test_tamil_songs.db"
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(TAMIL_DDL)
        conn.executescript(TAMIL_DATA)
    return str(db_path)


# ──────────────────────────────────────────────
# Mock LLM fixture
# ──────────────────────────────────────────────

def make_mock_llm(final_answer: str = "There are 6 songs.", tool_calls_sequence=None):
    """
    Create a mock LLM that:
    1. First returns an AIMessage with a list_tables tool call
    2. Then returns an AIMessage with run_query tool call
    3. Finally returns an AIMessage with no tool calls (final answer)
    """
    # Default sequence: list_tables → run_query → final answer
    if tool_calls_sequence is None:
        tool_calls_sequence = [
            AIMessage(
                content="",
                tool_calls=[{"name": "list_tables", "args": {}, "id": "tc1", "type": "tool_call"}],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_query",
                        "args": {"query": "SELECT COUNT(*) FROM Song"},
                        "id": "tc2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content=final_answer),
        ]

    call_count = {"n": 0}

    mock_llm = MagicMock()

    def invoke_side_effect(messages):
        idx = min(call_count["n"], len(tool_calls_sequence) - 1)
        call_count["n"] += 1
        return tool_calls_sequence[idx]

    async def ainvoke_side_effect(messages):
        idx = min(call_count["n"], len(tool_calls_sequence) - 1)
        call_count["n"] += 1
        return tool_calls_sequence[idx]

    mock_llm.invoke = invoke_side_effect
    mock_llm.ainvoke = AsyncMock(side_effect=ainvoke_side_effect)
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)

    # Support with_structured_output used by LLM-as-judge scorers
    judge_mock = MagicMock()
    judge_mock.ainvoke = AsyncMock(return_value=JudgeResult(score=0.9, reason="mock judge"))
    mock_llm.with_structured_output = MagicMock(return_value=judge_mock)

    return mock_llm


@pytest.fixture
def mock_llm():
    return make_mock_llm()


# ──────────────────────────────────────────────
# pytest-asyncio mode
# ──────────────────────────────────────────────
pytest_plugins = ["pytest_asyncio"]
