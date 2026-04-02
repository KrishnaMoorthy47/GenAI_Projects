from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage


@pytest.fixture
def mock_llm():
    """
    Mock LLM that cycles through valid supervisor routing responses on successive calls.
    Uses side_effect so tests that exercise the LLM fallback path get realistic,
    varying responses rather than a single hardcoded value.
    """
    llm = MagicMock()

    _valid_responses = ["web_research", "financial_data", "sentiment", "report_writer"]
    _call_count = {"n": 0}

    def _invoke(messages, **kwargs):
        response = _valid_responses[_call_count["n"] % len(_valid_responses)]
        _call_count["n"] += 1
        return AIMessage(content=response, tool_calls=[])

    llm.invoke = MagicMock(side_effect=_invoke)
    llm.bind_tools = MagicMock(return_value=llm)
    llm.with_structured_output = MagicMock(return_value=llm)
    return llm


@pytest.fixture
def mock_settings(monkeypatch):
    """Fixture that injects safe test settings."""
    monkeypatch.setenv("API_KEY", "test-secret")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "finagent_test")
    monkeypatch.setenv("POSTGRES_USER", "finagent")
    monkeypatch.setenv("POSTGRES_PASSWORD", "finagent")
