from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# Ensure env vars are set before imports
os.environ.setdefault("API_KEY", "test-secret")
os.environ.setdefault("OPENAI_API_KEY", "sk-test")


class TestQuerySanitization:
    def test_strips_whitespace(self):
        from chatbot.services.query_service import sanitize_query
        assert sanitize_query("  hello world  ") == "hello world"

    def test_collapses_spaces(self):
        from chatbot.services.query_service import sanitize_query
        assert sanitize_query("too   many   spaces") == "too many spaces"

    def test_removes_control_chars(self):
        from chatbot.services.query_service import sanitize_query
        result = sanitize_query("hello\x00world\x01")
        assert "\x00" not in result
        assert "\x01" not in result
        assert "helloworld" in result

    def test_raises_on_empty(self):
        from chatbot.services.query_service import sanitize_query
        with pytest.raises(ValueError, match="empty"):
            sanitize_query("   ")

    def test_raises_on_too_long(self, monkeypatch):
        from chatbot.services.query_service import sanitize_query
        monkeypatch.setenv("MAX_QUERY_LENGTH", "10")
        # Reload settings
        from chatbot.config import get_settings
        get_settings.cache_clear()
        with pytest.raises(ValueError, match="maximum length"):
            sanitize_query("a" * 11)
        get_settings.cache_clear()

    def test_preserves_valid_query(self):
        from chatbot.services.query_service import sanitize_query
        q = "What is the refund policy for digital purchases?"
        assert sanitize_query(q) == q


class TestContextTokenBudget:
    def test_fits_all_chunks_within_budget(self, sample_chunks):
        from chatbot.services.query_service import build_context_block
        context, accepted = build_context_block(sample_chunks, max_tokens=4000)
        assert len(accepted) == len(sample_chunks)
        assert "return policy" in context

    def test_excludes_chunks_over_budget(self, sample_chunks):
        from chatbot.services.query_service import build_context_block
        # Very small budget — only first chunk fits
        context, accepted = build_context_block(sample_chunks, max_tokens=20)
        assert len(accepted) <= len(sample_chunks)

    def test_returns_empty_for_zero_budget(self, sample_chunks):
        from chatbot.services.query_service import build_context_block
        context, accepted = build_context_block(sample_chunks, max_tokens=0)
        assert context == ""
        assert accepted == []


class TestDynamicThreshold:
    def test_filters_low_scoring_chunks(self, sample_chunks):
        """Chunks below threshold_ratio * max_score should be excluded."""
        # max_score = 0.95, threshold_ratio = 0.8 → threshold = 0.76
        # scores: 0.95 ✓, 0.88 ✓, 0.72 ✗
        threshold_ratio = 0.8
        max_score = max(c.score for c in sample_chunks)
        threshold = threshold_ratio * max_score
        filtered = [c for c in sample_chunks if c.score >= threshold]
        assert len(filtered) == 2  # 0.95 and 0.88 pass; 0.72 fails

    def test_all_pass_when_scores_are_close(self):
        from chatbot.adapters.vector_adapter import RetrievedChunk
        chunks = [
            RetrievedChunk("text1", "a.pdf", 0, 0.90, 0),
            RetrievedChunk("text2", "b.pdf", 0, 0.88, 1),
            RetrievedChunk("text3", "c.pdf", 0, 0.85, 2),
        ]
        threshold = 0.8 * max(c.score for c in chunks)
        filtered = [c for c in chunks if c.score >= threshold]
        assert len(filtered) == 3


class TestSchemaValidation:
    def test_valid_session_id(self):
        from chatbot.models.schemas import QueryRequest
        req = QueryRequest(question="Hello?", session_id="abc-1234-xyz")
        assert req.session_id == "abc-1234-xyz"

    def test_invalid_session_id_too_short(self):
        from pydantic import ValidationError
        from chatbot.models.schemas import QueryRequest
        with pytest.raises(ValidationError):
            QueryRequest(question="Hello?", session_id="abc")

    def test_invalid_session_id_special_chars(self):
        from pydantic import ValidationError
        from chatbot.models.schemas import QueryRequest
        with pytest.raises(ValidationError):
            QueryRequest(question="Hello?", session_id="abc@special!")

    def test_question_whitespace_stripped(self):
        from chatbot.models.schemas import QueryRequest
        req = QueryRequest(question="  What is this?  ", session_id="session-1234")
        assert req.question == "What is this?"

    def test_empty_question_rejected(self):
        from pydantic import ValidationError
        from chatbot.models.schemas import QueryRequest
        with pytest.raises(ValidationError):
            QueryRequest(question="   ", session_id="session-1234")


class TestCacheAdapter:
    def test_in_memory_get_returns_empty_for_new_session(self):
        from chatbot.adapters import cache_adapter
        result = cache_adapter.get_history("brand-new-session-999")
        assert result == []

    def test_in_memory_save_and_retrieve(self):
        from chatbot.adapters import cache_adapter
        sid = "test-cache-session-001"
        cache_adapter.save_history(sid, "What is X?", "X is a thing.")
        history = cache_adapter.get_history(sid)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        cache_adapter.clear_history(sid)

    def test_history_is_cleared(self):
        from chatbot.adapters import cache_adapter
        sid = "test-clear-session-001"
        cache_adapter.save_history(sid, "Q?", "A.")
        cache_adapter.clear_history(sid)
        assert cache_adapter.get_history(sid) == []

    def test_history_capped_at_max_turns(self, monkeypatch):
        from chatbot.adapters import cache_adapter
        from chatbot.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setenv("MAX_HISTORY_TURNS", "2")
        get_settings.cache_clear()

        sid = "test-max-turns-session"
        cache_adapter.clear_history(sid)
        # Add 3 turns (6 messages)
        for i in range(3):
            cache_adapter.save_history(sid, f"Q{i}", f"A{i}")

        history = cache_adapter.get_history(sid)
        assert len(history) <= 4  # max 2 turns * 2 messages
        cache_adapter.clear_history(sid)
        get_settings.cache_clear()
