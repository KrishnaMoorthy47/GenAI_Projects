from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """Inject required env vars and clear lru_cache so every test gets clean Settings."""
    monkeypatch.setenv("API_KEY", "dev-secret")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setenv("LLMWHISPERER_API_KEY", "test-whisper")
    from pdfextract.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def app():
    from pdfextract.main import app
    return app


@pytest.fixture
def headers():
    return {"x-api-key": "dev-secret"}
