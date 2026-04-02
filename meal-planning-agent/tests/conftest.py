from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """Inject required env vars and clear lru_cache so every test gets clean Settings."""
    monkeypatch.setenv("API_KEY", "dev-secret")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")
    from mealplanner.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def app():
    with (
        patch("mealplanner.services.mongodb.init_mongodb", new_callable=AsyncMock),
        patch("mealplanner.services.mongodb.close_mongodb", new_callable=AsyncMock),
    ):
        from mealplanner.main import app
        return app


@pytest.fixture
def headers():
    return {"x-api-key": "dev-secret"}
