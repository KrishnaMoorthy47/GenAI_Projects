import pytest


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """Inject required env vars and clear lru_cache so every test gets clean Settings."""
    monkeypatch.setenv("API_KEY", "dev-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

    from voiceagent.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
