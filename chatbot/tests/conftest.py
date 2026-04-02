from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_client():
    """FastAPI test client with API key header."""
    os.environ.setdefault("API_KEY", "test-secret")
    os.environ.setdefault("OPENAI_API_KEY", "sk-test")
    os.environ.setdefault("LLM_PROVIDER", "openai")

    # Import here so env vars are set first
    from chatbot.main import create_app
    app = create_app()
    return TestClient(app)


@pytest.fixture
def auth_headers():
    return {"x-api-key": "test-secret"}


@pytest.fixture
def sample_chunks():
    """Sample FAISS search results for testing."""
    from chatbot.adapters.vector_adapter import RetrievedChunk
    return [
        RetrievedChunk(text="The return policy allows 30-day returns.", source="policy.pdf", page_num=0, score=0.95, vector_id=0),
        RetrievedChunk(text="Digital products are non-refundable.", source="policy.pdf", page_num=1, score=0.88, vector_id=1),
        RetrievedChunk(text="Contact support at support@example.com.", source="faq.txt", page_num=0, score=0.72, vector_id=2),
    ]
