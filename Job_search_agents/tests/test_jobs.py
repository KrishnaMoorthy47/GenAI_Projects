from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import io

import pytest
from httpx import ASGITransport, AsyncClient

MOCK_CV_TEXT = "Krishna Moorthy — AI Engineer with 3 years experience in LangChain, LangGraph, FastAPI..."
MINIMAL_PDF = b"%PDF-1.4 test"


@pytest.mark.asyncio
async def test_health(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_list_companies(app, headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/v1/companies", headers=headers)
    assert r.status_code == 200
    assert "Google" in r.json()


@pytest.mark.asyncio
async def test_start_job_search(app, headers):
    with (
        patch("jobsearch.api.jobs.extract_cv_text", return_value=MOCK_CV_TEXT),
        patch("jobsearch.api.jobs.run_job_search", new_callable=AsyncMock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/search",
                files={"cv": ("cv.pdf", MINIMAL_PDF, "application/pdf")},
                data={"companies": "Google,Amazon", "role": "GenAI Engineer"},
                headers=headers,
            )
    assert r.status_code == 200
    data = r.json()
    assert "search_id" in data
    assert data["status"] == "pending"
    assert data["companies"] == ["Google", "Amazon"]


@pytest.mark.asyncio
async def test_non_pdf_rejected(app, headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/api/v1/search",
            files={"cv": ("cv.txt", b"hello", "text/plain")},
            data={"companies": "Google", "role": "Engineer"},
            headers=headers,
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_search_status_not_found(app, headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/v1/search/nonexistent/status", headers=headers)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_unauthorized(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/v1/companies", headers={"x-api-key": "wrong"})
    assert r.status_code == 401
