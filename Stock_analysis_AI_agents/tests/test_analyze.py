from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

MOCK_QUOTE = {
    "symbol": "AAPL",
    "name": "Apple Inc.",
    "price": 195.5,
    "change_pct": 0.015,
    "market_cap": 3000000000000,
    "pe_ratio": 28.5,
    "52w_high": 220.0,
    "52w_low": 160.0,
    "sector": "Technology",
    "currency": "USD",
}

MOCK_REPORT = "## AAPL Analysis\n\n**Recommendation: BUY**\n\nStrong fundamentals..."


@pytest.mark.asyncio
async def test_health(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_analyze_returns_job_id(app, headers):
    with patch("stockagent.api.analyze.run_analysis", new_callable=AsyncMock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/api/v1/analyze", json={"query": "Is AAPL a good buy?"}, headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert "job_id" in data
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_analyze_empty_query_rejected(app, headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/analyze", json={"query": "  "}, headers=headers)
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_get_analysis_not_found(app, headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/v1/analyze/nonexistent", headers=headers)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_stock_quote(app, headers):
    mock_info = {
        "longName": "Apple Inc.", "currentPrice": 195.5, "regularMarketChangePercent": 0.015,
        "marketCap": 3_000_000_000_000, "trailingPE": 28.5, "fiftyTwoWeekHigh": 220.0,
        "fiftyTwoWeekLow": 160.0, "sector": "Technology", "currency": "USD",
    }
    with patch("stockagent.api.analyze.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.info = mock_info
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/stock/AAPL/quote", headers=headers)
    assert r.status_code == 200
    assert r.json()["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_popular_stocks(app, headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/v1/stocks/popular", headers=headers)
    assert r.status_code == 200
    assert len(r.json()) > 0


@pytest.mark.asyncio
async def test_unauthorized(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/analyze", json={"query": "AAPL"}, headers={"x-api-key": "wrong"})
    assert r.status_code == 401
