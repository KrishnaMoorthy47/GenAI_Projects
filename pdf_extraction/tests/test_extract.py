from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from pdfextract.models.document import ExtractionResult, Header, ProductLineItem

MOCK_RESULT = ExtractionResult(
    header=Header(
        purchase_order="PO-12345",
        invoice_order="INV-9999",
        vendor="ACME Corp",
        ship_to="123 Main St",
        bill_to="456 Oak Ave",
    ),
    line_items=[
        ProductLineItem(
            quantity=10,
            unit_price=5.99,
            currency="USD",
            uom="each",
            part_number="PART-001",
            product_description="Stainless steel bolt, M6x20mm",
            spell_corrected_product_description="Stainless steel bolt, M6 x 20mm",
            query_wout_uom="stainless steel bolt M6 20mm",
            product_metadata="productname:bolt|material:stainless steel|size:M6x20mm",
        )
    ],
)

MINIMAL_PDF = b"%PDF-1.4 minimal"


@pytest.mark.asyncio
async def test_health(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_extract_pdf(app, headers):
    with (
        patch("pdfextract.api.extract.extract_text_from_pdf", new_callable=AsyncMock, return_value="raw text"),
        patch("pdfextract.api.extract.extract_structured", new_callable=AsyncMock, return_value=MOCK_RESULT),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/v1/extract",
                files={"file": ("invoice.pdf", MINIMAL_PDF, "application/pdf")},
                headers=headers,
            )
    assert r.status_code == 200
    data = r.json()
    assert "job_id" in data
    assert data["status"] in ("pending", "completed")


@pytest.mark.asyncio
async def test_extract_non_pdf_rejected(app, headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/api/v1/extract",
            files={"file": ("doc.txt", b"hello", "text/plain")},
            headers=headers,
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_get_extraction_not_found(app, headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/v1/extract/nonexistent-id", headers=headers)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_unauthorized(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/api/v1/extract",
            files={"file": ("invoice.pdf", MINIMAL_PDF, "application/pdf")},
            headers={"x-api-key": "wrong"},
        )
    assert r.status_code == 401
