from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Security, UploadFile, File
from fastapi.security.api_key import APIKeyHeader

from pdfextract.config import get_settings
from pdfextract.models.document import ExtractionResponse
from pdfextract.services.extractor import extract_structured
from pdfextract.services.whisperer import extract_text_from_pdf

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")

_api_key_header = APIKeyHeader(name="x-api-key", auto_error=True)

# In-memory job store  (same pattern as agenteval)
_job_store: dict[str, dict[str, Any]] = {}


def verify_key(key: str = Security(_api_key_header)) -> str:
    if key != get_settings().api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key


async def _run_extraction(job_id: str, pdf_bytes: bytes, filename: str) -> None:
    store = _job_store[job_id]
    try:
        store["status"] = "processing"
        logger.info("[%s] Extracting text via LLM Whisperer", job_id)
        raw_text = await extract_text_from_pdf(pdf_bytes)

        logger.info("[%s] Running structured extraction", job_id)
        result = await extract_structured(raw_text)

        store["status"] = "completed"
        store["result"] = result
        logger.info("[%s] Extraction complete — %d line items", job_id, len(result.line_items))
    except Exception as exc:
        logger.exception("[%s] Extraction failed: %s", job_id, exc)
        store["status"] = "failed"
        store["error"] = str(exc)


@router.post("/extract", dependencies=[Depends(verify_key)], response_model=ExtractionResponse)
async def extract_pdf(file: UploadFile = File(...)) -> ExtractionResponse:
    """
    Upload a PDF (invoice, PO, etc.) for extraction.
    Returns immediately with a job_id. Poll /extract/{job_id} for results.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()
    job_id = str(uuid.uuid4())
    _job_store[job_id] = {
        "status": "pending",
        "filename": file.filename,
        "error": None,
        "result": None,
    }
    asyncio.create_task(_run_extraction(job_id, pdf_bytes, file.filename))

    return ExtractionResponse(
        job_id=job_id,
        status="pending",
        filename=file.filename,
    )


@router.get("/extract/{job_id}", dependencies=[Depends(verify_key)], response_model=ExtractionResponse)
async def get_extraction(job_id: str) -> ExtractionResponse:
    """Poll the status/result of an extraction job."""
    if job_id not in _job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    entry = _job_store[job_id]
    return ExtractionResponse(
        job_id=job_id,
        status=entry["status"],
        filename=entry.get("filename", ""),
        error=entry.get("error"),
        result=entry.get("result"),
    )
