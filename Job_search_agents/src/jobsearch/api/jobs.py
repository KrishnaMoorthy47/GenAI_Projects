from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Security, UploadFile
from fastapi.security.api_key import APIKeyHeader

from jobsearch.config import get_settings
from jobsearch.models.response import JobSearchResultsResponse, JobSearchStartResponse, JobSearchStatusResponse
from jobsearch.services.browser_agent import (
    SUPPORTED_COMPANIES,
    create_search,
    get_search_store,
    run_job_search,
)
from jobsearch.services.cv_parser import extract_cv_text

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")

_api_key_header = APIKeyHeader(name="x-api-key", auto_error=True)


def verify_key(key: str = Security(_api_key_header)) -> str:
    if key != get_settings().api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key


@router.get("/companies", dependencies=[Depends(verify_key)], response_model=list[str])
async def list_companies() -> list[str]:
    """Return the list of supported company career sites."""
    return SUPPORTED_COMPANIES


@router.post("/search", dependencies=[Depends(verify_key)], response_model=JobSearchStartResponse)
async def start_job_search(
    cv: UploadFile = File(..., description="Your CV as a PDF file"),
    companies: str = Form(..., description="Comma-separated list of companies (e.g. Google,Amazon)"),
    role: str = Form(default="GenAI Engineer", description="Job role to search for"),
) -> JobSearchStartResponse:
    """
    Upload your CV and start a parallel job search across selected companies.
    Returns a search_id immediately. Poll /search/{search_id}/status for progress.
    """
    if not cv.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="CV must be a PDF file")

    pdf_bytes = await cv.read()
    cv_text = extract_cv_text(pdf_bytes)

    if not cv_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from CV PDF")

    company_list = [c.strip() for c in companies.split(",") if c.strip()]
    if not company_list:
        raise HTTPException(status_code=400, detail="At least one company must be specified")

    search_id = create_search(cv_text, company_list, role)
    asyncio.create_task(run_job_search(search_id, cv_text, company_list, role))

    return JobSearchStartResponse(
        search_id=search_id,
        status="pending",
        companies=company_list,
        role=role,
        cv_chars=len(cv_text),
    )


@router.get("/search/{search_id}/status", dependencies=[Depends(verify_key)], response_model=JobSearchStatusResponse)
async def get_search_status(search_id: str) -> JobSearchStatusResponse:
    """Poll the status of a running job search."""
    store = get_search_store()
    if search_id not in store:
        raise HTTPException(status_code=404, detail="Search not found")
    entry = store[search_id]
    return JobSearchStatusResponse(
        search_id=search_id,
        status=entry["status"],
        companies=entry["companies"],
        role=entry["role"],
        progress=entry["progress"],
        jobs_found=len(entry["jobs"]),
        error=entry.get("error"),
    )


@router.get("/search/{search_id}/results", dependencies=[Depends(verify_key)], response_model=JobSearchResultsResponse)
async def get_search_results(search_id: str) -> JobSearchResultsResponse:
    """Get the ranked job results for a completed search."""
    store = get_search_store()
    if search_id not in store:
        raise HTTPException(status_code=404, detail="Search not found")
    entry = store[search_id]
    if entry["status"] not in ("completed", "running"):
        raise HTTPException(status_code=409, detail=f"Search not ready — status: {entry['status']}")
    return JobSearchResultsResponse(
        search_id=search_id,
        status=entry["status"],
        role=entry["role"],
        jobs=entry["jobs"],
        total=len(entry["jobs"]),
    )
