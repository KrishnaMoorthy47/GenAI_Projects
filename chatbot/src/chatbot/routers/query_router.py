from __future__ import annotations

import logging
import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from chatbot.adapters import cache_adapter, vector_adapter
from chatbot.config import get_settings
from chatbot.middleware.auth import verify_api_key
from chatbot.models.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from chatbot.services.chatbot_service import ContentSafetyError, process_query
from chatbot.services.ingestion_service import UPLOAD_SUPPORTED_EXTENSIONS, ingest_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chatbot"])


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    _: str = Depends(verify_api_key),
) -> QueryResponse:
    """
    Main chat endpoint. Accepts a question + session_id, returns an answer
    grounded in your indexed documents.
    """
    try:
        return process_query(request)
    except ContentSafetyError as exc:
        raise HTTPException(status_code=451, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form(default="default"),
    _: str = Depends(verify_api_key),
) -> UploadResponse:
    """
    Upload a document (PDF, DOCX, XLSX, or image) to be indexed into the vector DB.
    Returns the collection name, chunk count, and elapsed time.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in UPLOAD_SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Supported: {sorted(UPLOAD_SUPPORTED_EXTENSIONS)}"
            ),
        )

    # Write to a temp file so extractors can open it by path
    suffix = ext
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = ingest_file(file_path=tmp_path, collection=collection)
        # Restore original filename in result
        result.source = file.filename
        return UploadResponse(
            collection=result.collection,
            source=result.source,
            chunks_indexed=result.chunks_indexed,
            elapsed_seconds=result.elapsed_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Liveness check. Returns index and Redis connection status.
    No authentication required.
    """
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        index_loaded=vector_adapter.is_index_loaded(settings.default_collection),
        redis_connected=cache_adapter.is_redis_connected(),
    )
