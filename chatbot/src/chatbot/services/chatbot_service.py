from __future__ import annotations

import logging

from chatbot.adapters import llm_adapter
from chatbot.config import get_settings
from chatbot.models.schemas import QueryRequest, QueryResponse, SourceDocument
from chatbot.services.query_service import RAGResult, sanitize_query, run_rag_pipeline

logger = logging.getLogger(__name__)


def process_query(request: QueryRequest) -> QueryResponse:
    """
    Orchestrator: validate → sanitize → content safety → RAG pipeline → respond.
    """
    settings = get_settings()

    # Step 2: Sanitize query
    try:
        clean_question = sanitize_query(request.question)
    except ValueError as exc:
        return QueryResponse(
            answer=str(exc),
            session_id=request.session_id,
            sources=[],
            has_context=False,
        )

    # Step 11 (content safety): OpenAI Moderation API check
    if settings.enable_content_safety:
        is_safe = llm_adapter.moderate(clean_question)
        if not is_safe:
            # Return 451 — caller (router) must raise HTTPException
            raise ContentSafetyError("This query cannot be processed due to content policy.")

    # Steps 3–11: Full RAG pipeline
    try:
        result: RAGResult = run_rag_pipeline(
            question=clean_question,
            session_id=request.session_id,
            collection_name=request.collection_name,
        )
    except Exception as exc:
        logger.exception("RAG pipeline error for session %s: %s", request.session_id, exc)
        return QueryResponse(
            answer="I encountered an error while processing your question. Please try again.",
            session_id=request.session_id,
            sources=[],
            has_context=False,
        )

    sources = [
        SourceDocument(filename=s["filename"], page_num=s["page_num"])
        for s in result.sources
    ]

    return QueryResponse(
        answer=result.answer,
        session_id=request.session_id,
        sources=sources,
        has_context=result.has_context,
    )


class ContentSafetyError(Exception):
    """Raised when content moderation flags the user's query."""
    pass
