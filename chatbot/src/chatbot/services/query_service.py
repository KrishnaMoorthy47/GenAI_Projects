from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import tiktoken

from chatbot.adapters import cache_adapter, llm_adapter, vector_adapter
from chatbot.adapters.supabase_adapter import RetrievedChunk as SupabaseChunk
from chatbot.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    answer: str
    sources: List[dict]  # [{"filename": str, "page_num": int}]
    has_context: bool


# ── Step 2: Query Sanitization ─────────────────────────────────────────────────

def sanitize_query(raw_question: str) -> str:
    """Strip, collapse spaces, remove control chars, enforce max length."""
    settings = get_settings()

    # Remove null bytes and non-printable control chars (keep newline and tab)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw_question)
    # Collapse multiple spaces
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = cleaned.strip()

    if len(cleaned) > settings.max_query_length:
        raise ValueError(
            f"Query exceeds maximum length of {settings.max_query_length} characters."
        )
    if not cleaned:
        raise ValueError("Query cannot be empty.")
    return cleaned


# ── Step 6: Context Token Budget ──────────────────────────────────────────────

def build_context_block(
    chunks: List[vector_adapter.RetrievedChunk],
    max_tokens: int,
) -> Tuple[str, List[vector_adapter.RetrievedChunk]]:
    """
    Pack chunks into a context string within the token budget.
    Returns (context_string, accepted_chunks).
    Chunks are added in score order (best first). Chunks that don't fit are skipped.
    """
    try:
        encoder = tiktoken.encoding_for_model("gpt-4o")
    except Exception:
        encoder = tiktoken.get_encoding("cl100k_base")

    context_parts: List[str] = []
    accepted: List[vector_adapter.RetrievedChunk] = []
    token_count = 0

    for chunk in chunks:
        chunk_tokens = len(encoder.encode(chunk.text))
        if token_count + chunk_tokens <= max_tokens:
            context_parts.append(
                f"{chunk.text}\nSource: {chunk.source}, page {chunk.page_num + 1}"
            )
            accepted.append(chunk)
            token_count += chunk_tokens

    separator = "\n\n---\n\n"
    return separator.join(context_parts), accepted


# ── Step 8: Relevance Pre-Check (optional) ────────────────────────────────────

def relevance_precheck(question: str, context: str) -> bool:
    """
    Lightweight LLM call to verify context is actually relevant to the question.
    Returns True if relevant, False otherwise.
    """
    prompt = (
        "Given the following context and question, answer YES if the context contains "
        "information that could answer the question, or NO if it does not.\n\n"
        f"Context:\n{context[:2000]}\n\n"
        f"Question: {question}\n\n"
        "Answer (YES/NO only):"
    )
    try:
        response = llm_adapter.chat_completion([
            {"role": "user", "content": prompt}
        ])
        return response.strip().upper().startswith("YES")
    except Exception as exc:
        logger.warning("Relevance pre-check failed: %s — proceeding with main LLM call", exc)
        return True  # Fail open


# ── Step 9: Build Prompt and Call LLM ─────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful Q&A assistant that answers questions based solely on the provided documentation. "
    "Only use information from the context provided — do not rely on external knowledge or make up information. "
    "If the context does not contain enough information to answer confidently, say so explicitly rather than guessing. "
    "Be professional, concise, and direct. Use plain text unless the question calls for a list or table. "
    "Respond in the same language as the user's question."
)


def build_messages(question: str, context: str, history: List[dict]) -> List[dict]:
    """Assemble the OpenAI chat messages list: system + history + user-with-context."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    user_content = (
        "The following information is retrieved from the documentation:\n\n"
        "---\n"
        f"{context}\n"
        "---\n\n"
        f"{question}"
    )
    messages.append({"role": "user", "content": user_content})
    return messages


# ── Main RAG Pipeline (Steps 3–11) ────────────────────────────────────────────

def run_rag_pipeline(
    question: str,
    session_id: str,
    collection_name: str = "default",
) -> RAGResult:
    """
    Execute the full 11-step RAG pipeline.
    Assumes Step 1 (auth) and Step 2 (sanitization) are handled by the caller.
    """
    settings = get_settings()

    # Step 3 & 4: Embed query + similarity search (Supabase or FAISS)
    query_vector = llm_adapter.embed_query(question)

    if settings.vector_backend == "supabase":
        from chatbot.adapters import supabase_adapter
        try:
            raw_chunks = supabase_adapter.search(
                collection=collection_name,
                query_vec=query_vector,
                k=settings.vector_search_k,
            )
        except Exception as exc:
            logger.error("Supabase search failed: %s", exc)
            return RAGResult(
                answer="Vector search is temporarily unavailable. Please try again later.",
                sources=[],
                has_context=False,
            )
    else:
        # FAISS fallback
        if not vector_adapter.is_index_loaded(collection_name):
            loaded = vector_adapter.load_index(settings.data_indexed_dir, collection_name)
            if not loaded:
                return RAGResult(
                    answer=(
                        "The document index has not been built yet. "
                        "Please run the ingestion script first: python scripts/ingest.py"
                    ),
                    sources=[],
                    has_context=False,
                )
        raw_chunks = vector_adapter.search(
            query_vector,
            k=settings.vector_search_k,
            collection=collection_name,
        )

    if not raw_chunks:
        return RAGResult(
            answer="I don't have any relevant information about that in my documents.",
            sources=[],
            has_context=False,
        )

    # Step 5: Dynamic threshold filtering
    max_score = max(c.score for c in raw_chunks)
    threshold = settings.threshold_ratio * max_score
    filtered_chunks = [c for c in raw_chunks if c.score >= threshold]

    if not filtered_chunks:
        logger.info("All chunks fell below threshold (max=%.4f, threshold=%.4f)", max_score, threshold)
        return RAGResult(
            answer="I don't have relevant information about that in my documents.",
            sources=[],
            has_context=False,
        )

    # Step 6: Context token budget
    context_str, accepted_chunks = build_context_block(
        filtered_chunks, max_tokens=settings.max_context_tokens
    )

    if not context_str.strip():
        return RAGResult(
            answer="I couldn't retrieve enough context to answer that question.",
            sources=[],
            has_context=False,
        )

    # Step 7: Load chat history
    history = cache_adapter.get_history(session_id)

    # Step 8: Optional relevance pre-check
    if settings.enable_relevance_precheck:
        if not relevance_precheck(question, context_str):
            logger.info("Relevance pre-check returned false — skipping LLM call")
            return RAGResult(
                answer="I don't have information about that in my documents.",
                sources=[],
                has_context=False,
            )

    # Step 9: Main LLM call
    messages = build_messages(question, context_str, history)
    answer = llm_adapter.chat_completion(messages)

    # Step 10: Save updated history
    cache_adapter.save_history(session_id, question, answer)

    # Step 11: Build structured response with deduplicated sources
    seen: set[str] = set()
    sources: List[dict] = []
    for chunk in accepted_chunks:
        key = f"{chunk.source}:{chunk.page_num}"
        if key not in seen:
            seen.add(key)
            sources.append({"filename": chunk.source, "page_num": chunk.page_num + 1})

    return RAGResult(answer=answer, sources=sources, has_context=True)
