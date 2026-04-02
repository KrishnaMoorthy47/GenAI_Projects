from __future__ import annotations

import asyncio
import logging
import os
import tempfile

from pdfextract.config import get_settings

logger = logging.getLogger(__name__)


def _whisper_sync(file_path: str) -> str:
    """Run LLM Whisperer synchronously (called in executor)."""
    from unstract.llmwhisperer.client import LLMWhispererClient

    settings = get_settings()
    client = LLMWhispererClient(
        base_url=settings.llmwhisperer_base_url,
        api_key=settings.llmwhisperer_api_key,
    )
    result = client.whisper(
        file_path=file_path,
        processing_mode="ocr",
        output_mode="line-printer",
        force_text_processing="false",
        line_splitter_tolerance="0.5",
    )
    return result.get("extracted_text", "")


async def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Write PDF bytes to a temp file, run LLM Whisperer, return raw text."""
    settings = get_settings()

    if not settings.llmwhisperer_api_key:
        raise ValueError("LLMWHISPERER_API_KEY is not configured")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        tmp_path = f.name

    try:
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, _whisper_sync, tmp_path)
        logger.info("LLM Whisperer extracted %d chars from PDF", len(text))
        return text
    finally:
        os.unlink(tmp_path)
