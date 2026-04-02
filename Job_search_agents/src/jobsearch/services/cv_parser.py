from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)


def extract_cv_text(pdf_bytes: bytes) -> str:
    """Extract text from a PDF CV using PyPDF2."""
    from PyPDF2 import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    logger.info("Extracted %d characters from CV PDF", len(text))
    return text
