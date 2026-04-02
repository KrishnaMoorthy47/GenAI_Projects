from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential

from pdfextract.config import get_llm
from pdfextract.models.document import ExtractionResult, Header, ProductLineItem

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _invoke_chain(chain, inputs: dict):
    return chain.invoke(inputs)


_SYSTEM = """You are an expert document extraction algorithm specialised in invoices and purchase orders.
Extract ALL product line items and the document header from the provided text.
Be precise — extract every field described in the schema.
If a field is not present in the document, return null for that field."""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "Extract all data from this document:\n\n{text}"),
])


def _extract_sync(raw_text: str) -> ExtractionResult:
    """Run structured LLM extraction synchronously (called in executor)."""
    llm = get_llm(temperature=0.0)

    # Extract header
    header_llm = llm.with_structured_output(Header)
    header_chain = _PROMPT | header_llm
    header = _invoke_chain(header_chain, {"text": raw_text})

    # Extract line items — ask LLM to return a list
    from pydantic import BaseModel
    from typing import List

    class LineItemList(BaseModel):
        items: List[ProductLineItem]

    items_llm = llm.with_structured_output(LineItemList)
    items_chain = _PROMPT | items_llm
    items_result = _invoke_chain(items_chain, {
        "text": (
            f"{raw_text}\n\n"
            "Return ALL product line items found in this document as a list."
        )
    })

    return ExtractionResult(
        header=header,
        line_items=items_result.items if items_result else [],
    )


async def extract_structured(raw_text: str) -> ExtractionResult:
    """Async wrapper — runs extraction in thread pool to avoid blocking event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _extract_sync, raw_text)
