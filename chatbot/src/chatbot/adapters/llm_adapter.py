from __future__ import annotations

import base64
import logging
from typing import List

from openai import AzureOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from chatbot.config import get_settings

logger = logging.getLogger(__name__)

# Separate clients: chat can be Groq/Ollama/OpenAI; embed is always OpenAI
_chat_client: OpenAI | None = None
_embed_client: OpenAI | None = None


def _get_chat_client() -> OpenAI:
    global _chat_client
    if _chat_client is None:
        settings = get_settings()
        if settings.llm_provider == "groq":
            _chat_client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=settings.groq_api_key,
            )
        elif settings.llm_provider == "ollama":
            _chat_client = OpenAI(
                base_url=f"{settings.ollama_base_url}/v1",
                api_key="ollama",
            )
        # elif settings.llm_provider == "azure_openai":
        #     _chat_client = AzureOpenAI(
        #         azure_endpoint=settings.azure_openai_endpoint,
        #         api_key=settings.azure_openai_api_key,
        #         api_version=settings.azure_openai_api_version,
        #     )
        else:
            _chat_client = OpenAI(api_key=settings.openai_api_key)
    return _chat_client


def _get_embed_client() -> OpenAI:
    """Always OpenAI — Groq does not provide an embeddings API."""
    global _embed_client
    if _embed_client is None:
        settings = get_settings()
        _embed_client = OpenAI(api_key=settings.openai_api_key or None)
    return _embed_client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def embed_query(text: str) -> List[float]:
    """Embed a single text string into a float vector."""
    settings = get_settings()
    client = _get_embed_client()
    model = settings.embedding_model  # always text-embedding-3-small via OpenAI
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


def embed_texts(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """Embed a list of texts in batches of batch_size."""
    settings = get_settings()
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.info("Embedding batch %d–%d of %d...", i + 1, i + len(batch), len(texts))
        all_embeddings.extend(_embed_batch(batch, settings.embedding_model))
    return all_embeddings


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _embed_batch(batch: List[str], model: str) -> List[List[float]]:
    client = _get_embed_client()
    response = client.embeddings.create(model=model, input=batch)
    return [item.embedding for item in response.data]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def chat_completion(messages: List[dict]) -> str:
    """Call the LLM with a list of messages and return the response text."""
    settings = get_settings()
    client = _get_chat_client()
    model = settings.groq_model if settings.llm_provider == "groq" else settings.llm_model
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    return response.choices[0].message.content or ""


def vision_extract(image_bytes: bytes) -> str:
    """
    Send an image to Groq vision and return the extracted text / description.
    Uses llama-3.2-11b-vision-preview via Groq's OpenAI-compatible API.
    Falls back gracefully and returns an empty string on error.
    """
    settings = get_settings()
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=settings.groq_api_key,
    )
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    try:
        response = client.chat.completions.create(
            model=settings.groq_vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Extract all text, numbers, and table content from this image. "
                                "Preserve table structure using pipe-separated columns. "
                                "Return only the extracted content, no commentary."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=2048,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("Groq vision extraction failed: %s", exc)
        return ""


def moderate(text: str) -> bool:
    """
    Check text against OpenAI's moderation API.
    Returns True if content is SAFE, False if flagged.
    Fails open on API error (logs warning, returns True).
    """
    settings = get_settings()
    if settings.llm_provider in ("ollama", "groq", "azure_openai"):
        # These providers don't have a moderation endpoint — skip, use OpenAI embed client
        return True

    try:
        client = _get_embed_client()
        response = client.moderations.create(input=text)
        flagged = response.results[0].flagged
        if flagged:
            categories = response.results[0].categories
            logger.warning("Content moderation flagged input. Categories: %s", categories)
        return not flagged
    except Exception as exc:
        logger.warning("Moderation API call failed (failing open): %s", exc)
        return True  # Fail open — don't block users when moderation is unavailable
