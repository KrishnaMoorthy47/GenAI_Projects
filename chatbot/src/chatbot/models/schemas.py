from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(
        description="The user's question.",
        min_length=1,
        max_length=1000,
    )
    session_id: str = Field(
        description="Unique conversation session identifier (UUID v4 recommended).",
    )
    collection_name: str = Field(
        default="default",
        description="Document collection to search. Default: 'default'.",
    )
    language: Optional[str] = Field(
        default=None,
        description="ISO 639-1 language code for response (e.g. 'en', 'fr'). Auto-detected if omitted.",
    )

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Question cannot be empty or whitespace.")
        return stripped

    @field_validator("session_id")
    @classmethod
    def session_id_format(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9\-_]{8,64}$", v):
            raise ValueError(
                "session_id must be 8–64 characters, alphanumeric with hyphens and underscores only."
            )
        return v


class SourceDocument(BaseModel):
    filename: str = Field(description="Source filename (e.g. 'user_manual.pdf')")
    page_num: int = Field(description="1-indexed page number within the document")


class QueryResponse(BaseModel):
    answer: str = Field(description="LLM-generated answer grounded in retrieved documents")
    session_id: str = Field(description="Echo of the session_id from the request")
    sources: List[SourceDocument] = Field(
        default_factory=list,
        description="Source documents used. Empty if no relevant context found.",
    )
    has_context: bool = Field(
        description="True if documents were found and used; False if answering without document context."
    )


class UploadResponse(BaseModel):
    collection: str = Field(description="Collection the document was indexed into")
    source: str = Field(description="Original filename")
    chunks_indexed: int = Field(description="Number of text chunks stored in the vector DB")
    elapsed_seconds: float = Field(description="Total processing time in seconds")


class HealthResponse(BaseModel):
    status: str = Field(default="healthy")
    version: str
    index_loaded: bool = Field(description="True if the FAISS index is loaded in memory")
    redis_connected: bool = Field(description="True if Redis is reachable; False if using in-memory fallback")
