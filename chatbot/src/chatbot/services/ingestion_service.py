from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import List

from langchain_community.document_loaders import BSHTMLLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import TokenTextSplitter

from chatbot.adapters import llm_adapter, vector_adapter
from chatbot.config import get_settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".htm"}
UPLOAD_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".jpg", ".jpeg", ".png", ".webp", ".gif", ".tiff", ".tif"}


@dataclass
class IngestionResult:
    documents_processed: int
    total_chunks: int
    vectors_indexed: int
    elapsed_seconds: float
    collection: str
    index_path: str


# ── Step 1: Discover Input Documents ──────────────────────────────────────────

def discover_documents(input_dir: str) -> List[str]:
    """Recursively find all supported documents in input_dir."""
    paths: List[str] = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.startswith("."):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                paths.append(os.path.join(root, filename))
    logger.info("Discovered %d documents in %s", len(paths), input_dir)
    return paths


# ── Step 2: Load and Chunk Documents ──────────────────────────────────────────

def load_and_chunk_documents(file_paths: List[str], chunk_size: int, chunk_overlap: int):
    """
    Load each document, split into overlapping token-based chunks.
    Returns list of dicts: {text, source, page_num, chunk_index}
    """
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks: List[dict] = []

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext in (".html", ".htm"):
                loader = BSHTMLLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding="utf-8")

            docs = loader.load()
        except Exception as exc:
            logger.error("Failed to load %s: %s — skipping", file_path, exc)
            continue

        for doc in docs:
            page_num = doc.metadata.get("page", 0)
            chunks = splitter.split_text(doc.page_content)

            for idx, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue
                all_chunks.append({
                    "text": chunk_text,
                    "source": filename,
                    "page_num": int(page_num),
                    "chunk_index": idx,
                })

        logger.info("Loaded and chunked: %s (%d chunks)", filename, len(all_chunks))

    logger.info("Total chunks after processing all documents: %d", len(all_chunks))
    return all_chunks


# ── Steps 3–5: Embed → Index → Save → Report ──────────────────────────────────

def ingest(
    input_dir: str | None = None,
    indexed_dir: str | None = None,
    collection: str | None = None,
) -> IngestionResult:
    """
    Full ingestion pipeline:
    1. Discover documents
    2. Load and chunk
    3. Embed all chunks
    4. Build and save FAISS index
    5. Report results
    """
    settings = get_settings()
    input_dir = input_dir or settings.data_input_dir
    indexed_dir = indexed_dir or settings.data_indexed_dir
    collection = collection or settings.default_collection

    start_time = time.time()

    # Step 1
    file_paths = discover_documents(input_dir)
    if not file_paths:
        raise FileNotFoundError(
            f"No supported documents found in '{input_dir}'. "
            f"Add .pdf, .txt, .md, or .html files and retry."
        )

    # Step 2
    chunks = load_and_chunk_documents(
        file_paths,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise ValueError("No text could be extracted from the provided documents.")

    # Step 3: Embed all chunks
    texts = [c["text"] for c in chunks]
    embeddings = llm_adapter.embed_texts(texts)

    # Step 4: Build metadata dict and save FAISS index
    metadata = {
        i: {
            "text": chunks[i]["text"],
            "source": chunks[i]["source"],
            "page_num": chunks[i]["page_num"],
            "chunk_index": chunks[i]["chunk_index"],
        }
        for i in range(len(chunks))
    }

    index_path = os.path.join(indexed_dir, collection)
    vector_adapter.save_index(embeddings, metadata, indexed_dir, collection)

    elapsed = time.time() - start_time

    # Step 5: Report
    result = IngestionResult(
        documents_processed=len(file_paths),
        total_chunks=len(chunks),
        vectors_indexed=len(embeddings),
        elapsed_seconds=round(elapsed, 1),
        collection=collection,
        index_path=index_path,
    )

    logger.info(
        "Ingestion complete — docs=%d, chunks=%d, vectors=%d, time=%.1fs, path=%s",
        result.documents_processed,
        result.total_chunks,
        result.vectors_indexed,
        result.elapsed_seconds,
        result.index_path,
    )
    return result


# ── Upload Pipeline (single file → Supabase) ──────────────────────────────────

@dataclass
class UploadResult:
    collection: str
    source: str
    chunks_indexed: int
    elapsed_seconds: float


def ingest_file(
    file_path: str,
    collection: str,
) -> UploadResult:
    """
    Ingest a single uploaded file into Supabase pgvector.
    1. Extract text via extractors.route()
    2. Chunk via TokenTextSplitter
    3. Embed via llm_adapter.embed_texts()
    4. Insert into Supabase via supabase_adapter.insert_chunks()
    """
    from chatbot.adapters import supabase_adapter
    from chatbot.services import extractors

    settings = get_settings()
    start_time = time.time()
    source = os.path.basename(file_path)

    # Step 1: Extract raw chunks from the file
    raw_chunks = extractors.route(file_path)
    if not raw_chunks:
        raise ValueError(f"No text could be extracted from '{source}'.")

    # Step 2: Split each chunk's text with TokenTextSplitter
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    final_chunks: List[dict] = []
    for raw in raw_chunks:
        sub_texts = splitter.split_text(raw["text"])
        for idx, sub_text in enumerate(sub_texts):
            if sub_text.strip():
                final_chunks.append({
                    "text": sub_text,
                    "source": raw["source"],
                    "page_num": raw.get("page_num", 0),
                    "chunk_index": idx,
                })

    if not final_chunks:
        raise ValueError(f"No text chunks produced from '{source}'.")

    # Step 3: Embed all chunks
    texts = [c["text"] for c in final_chunks]
    embeddings = llm_adapter.embed_texts(texts)

    # Step 4: Insert into Supabase
    supabase_adapter.insert_chunks(collection, final_chunks, embeddings)

    elapsed = round(time.time() - start_time, 1)
    logger.info(
        "Upload ingestion complete — collection='%s', source='%s', chunks=%d, time=%.1fs",
        collection, source, len(final_chunks), elapsed,
    )
    return UploadResult(
        collection=collection,
        source=source,
        chunks_indexed=len(final_chunks),
        elapsed_seconds=elapsed,
    )
