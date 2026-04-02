from __future__ import annotations

import logging
from typing import List

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from chatbot.adapters.vector_adapter import RetrievedChunk
from chatbot.config import get_settings

logger = logging.getLogger(__name__)


def get_conn() -> psycopg2.extensions.connection:
    """Open a new psycopg2 connection to Supabase using SUPABASE_DB_URL."""
    settings = get_settings()
    conn = psycopg2.connect(settings.supabase_db_url)
    register_vector(conn)
    return conn


def insert_chunks(
    collection: str,
    chunks: List[dict],
    embeddings: List[List[float]],
) -> None:
    """
    Batch-insert chunks + their embeddings into chatbot_chunks.
    Each chunk dict must have: text, source, page_num, chunk_index.
    """
    if not chunks:
        return

    rows = [
        (
            collection,
            chunks[i]["source"],
            chunks[i].get("page_num", 0),
            chunks[i].get("chunk_index", i),
            chunks[i]["text"],
            embeddings[i],
        )
        for i in range(len(chunks))
    ]

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO chatbot_chunks
                    (collection, source, page_num, chunk_index, content, embedding)
                VALUES %s
                """,
                rows,
                template="(%s, %s, %s, %s, %s, %s::vector)",
            )
        conn.commit()
        logger.info("Inserted %d chunks into Supabase (collection='%s')", len(rows), collection)
    finally:
        conn.close()


def search(
    collection: str,
    query_vec: List[float],
    k: int = 5,
) -> List[RetrievedChunk]:
    """
    Cosine similarity search using pgvector <=> operator.
    Returns up to k chunks ordered by similarity (highest first).
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, content, source, page_num,
                       1 - (embedding <=> %s::vector) AS score
                FROM chatbot_chunks
                WHERE collection = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vec, collection, query_vec, k),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        RetrievedChunk(
            vector_id=row[0],
            text=row[1],
            source=row[2],
            page_num=row[3] or 0,
            score=float(row[4]),
        )
        for row in rows
    ]


def delete_collection(collection: str) -> None:
    """Delete all chunks for a given collection."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chatbot_chunks WHERE collection = %s", (collection,))
        conn.commit()
        logger.info("Deleted collection '%s' from Supabase", collection)
    finally:
        conn.close()


def collection_exists(collection: str) -> bool:
    """Return True if at least one chunk exists for the collection."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM chatbot_chunks WHERE collection = %s LIMIT 1",
                (collection,),
            )
            return cur.fetchone() is not None
    finally:
        conn.close()
