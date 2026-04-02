from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page_num: int  # 0-indexed from loader, converted to 1-indexed in response
    score: float   # Higher = more similar (inner product after L2 normalization)
    vector_id: int


# Module-level cache: collection_name → (faiss_index, metadata_dict)
_index_cache: Dict[str, Tuple[faiss.Index, Dict[int, dict]]] = {}


def _index_path(indexed_dir: str, collection: str) -> str:
    return os.path.join(indexed_dir, collection)


def load_index(indexed_dir: str, collection: str = "default") -> bool:
    """
    Load FAISS index and metadata from disk into module-level cache.
    Returns True if loaded successfully, False if index files not found.
    Thread-safe for concurrent reads; not safe for concurrent writes.
    """
    if collection in _index_cache:
        return True  # Already cached

    path = _index_path(indexed_dir, collection)
    faiss_file = os.path.join(path, "index.faiss")
    meta_file = os.path.join(path, "index.pkl")

    if not os.path.exists(faiss_file) or not os.path.exists(meta_file):
        logger.warning("FAISS index not found at %s. Run scripts/ingest.py first.", path)
        return False

    index = faiss.read_index(faiss_file)
    with open(meta_file, "rb") as f:
        metadata: Dict[int, dict] = pickle.load(f)

    _index_cache[collection] = (index, metadata)
    logger.info(
        "FAISS index loaded: collection='%s', vectors=%d, path=%s",
        collection, index.ntotal, path,
    )
    return True


def reload_index(indexed_dir: str, collection: str = "default") -> bool:
    """Force-reload index from disk (clears cache entry first)."""
    _index_cache.pop(collection, None)
    return load_index(indexed_dir, collection)


def is_index_loaded(collection: str = "default") -> bool:
    return collection in _index_cache


def search(
    query_vector: List[float],
    k: int = 5,
    collection: str = "default",
) -> List[RetrievedChunk]:
    """
    Search the FAISS index for the k most similar chunks.
    Returns a list of RetrievedChunk sorted by score descending.
    """
    if collection not in _index_cache:
        raise RuntimeError(f"FAISS index for collection '{collection}' is not loaded.")

    index, metadata = _index_cache[collection]

    # Normalize query vector for cosine similarity (IndexFlatIP)
    vec = np.array([query_vector], dtype=np.float32)
    faiss.normalize_L2(vec)

    scores, ids = index.search(vec, k)

    results: List[RetrievedChunk] = []
    for score, vid in zip(scores[0], ids[0]):
        if vid == -1:
            continue  # FAISS returns -1 when fewer than k results exist
        meta = metadata.get(int(vid), {})
        results.append(RetrievedChunk(
            text=meta.get("text", ""),
            source=meta.get("source", "unknown"),
            page_num=meta.get("page_num", 0),
            score=float(score),
            vector_id=int(vid),
        ))

    return results


def save_index(
    embeddings: List[List[float]],
    metadata: Dict[int, dict],
    indexed_dir: str,
    collection: str = "default",
) -> None:
    """
    Build a new FAISS IndexFlatIP, add all embeddings, and save index + metadata to disk.
    Embeddings are L2-normalized before indexing for cosine similarity.
    """
    path = _index_path(indexed_dir, collection)
    os.makedirs(path, exist_ok=True)

    vectors = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    faiss_file = os.path.join(path, "index.faiss")
    meta_file = os.path.join(path, "index.pkl")

    faiss.write_index(index, faiss_file)
    with open(meta_file, "wb") as f:
        pickle.dump(metadata, f)

    # Update the in-memory cache so the app can serve queries immediately after ingestion
    _index_cache[collection] = (index, metadata)

    logger.info(
        "FAISS index saved: collection='%s', vectors=%d, path=%s",
        collection, index.ntotal, path,
    )
