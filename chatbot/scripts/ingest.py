#!/usr/bin/env python3
"""
Document ingestion script for the Personal RAG Chatbot.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --input data/input --collection default
    python scripts/ingest.py --help

Loads all .pdf, .txt, .md, and .html files from the input directory,
chunks them, embeds them using the configured embedding model, and
saves the FAISS index to data/indexed/<collection>/.
"""
from __future__ import annotations

import argparse
import logging
import sys
import os

# Allow running from project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Load .env before importing settings
from dotenv import load_dotenv
load_dotenv()

from chatbot.services.ingestion_service import ingest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG chatbot vector store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input directory containing documents (default: DATA_INPUT_DIR from .env)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for FAISS index (default: DATA_INDEXED_DIR from .env)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Collection name for the index (default: DEFAULT_COLLECTION from .env)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Personal RAG Chatbot — Document Ingestion")
    print("=" * 60)

    try:
        result = ingest(
            input_dir=args.input,
            indexed_dir=args.output,
            collection=args.collection,
        )
        print()
        print("✓ Ingestion complete.")
        print(f"  Documents processed : {result.documents_processed}")
        print(f"  Total chunks created: {result.total_chunks}")
        print(f"  Vectors indexed     : {result.vectors_indexed}")
        print(f"  Index saved to      : {result.index_path}")
        print(f"  Time elapsed        : {result.elapsed_seconds}s")
        print()
        print("You can now start the chatbot and query your documents.")
    except FileNotFoundError as exc:
        print(f"\n✗ Error: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        print(f"\n✗ Ingestion failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
