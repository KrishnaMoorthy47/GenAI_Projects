from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("API_KEY", "test-secret")
os.environ.setdefault("OPENAI_API_KEY", "sk-test")


class TestDiscoverDocuments:
    def test_finds_supported_extensions(self):
        from chatbot.services.ingestion_service import discover_documents

        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["a.pdf", "b.txt", "c.md", "d.html", "e.xyz", ".hidden.pdf"]:
                open(os.path.join(tmpdir, name), "w").close()

            found = discover_documents(tmpdir)

        names = [os.path.basename(f) for f in found]
        assert "a.pdf" in names
        assert "b.txt" in names
        assert "c.md" in names
        assert "d.html" in names
        assert "e.xyz" not in names       # unsupported extension
        assert ".hidden.pdf" not in names  # hidden file

    def test_returns_empty_for_empty_dir(self):
        from chatbot.services.ingestion_service import discover_documents

        with tempfile.TemporaryDirectory() as tmpdir:
            result = discover_documents(tmpdir)
        assert result == []


class TestLoadAndChunk:
    def test_chunks_text_file(self):
        from chatbot.services.ingestion_service import load_and_chunk_documents

        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("This is a test document. " * 50)
            f.flush()
            path = f.name

        try:
            chunks = load_and_chunk_documents([path], chunk_size=50, chunk_overlap=20)
            assert len(chunks) >= 1
            for chunk in chunks:
                assert "text" in chunk
                assert "source" in chunk
                assert "page_num" in chunk
                assert chunk["text"].strip()
        finally:
            os.unlink(path)

    def test_skips_unreadable_file(self):
        from chatbot.services.ingestion_service import load_and_chunk_documents
        # Non-existent file should be skipped without crashing
        result = load_and_chunk_documents(["/nonexistent/path/file.txt"], chunk_size=100, chunk_overlap=50)
        assert result == []


class TestVectorAdapter:
    def test_save_and_load_index(self):
        from chatbot.adapters import vector_adapter

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake embeddings (dimension 4 for speed)
            embeddings = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
            metadata = {
                0: {"text": "chunk one", "source": "a.pdf", "page_num": 0},
                1: {"text": "chunk two", "source": "b.pdf", "page_num": 1},
            }

            vector_adapter.save_index(embeddings, metadata, tmpdir, collection="test-col")

            # Verify files were created
            assert os.path.exists(os.path.join(tmpdir, "test-col", "index.faiss"))
            assert os.path.exists(os.path.join(tmpdir, "test-col", "index.pkl"))

            # Load and search
            assert vector_adapter.is_index_loaded("test-col")
            results = vector_adapter.search([0.1, 0.2, 0.3, 0.4], k=2, collection="test-col")
            assert len(results) >= 1
            assert any(r.source == "a.pdf" for r in results)

    def test_load_returns_false_for_missing_index(self):
        from chatbot.adapters import vector_adapter
        with tempfile.TemporaryDirectory() as tmpdir:
            result = vector_adapter.load_index(tmpdir, collection="nonexistent")
        assert result is False

    def test_is_index_loaded_false_before_load(self):
        from chatbot.adapters import vector_adapter
        assert not vector_adapter.is_index_loaded("definitely-not-loaded-xyz")
