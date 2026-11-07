"""
Unit tests for NL2SH FAISS vector store.
Tests building, dimension validation, persistence, and safe JSON metadata loading.
Uses MockEmbedder for fast, offline, and deterministic execution.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import numpy as np
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RAG.corpus import CorpusDocument
from RAG.embedder import MockEmbedder
from RAG.vector_store import FAISSVectorStore


@pytest.fixture
def sample_docs():
    return [
        CorpusDocument(
            doc_id="doc_1",
            command="find",
            title="Find files by name",
            section="Search",
            content="Find files matching pattern.",
            examples=["find . -name '*.py'"],
            source_url="https://example.com/find",
            source_name="Find Manual",
            license="GPL",
            command_category="file_search",
            safety_level="safe",
        ),
        CorpusDocument(
            doc_id="doc_2",
            command="grep",
            title="Search text in files",
            section="Search",
            content="Search regular expressions in files.",
            examples=["grep 'error' log.txt"],
            source_url="https://example.com/grep",
            source_name="Grep Manual",
            license="GPL",
            command_category="text_search",
            safety_level="safe",
        ),
        CorpusDocument(
            doc_id="doc_3",
            command="chmod",
            title="Change file modes",
            section="Permissions",
            content="Change file permissions.",
            examples=["chmod 755 run.sh"],
            source_url="https://example.com/chmod",
            source_name="Chmod Manual",
            license="GPL",
            command_category="permissions",
            safety_level="file_modify",
        ),
    ]


def test_build_and_search_index(sample_docs):
    """Test building in-memory FAISS index and querying vectors."""
    embedder = MockEmbedder(dimension=128)
    store = FAISSVectorStore(dimension=128)

    store.build(sample_docs, embedder)

    assert store.is_built is True
    assert store.count == 3
    assert store.dimension == 128

    # Query with mock embedding
    query_vec = embedder.embed_query("find python files")
    results = store.search(query_vec, top_k=2)

    assert len(results) == 2
    for doc, score in results:
        assert isinstance(doc, CorpusDocument)
        assert isinstance(score, float)


def test_embedding_dimension_mismatch(sample_docs):
    """Test that embedding dimension mismatches raise ValueError."""
    embedder = MockEmbedder(dimension=64)
    store = FAISSVectorStore(dimension=128)

    with pytest.raises(ValueError, match="Expected embedding dimension 128, got 64"):
        store.build(sample_docs, embedder)


def test_index_persistence_and_reload(sample_docs):
    """Test serializing index to disk and reloading it cleanly."""
    embedder = MockEmbedder(dimension=128)
    store = FAISSVectorStore(dimension=128)
    store.build(sample_docs, embedder)

    with tempfile.TemporaryDirectory() as tmpdir:
        store.save(tmpdir)

        index_file = Path(tmpdir) / "index.faiss"
        meta_file = Path(tmpdir) / "metadata.json"
        assert index_file.is_file()
        assert meta_file.is_file()

        # Reload store from disk
        loaded_store = FAISSVectorStore.load(tmpdir)
        assert loaded_store.is_built is True
        assert loaded_store.count == 3
        assert loaded_store.dimension == 128
        assert len(loaded_store.documents) == 3

        # Verify search produces identical results
        query_vec = embedder.embed_query("search text")
        orig_results = store.search(query_vec, top_k=3)
        loaded_results = loaded_store.search(query_vec, top_k=3)

        assert [d.doc_id for d, s in orig_results] == [d.doc_id for d, s in loaded_results]
        assert np.allclose([s for d, s in orig_results], [s for d, s in loaded_results], atol=1e-5)


def test_load_missing_files_handling():
    """Test that loading from a directory with missing files raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError, match="index file missing"):
            FAISSVectorStore.load(tmpdir)


def test_reproducible_rebuilding(sample_docs):
    """Test that rebuilding the index produces identical vectors with deterministic embedder."""
    embedder = MockEmbedder(dimension=128)

    store1 = FAISSVectorStore(dimension=128)
    store1.build(sample_docs, embedder)

    store2 = FAISSVectorStore(dimension=128)
    store2.build(sample_docs, embedder)

    q = embedder.embed_query("test query")
    res1 = store1.search(q, top_k=3)
    res2 = store2.search(q, top_k=3)

    assert [d.doc_id for d, s in res1] == [d.doc_id for d, s in res2]
    assert np.allclose([s for d, s in res1], [s for d, s in res2], atol=1e-5)

