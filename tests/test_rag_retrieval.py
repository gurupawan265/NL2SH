"""
Unit tests for NL2SH BashRetriever.
Tests query handling, ranking, metadata preservation, relevance thresholds, and context formatting.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RAG.corpus import CorpusDocument
from RAG.embedder import MockEmbedder
from RAG.retriever import BashRetriever, RetrievalResult
from RAG.vector_store import FAISSVectorStore


@pytest.fixture
def populated_retriever():
    docs = [
        CorpusDocument(
            doc_id="find_01",
            command="find",
            title="Search Files by Pattern",
            section="Search",
            content="Locates files matching criteria.",
            examples=["find /workspace -name '*.py'"],
            source_url="https://example.com/find",
            source_name="GNU Findutils",
            license="GPL",
            command_category="file_search",
            safety_level="safe",
        ),
        CorpusDocument(
            doc_id="grep_01",
            command="grep",
            title="Search Pattern in Files",
            section="Search",
            content="Searches text patterns inside files.",
            examples=["grep 'error' log.txt"],
            source_url="https://example.com/grep",
            source_name="GNU Grep",
            license="GPL",
            command_category="text_search",
            safety_level="safe",
        ),
        CorpusDocument(
            doc_id="chmod_01",
            command="chmod",
            title="Change File Mode",
            section="Permissions",
            content="Updates access bits on files.",
            examples=["chmod 755 script.sh"],
            source_url="https://example.com/chmod",
            source_name="GNU Coreutils",
            license="GPL",
            command_category="permissions",
            safety_level="file_modify",
        ),
    ]

    embedder = MockEmbedder(dimension=128)
    store = FAISSVectorStore(dimension=128)
    store.build(docs, embedder)

    retriever = BashRetriever(vector_store=store, embedder=embedder, default_top_k=2)
    return retriever


def test_empty_query_returns_empty_list(populated_retriever):
    """Test that empty, blank, or None queries safely return empty results."""
    assert populated_retriever.retrieve("") == []
    assert populated_retriever.retrieve("   ") == []
    assert populated_retriever.retrieve(None) == []


def test_retrieval_ranking_and_metadata(populated_retriever):
    """Test retrieval returns ranked RetrievalResult objects with complete metadata."""
    results = populated_retriever.retrieve("find all python files", top_k=2)
    assert len(results) == 2

    # Scores must be in descending order
    assert results[0].score >= results[1].score

    # Check fields
    top = results[0]
    assert isinstance(top, RetrievalResult)
    assert top.doc_id in ["find_01", "grep_01", "chmod_01"]
    assert top.command in ["find", "grep", "chmod"]
    assert len(top.examples) > 0
    assert "source_url" in top.metadata
    assert "license" in top.metadata


def test_retrieval_top_k_bounds(populated_retriever):
    """Test top_k clamping and edge cases."""
    assert populated_retriever.retrieve("search query", top_k=0) == []
    assert populated_retriever.retrieve("search query", top_k=-5) == []

    # top_k larger than total index size returns all documents
    res_all = populated_retriever.retrieve("search query", top_k=100)
    assert len(res_all) == 3


def test_relevance_filtering_threshold(populated_retriever):
    """Test that setting score_threshold filters out weak matches."""
    # Run without threshold
    unfiltered = populated_retriever.retrieve("search", top_k=3)
    assert len(unfiltered) == 3

    # Use a high threshold that filters some or all
    highest_score = unfiltered[0].score
    filtered = populated_retriever.retrieve("search", top_k=3, score_threshold=highest_score + 0.1)
    assert len(filtered) == 0

    # Intermediate threshold
    mid_filtered = populated_retriever.retrieve("search", top_k=3, score_threshold=highest_score - 0.001)
    assert len(mid_filtered) >= 1


def test_format_context(populated_retriever):
    """Test formatting retrieved results into an LLM context block."""
    results = populated_retriever.retrieve("search query", top_k=2)
    context_str = populated_retriever.format_context(results)

    assert "[1] Utility:" in context_str
    assert "Description:" in context_str
    assert "Examples:" in context_str

    # Empty results format test
    empty_context = populated_retriever.format_context([])
    assert "No relevant documentation found." in empty_context

