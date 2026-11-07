"""
Unit tests for NL2SH RAGPipeline.
Verifies end-to-end coordination of retriever, prompt builder, generator, and validator.
All dependencies are injected and use mocks; no network calls or shell executions occur.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RAG.corpus import CorpusDocument
from RAG.embedder import MockEmbedder
from RAG.generator import GenerationResult, MockCommandGenerator
from RAG.pipeline import RAGPipeline
from RAG.prompt_builder import PromptBuilder
from RAG.retriever import BashRetriever, RetrievalResult
from RAG.validator import CommandValidator
from RAG.vector_store import FAISSVectorStore


@pytest.fixture
def mock_pipeline():
    docs = [
        CorpusDocument(
            doc_id="doc_find_01",
            command="find",
            title="Search Files",
            section="Search",
            content="Searches directory hierarchies.",
            examples=["find . -name '*.sh'"],
            source_url="https://example.com",
            source_name="Findutils",
            license="GPL",
            command_category="file_search",
            safety_level="safe",
        ),
        CorpusDocument(
            doc_id="doc_grep_01",
            command="grep",
            title="Search Text",
            section="Search",
            content="Matches regular expressions.",
            examples=["grep 'error' app.log"],
            source_url="https://example.com",
            source_name="Grep",
            license="GPL",
            command_category="text_search",
            safety_level="safe",
        ),
    ]

    embedder = MockEmbedder(dimension=128)
    store = FAISSVectorStore(dimension=128)
    store.build(docs, embedder)

    retriever = BashRetriever(vector_store=store, embedder=embedder, default_top_k=2)

    canned = {
        "find shell scripts": "find /workspace -name '*.sh'",
        "search error": "grep -i 'error' /workspace/log.txt",
    }
    generator = MockCommandGenerator(
        canned_responses=canned,
        default_response="ls -l",
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        prompt_builder=PromptBuilder(),
        validator=CommandValidator(),
    )
    return pipeline


def test_pipeline_end_to_end_success(mock_pipeline):
    """Test full pipeline execution producing validated GenerationResult."""
    result = mock_pipeline.generate("find shell scripts")

    assert isinstance(result, GenerationResult)
    assert result.is_valid is True
    assert result.command == "find /workspace -name '*.sh'"
    assert result.provider_name == "mock"
    assert len(result.retrieved_doc_ids) > 0
    assert result.rejection_reason is None


def test_pipeline_empty_instruction(mock_pipeline):
    """Test pipeline rejects empty or blank instructions before calling retriever or LLM."""
    res1 = mock_pipeline.generate("")
    assert res1.is_valid is False
    assert "Empty instruction" in res1.rejection_reason

    res2 = mock_pipeline.generate("   \t  ")
    assert res2.is_valid is False
    assert "Empty instruction" in res2.rejection_reason


def test_pipeline_no_retrieval_context_handling(mock_pipeline):
    """Test pipeline behavior when score threshold eliminates all retrieved documents."""
    # Set an impossible threshold (score > 10.0) so 0 documents are returned
    res = mock_pipeline.generate("search something", score_threshold=10.0)

    assert res.is_valid is False
    assert "No relevant Bash documentation retrieved" in res.rejection_reason
    assert res.is_clarification is True


def test_pipeline_execution_context_propagation(mock_pipeline):
    """Test execution context (e.g. current_dir) is propagated through the pipeline to prompt."""
    mock_pipeline.generate(
        "find shell scripts",
        execution_context={"current_dir": "/workspace/custom_dir"},
    )

    # Check last prompt recorded by MockCommandGenerator
    last_prompt = mock_pipeline.generator.last_prompt
    assert "Current Working Directory: /workspace/custom_dir" in last_prompt


def test_pipeline_preserves_retrieved_doc_ids(mock_pipeline):
    """Test that all retrieved document IDs are recorded in the generation result."""
    result = mock_pipeline.generate("search error in logs")
    assert result.is_valid is True
    assert "doc_find_01" in result.retrieved_doc_ids or "doc_grep_01" in result.retrieved_doc_ids

