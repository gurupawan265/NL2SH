"""
Unit tests for NL2SH PromptBuilder.
Verifies prompt construction, reference documentation formatting, user task isolation,
and execution environment inclusion.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RAG.prompt_builder import PromptBuilder, SYSTEM_PROMPT
from RAG.retriever import RetrievalResult


@pytest.fixture
def sample_retrieved_docs():
    return [
        RetrievalResult(
            doc_id="find_01",
            command="find",
            title="Search Files by Pattern",
            section="Search",
            content="Searches directory trees matching name patterns.",
            examples=["find /workspace -name '*.py'"],
            score=0.85,
            command_category="file_search",
            safety_level="safe",
            metadata={"source_name": "GNU Findutils Manual"},
        ),
        RetrievalResult(
            doc_id="find_02",
            command="find",
            title="Filter by Modification Time",
            section="Search",
            content="Matches files modified within specified days.",
            examples=["find /workspace -mtime -7"],
            score=0.72,
            command_category="file_search",
            safety_level="safe",
            metadata={"source_name": "GNU Findutils Manual"},
        ),
    ]


def test_relevant_documentation_included(sample_retrieved_docs):
    """Test that retrieved documentation is formatted and included in prompt."""
    builder = PromptBuilder()
    prompt = builder.build_user_prompt(
        instruction="Find Python files modified recently",
        context_documents=sample_retrieved_docs,
    )

    assert "Utility: find (Search Files by Pattern)" in prompt
    assert "find /workspace -name '*.py'" in prompt
    assert "Filter by Modification Time" in prompt
    assert "find /workspace -mtime -7" in prompt
    assert "GNU Findutils Manual" in prompt


def test_irrelevant_corpus_documents_excluded(sample_retrieved_docs):
    """Test that documents not in retrieved context are NOT in prompt."""
    builder = PromptBuilder()
    prompt = builder.build_user_prompt(
        instruction="Find Python files",
        context_documents=sample_retrieved_docs,
    )

    # Utilities not in sample_retrieved_docs must not appear
    assert "chmod" not in prompt
    assert "tar -czf" not in prompt
    assert "grep -i" not in prompt


def test_user_task_isolated_from_reference(sample_retrieved_docs):
    """Test that user task is strictly partitioned inside <USER_TASK> tags."""
    builder = PromptBuilder()
    instruction = "Delete all temporary .tmp files"
    prompt = builder.build_user_prompt(instruction, sample_retrieved_docs)

    assert "<USER_TASK>" in prompt
    assert "</USER_TASK>" in prompt
    assert "<REFERENCE_DOCUMENTATION>" in prompt
    assert "</REFERENCE_DOCUMENTATION>" in prompt

    # User task must be inside <USER_TASK> section
    user_section = prompt.split("<USER_TASK>")[1].split("</USER_TASK>")[0].strip()
    assert user_section == instruction


def test_execution_context_inclusion(sample_retrieved_docs):
    """Test that execution context (e.g. current working directory) is included when provided."""
    builder = PromptBuilder()
    exec_ctx = {"current_dir": "/workspace/project"}
    prompt = builder.build_user_prompt("List files", sample_retrieved_docs, execution_context=exec_ctx)

    assert "Execution Environment:" in prompt
    assert "Current Working Directory: /workspace/project" in prompt


def test_empty_context_handling():
    """Test prompt building when zero documents are retrieved."""
    builder = PromptBuilder()
    prompt = builder.build_user_prompt("arbitrary task", context_documents=[])

    assert "No reference documentation available." in prompt
    assert "<USER_TASK>" in prompt
    assert "arbitrary task" in prompt


def test_system_prompt_rules():
    """Test that default system prompt contains strict output instructions."""
    builder = PromptBuilder()
    full_prompt = builder.build_full_prompt("List files", [])

    assert "Output ONLY the raw executable Bash command" in full_prompt
    assert "Do NOT wrap the command in markdown" in full_prompt
    assert "CLARIFICATION:" in full_prompt
    assert "Do NOT invent flags" in full_prompt

