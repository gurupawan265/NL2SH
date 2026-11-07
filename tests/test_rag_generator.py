"""
Unit tests for NL2SH Command Generators.
Tests MockCommandGenerator and OllamaCommandGenerator error handling.
Ensures zero external network calls, zero model downloads, and zero shell execution.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RAG.generator import (
    BaseCommandGenerator,
    GenerationResult,
    MockCommandGenerator,
    OllamaCommandGenerator,
)
from RAG.retriever import RetrievalResult


@pytest.fixture
def sample_context():
    return [
        RetrievalResult(
            doc_id="doc_ls_01",
            command="ls",
            title="List Files",
            section="Navigation",
            content="Lists directory contents.",
            examples=["ls -la"],
            score=0.9,
            command_category="navigation",
            safety_level="safe",
            metadata={"source_name": "Coreutils"},
        )
    ]


def test_mock_generator_deterministic_default(sample_context):
    """Test MockCommandGenerator returns configured default response."""
    gen = MockCommandGenerator(default_response="pwd")
    res = gen.generate("Where am I?", sample_context)

    assert isinstance(res, GenerationResult)
    assert res.is_valid is True
    assert res.command == "pwd"
    assert res.provider_name == "mock"
    assert res.retrieved_doc_ids == ["doc_ls_01"]
    assert gen.invocation_count == 1


def test_mock_generator_canned_responses(sample_context):
    """Test MockCommandGenerator returns mapped responses matching keywords."""
    canned = {
        "count lines": "wc -l < file.txt",
        "find python": "find /workspace -name '*.py'",
    }
    gen = MockCommandGenerator(canned_responses=canned, default_response="echo default")

    r1 = gen.generate("count lines in log", sample_context)
    assert r1.command == "wc -l < file.txt"

    r2 = gen.generate("find python scripts", sample_context)
    assert r2.command == "find /workspace -name '*.py'"

    r3 = gen.generate("unknown instruction", sample_context)
    assert r3.command == "echo default"


def test_mock_generator_with_markdown_fence(sample_context):
    """Test that generator automatically validates and strips markdown code fences."""
    gen = MockCommandGenerator(default_response="```bash\nfind . -type f\n```")
    res = gen.generate("Find files", sample_context)

    assert res.is_valid is True
    assert res.command == "find . -type f"
    assert "```" not in res.command


def test_ollama_generator_connection_failure_clear_error(sample_context):
    """
    Test that OllamaCommandGenerator handles unavailable service gracefully.
    Attempts to connect to an unreachable port, verifies clean ConnectionError without crashing.
    """
    # Connect to an invalid/unbound local port with short timeout
    unreachable_gen = OllamaCommandGenerator(
        model_name="qwen2.5-coder:1.5b",
        base_url="http://127.0.0.1:54321",  # Unused port
        timeout=1.0,
    )

    res = unreachable_gen.generate("List files", sample_context)

    assert res.is_valid is False
    assert res.command == ""
    assert res.rejection_reason is not None
    assert "Cannot connect to local Ollama service" in res.rejection_reason
    assert "54321" in res.rejection_reason


def test_generation_result_to_dict():
    """Test serialization of GenerationResult to dictionary."""
    gr = GenerationResult(
        instruction="List directory",
        command="ls -l",
        raw_response="ls -l",
        retrieved_doc_ids=["doc_1"],
        provider_name="mock",
        model_name="test-model",
        is_valid=True,
        warnings=[],
        rejection_reason=None,
    )
    d = gr.to_dict()
    assert d["instruction"] == "List directory"
    assert d["command"] == "ls -l"
    assert d["provider_name"] == "mock"
    assert d["is_valid"] is True

