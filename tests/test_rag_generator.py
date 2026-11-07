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
    GroqCommandGenerator,
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


def test_groq_missing_api_key_raises_error(monkeypatch):
    """Test that missing GROQ_API_KEY raises a clear ValueError on initialization."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # Also patch load_env_file so it doesn't read the disk .env file in this test
    monkeypatch.setattr("RAG.generator.load_env_file", lambda *args, **kwargs: None)

    with pytest.raises(ValueError) as excinfo:
        GroqCommandGenerator(api_key=None)
    assert "GROQ_API_KEY is missing or empty" in str(excinfo.value)


def test_groq_provider_configuration(monkeypatch):
    """Test GroqCommandGenerator configuration and model selection."""
    gen = GroqCommandGenerator(
        api_key="test-api-key",
        model_name="custom-code-model",
    )
    assert gen.provider_name == "groq"
    assert gen.model_name == "custom-code-model"
    assert gen.timeout == 15.0


def test_groq_successful_mocked_sdk_response(sample_context):
    """Test successful command generation using mocked Groq SDK client."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "find /workspace -name '*.txt'"
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage.model_dump.return_value = {
        "prompt_tokens": 42,
        "completion_tokens": 12,
        "total_tokens": 54,
    }

    mock_client.chat.completions.create.return_value = mock_resp

    gen = GroqCommandGenerator(
        api_key="test-api-key",
        model_name="qwen/qwen3.8-27b",
        client=mock_client,
    )

    res = gen.generate("Find text files", sample_context)

    assert res.is_valid is True
    assert res.command == "find /workspace -name '*.txt'"
    assert res.provider_name == "groq"
    assert res.model_name == "qwen/qwen3.8-27b"
    assert res.token_usage == {
        "prompt_tokens": 42,
        "completion_tokens": 12,
        "total_tokens": 54,
    }
    assert res.latency is not None


def test_groq_authentication_failure_handled_and_redacted(sample_context):
    """Test that authentication error is handled, returned as rejection, and key is redacted."""
    from unittest.mock import MagicMock
    import groq

    mock_client = MagicMock()
    # Simulate an auth error from groq with sensitive key info
    secret_key = "gsk_supersecretkey1234567890abcdef"
    auth_err = groq.AuthenticationError(
        f"Invalid token: Bearer {secret_key}",
        response=MagicMock(status_code=401),
        body={"error": {"message": f"Invalid key {secret_key}"}},
    )
    mock_client.chat.completions.create.side_effect = auth_err

    gen = GroqCommandGenerator(
        api_key=secret_key,
        client=mock_client,
    )

    res = gen.generate("List files", sample_context)

    assert res.is_valid is False
    assert res.command == ""
    assert res.provider_name == "groq"
    assert "authentication failed" in res.rejection_reason
    # STRICT SECURITY: Secret key must NEVER appear in rejection reason
    assert secret_key not in res.rejection_reason
    assert "[REDACTED" in res.rejection_reason


def test_groq_rate_limit_failure_handled(sample_context):
    """Test that rate limit error is handled cleanly without crashing."""
    from unittest.mock import MagicMock
    import groq

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = groq.RateLimitError(
        "Rate limit exceeded: 30 requests per minute",
        response=MagicMock(status_code=429),
        body={"error": {"message": "Rate limit exceeded"}},
    )

    gen = GroqCommandGenerator(
        api_key="test-key",
        client=mock_client,
    )

    res = gen.generate("List files", sample_context)

    assert res.is_valid is False
    assert "rate limit" in res.rejection_reason.lower()


def test_groq_timeout_handling(sample_context):
    """Test that request timeout is handled cleanly."""
    from unittest.mock import MagicMock
    import groq

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = groq.APITimeoutError(
        request=MagicMock()
    )

    gen = GroqCommandGenerator(
        api_key="test-key",
        client=mock_client,
        timeout=5.0,
    )

    res = gen.generate("List files", sample_context)

    assert res.is_valid is False
    assert "timed out" in res.rejection_reason.lower()


def test_groq_no_silent_fallback_to_mock(sample_context):
    """Ensure Groq provider NEVER falls back silently to Mock or Ollama on failure."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("Fatal Groq API failure")

    gen = GroqCommandGenerator(
        api_key="test-key",
        client=mock_client,
    )

    res = gen.generate("List files", sample_context)

    assert res.is_valid is False
    assert res.command == ""
    # Provider remains strictly 'groq', never 'mock'
    assert res.provider_name == "groq"
    assert res.command != "ls -la"


