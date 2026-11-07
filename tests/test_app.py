"""
Integration tests for app.py.
Verifies natural-language to shell translation, execution inside DockerSandbox,
exit code reporting, container cleanup, and graceful error handling.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import ask_human_approval, convert_to_shell, init_rag_pipeline, main, run_in_docker, run_interactive_session
from Sandbox.sandbox import CommandResult, DockerSandbox, SessionSandbox
from RAG.generator import GenerationResult, MockCommandGenerator, OllamaCommandGenerator
from RAG.pipeline import RAGPipeline
from RAG.embedder import MockEmbedder
from RAG.vector_store import FAISSVectorStore
from RAG.corpus import CorpusDocument


def test_convert_to_shell_mappings():
    """Verify natural-language queries generate the expected shell commands."""
    assert convert_to_shell("list files") == "ls"
    assert convert_to_shell("Please list files in folder") == "ls"
    assert convert_to_shell("current directory") == "pwd"
    assert convert_to_shell("show root folders") == "ls /"
    assert convert_to_shell("create folder") == "mkdir new_folder"
    assert convert_to_shell("unrecognized instruction") == "echo Command not understood"


def test_convert_to_shell_navigation_mappings():
    """Verify directory navigation queries generate the expected cd commands."""
    # Direct requirements
    assert convert_to_shell("change directory project") == "cd project"
    assert convert_to_shell("go to project") == "cd project"
    assert convert_to_shell("enter project") == "cd project"
    assert convert_to_shell("go back") == "cd .."
    assert convert_to_shell("go to parent directory") == "cd .."

    # Additional variants (optional "to", trimming, case insensitivity, spaces)
    assert convert_to_shell("change directory to project") == "cd project"
    assert convert_to_shell("change directory my project") == "cd 'my project'"
    assert convert_to_shell("enter 'docs folder'") == "cd 'docs folder'"
    assert convert_to_shell("go to parent directory") == "cd .."
    assert convert_to_shell("change directory") == "echo Command not understood"


def test_nl2sh_session_navigation_integration():
    """
    Real session integration test using the natural-language interface:
    1. Create directory project.
    2. Enter it using the natural-language interface.
    3. Ask for the current directory.
    4. Verify the output is /workspace/project.
    5. Also verify invalid directory navigation leaves current directory unchanged.
    """
    with SessionSandbox() as session:
        # 1. Create directory project
        res_create = session.execute("mkdir project")
        assert res_create.exit_code == 0

        # 2. Enter it using the natural-language interface
        cmd_enter = convert_to_shell("enter project")
        assert cmd_enter == "cd project"
        res_enter = session.execute(cmd_enter)
        assert res_enter.exit_code == 0
        assert session.current_dir == "/workspace/project"

        # 3. Ask for current directory
        cmd_pwd = convert_to_shell("current directory")
        assert cmd_pwd == "pwd"
        res_pwd = session.execute(cmd_pwd)
        assert res_pwd.exit_code == 0

        # 4. Verify output is /workspace/project
        assert res_pwd.stdout.strip() == "/workspace/project"

        # 5. Verify invalid directory navigation leaves current directory unchanged
        cmd_bad = convert_to_shell("enter nonexistent_folder")
        assert cmd_bad == "cd nonexistent_folder"
        res_bad = session.execute(cmd_bad)
        assert res_bad.exit_code != 0
        assert session.current_dir == "/workspace/project"

        # 6. Verify go back navigates to parent directory
        cmd_back = convert_to_shell("go back")
        assert cmd_back == "cd .."
        res_back = session.execute(cmd_back)
        assert res_back.exit_code == 0
        assert session.current_dir == "/workspace"


def test_run_in_docker_execution_real():
    """Verify run_in_docker executes inside Docker sandbox and returns output."""
    result = run_in_docker("echo hello")
    assert isinstance(result, CommandResult)
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert not result.timed_out


def test_run_in_docker_exit_code_reported():
    """Verify non-zero exit codes are correctly reported by run_in_docker."""
    result = run_in_docker("exit 42")
    assert isinstance(result, CommandResult)
    assert result.exit_code == 42


def test_run_in_docker_cleanup():
    """Verify the sandbox container is completely removed after run_in_docker finishes."""
    recorded_cid = []
    original_start = DockerSandbox.start

    def tracking_start(self):
        original_start(self)
        recorded_cid.append(self.container_id)

    with patch.object(DockerSandbox, "start", tracking_start):
        result = run_in_docker("echo cleanup_test")
        assert result.exit_code == 0

    assert len(recorded_cid) == 1
    cid = recorded_cid[0]
    assert cid is not None

    # Inspect Docker to verify the container was destroyed
    inspect_proc = subprocess.run(
        ["docker", "inspect", cid],
        capture_output=True,
        text=True
    )
    assert inspect_proc.returncode != 0, f"Container {cid} was not cleaned up!"


def test_run_in_docker_handles_docker_error():
    """Verify Docker errors are handled without raising an unhandled traceback."""
    with patch.object(DockerSandbox, "start", side_effect=RuntimeError("Docker daemon offline")):
        result = run_in_docker("echo hello")
        assert isinstance(result, CommandResult)
        assert result.exit_code == -1
        assert "Docker sandbox error" in result.stderr
        assert "Docker daemon offline" in result.stderr


def test_app_main_cli_success(monkeypatch, capsys):
    """Verify CLI main() runs end-to-end with simulated input and exits cleanly."""
    inputs = iter(["current directory", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    main()

    captured = capsys.readouterr().out
    assert "Generated shell command: pwd" in captured
    assert "/workspace" in captured
    assert "EXIT CODE: 0" in captured
    assert "Exiting." in captured


def test_app_main_cli_docker_error(capsys):
    """Verify CLI main() cleanly reports Docker sandbox errors."""
    with patch("app.SessionSandbox", side_effect=RuntimeError("Docker CLI executable not found in PATH.")):
        main()

    captured = capsys.readouterr().err
    assert "Docker sandbox error: Docker CLI executable not found in PATH." in captured


# =========================================================================
# Phase 4D: RAG Application Integration Tests
# =========================================================================

@pytest.fixture
def mock_vector_store():
    docs = [
        CorpusDocument(
            doc_id="test_find_01",
            command="find",
            title="Search Files",
            section="Search",
            content="Search files matching pattern.",
            examples=["find /workspace -name '*.py'"],
            source_url="https://example.com/find",
            source_name="Find Manual",
            license="GPL",
            command_category="file_search",
            safety_level="safe",
        )
    ]
    embedder = MockEmbedder(dimension=64)
    store = FAISSVectorStore(dimension=64)
    store.build(docs, embedder)
    return store, embedder


def test_init_rag_pipeline_provider_selection(mock_vector_store):
    """Test init_rag_pipeline supports mock, ollama, and rejects invalid providers."""
    store, emb = mock_vector_store

    # 1. Mock provider
    p_mock = init_rag_pipeline(provider="mock", embedder=emb, vector_store=store)
    assert p_mock.generator.provider_name == "mock"

    # 2. Ollama provider
    p_ollama = init_rag_pipeline(provider="ollama", model_name="qwen2.5-coder:1.5b", embedder=emb, vector_store=store)
    assert p_ollama.generator.provider_name == "ollama"
    assert p_ollama.generator.model_name == "qwen2.5-coder:1.5b"

    # 3. Invalid provider
    with pytest.raises(ValueError, match="Unsupported provider 'invalid_prov'"):
        init_rag_pipeline(provider="invalid_prov", embedder=emb, vector_store=store)


def test_init_rag_pipeline_reuses_store_and_embedder(mock_vector_store):
    """Test init_rag_pipeline reuses injected embedder and vector_store without loading disk."""
    store, emb = mock_vector_store
    p = init_rag_pipeline(provider="mock", embedder=emb, vector_store=store)
    assert p.retriever.vector_store is store
    assert p.retriever.embedder is emb


def test_ollama_unavailable_clear_error(mock_vector_store):
    """Verify Ollama connection failure returns a structured rejection without silent fallback."""
    store, emb = mock_vector_store
    p_ollama = init_rag_pipeline(
        provider="ollama",
        base_url="http://127.0.0.1:59999",  # Unused port
        embedder=emb,
        vector_store=store,
    )
    result = p_ollama.generate("Find python files")
    assert result.is_valid is False
    assert result.provider_name == "ollama"
    assert "Cannot connect to local Ollama service" in (result.rejection_reason or "")


def test_ask_human_approval_declined_by_default():
    """Test that pressing Enter or 'n' declines execution (Default: NO execution)."""
    gr = GenerationResult(
        instruction="Find files",
        command="find . -name '*.py'",
        raw_response="find . -name '*.py'",
        retrieved_doc_ids=["doc_1"],
        provider_name="mock",
        model_name="mock-model",
        is_valid=True,
    )
    # Empty input (pressing Enter)
    assert ask_human_approval(gr, input_func=lambda _: "") is False
    # Explicit 'n'
    assert ask_human_approval(gr, input_func=lambda _: "n") is False
    # Explicit 'no'
    assert ask_human_approval(gr, input_func=lambda _: "no") is False


def test_ask_human_approval_accepted_explicit():
    """Test that typing 'y' or 'yes' approves execution for safe commands."""
    gr = GenerationResult(
        instruction="List directory",
        command="ls -la",
        raw_response="ls -la",
        retrieved_doc_ids=["doc_1"],
        provider_name="mock",
        model_name="mock-model",
        is_valid=True,
        warnings=[],
    )
    assert ask_human_approval(gr, input_func=lambda _: "y") is True
    assert ask_human_approval(gr, input_func=lambda _: "yes") is True


def test_ask_human_approval_invalid_result_declined():
    """Test that invalid generation results are rejected immediately without asking to execute."""
    gr_invalid = GenerationResult(
        instruction="Bad task",
        command="",
        raw_response="error",
        is_valid=False,
        rejection_reason="Syntax error detected",
    )
    called = []
    approved = ask_human_approval(gr_invalid, input_func=lambda _: called.append("input_called"))
    assert approved is False
    assert len(called) == 0  # User is never prompted to execute an invalid command


def test_ask_human_approval_destructive_warning_requires_confirm():
    """Test that commands with destructive warnings require explicit 'CONFIRM' input."""
    gr_destructive = GenerationResult(
        instruction="Delete build files",
        command="rm -rf /workspace/build",
        raw_response="rm -rf /workspace/build",
        retrieved_doc_ids=["doc_rm"],
        provider_name="mock",
        model_name="mock-model",
        is_valid=True,
        warnings=["Destructive operation: Recursive directory deletion (rm -r)"],
    )

    # User says 'y' to first prompt, but fails to type 'CONFIRM' to second prompt
    inputs_fail = iter(["y", "yes"])
    assert ask_human_approval(gr_destructive, input_func=lambda _: next(inputs_fail)) is False

    # User says 'y' to first prompt, types 'CONFIRM' to second prompt
    inputs_ok = iter(["y", "CONFIRM"])
    assert ask_human_approval(gr_destructive, input_func=lambda _: next(inputs_ok)) is True


def test_run_interactive_session_rag_flow_execution_declined(mock_vector_store):
    """Test interactive REPL in RAG mode when user declines command execution."""
    store, emb = mock_vector_store
    canned = {"find python files": "find /workspace -name '*.py'"}
    pipeline = RAGPipeline.create_mock_pipeline(
        vector_store=store,
        embedder=emb,
        canned_responses=canned,
    )

    mock_session = MagicMock()
    mock_session.current_dir = "/workspace"

    # User inputs request, declines execution ('n'), then exits
    inputs = iter(["find python files", "n", "exit"])
    prints = []

    run_interactive_session(
        session=mock_session,
        mode="rag",
        pipeline=pipeline,
        input_func=lambda _: next(inputs),
        print_func=lambda *args, **kwargs: prints.append(" ".join(str(a) for a in args)),
    )

    # Command was generated and displayed, but NOT executed in sandbox
    output_text = "\n".join(prints)
    assert "find /workspace -name '*.py'" in output_text
    assert "Execution declined by user." in output_text
    mock_session.execute.assert_not_called()


def test_run_interactive_session_rag_flow_execution_approved(mock_vector_store):
    """Test interactive REPL in RAG mode when user explicitly approves execution ('y')."""
    store, emb = mock_vector_store
    canned = {"list all files": "ls -la"}
    pipeline = RAGPipeline.create_mock_pipeline(
        vector_store=store,
        embedder=emb,
        canned_responses=canned,
    )

    mock_session = MagicMock()
    mock_session.current_dir = "/workspace"
    mock_session.execute.return_value = CommandResult(
        stdout="file1.txt\nfile2.txt",
        stderr="",
        exit_code=0,
        timed_out=False,
        duration=0.05,
    )

    # User inputs request, approves execution ('y'), then exits
    inputs = iter(["list all files", "y", "exit"])
    prints = []

    run_interactive_session(
        session=mock_session,
        mode="rag",
        pipeline=pipeline,
        input_func=lambda _: next(inputs),
        print_func=lambda *args, **kwargs: prints.append(" ".join(str(a) for a in args)),
    )

    output_text = "\n".join(prints)
    assert "ls -la" in output_text
    assert "file1.txt" in output_text
    assert "EXIT CODE: 0" in output_text
    mock_session.execute.assert_called_once_with("ls -la")


def test_run_interactive_session_propagates_cwd(mock_vector_store):
    """Test that current_dir from session is passed to pipeline generate execution_context."""
    store, emb = mock_vector_store
    pipeline = RAGPipeline.create_mock_pipeline(vector_store=store, embedder=emb)

    mock_session = MagicMock()
    mock_session.current_dir = "/workspace/subproject"

    inputs = iter(["list files", "n", "exit"])
    run_interactive_session(
        session=mock_session,
        mode="rag",
        pipeline=pipeline,
        input_func=lambda _: next(inputs),
        print_func=lambda *args, **kwargs: None,
    )

    # Check last prompt recorded in MockCommandGenerator
    assert "Current Working Directory: /workspace/subproject" in pipeline.generator.last_prompt


def test_run_interactive_session_handles_sandbox_error(mock_vector_store):
    """Test that sandbox execution error outputs are captured and displayed without crashing."""
    store, emb = mock_vector_store
    pipeline = RAGPipeline.create_mock_pipeline(vector_store=store, embedder=emb)

    mock_session = MagicMock()
    mock_session.current_dir = "/workspace"
    mock_session.execute.return_value = CommandResult(
        stdout="",
        stderr="Sandbox container timeout error",
        exit_code=-1,
        timed_out=True,
        duration=10.0,
    )

    inputs = iter(["list files", "y", "exit"])
    prints = []

    run_interactive_session(
        session=mock_session,
        mode="rag",
        pipeline=pipeline,
        input_func=lambda _: next(inputs),
        print_func=lambda *args, **kwargs: prints.append(" ".join(str(a) for a in args)),
    )

    output_text = "\n".join(prints)
    assert "ERROR:\nSandbox container timeout error" in output_text
    assert "EXIT CODE: -1" in output_text


def test_main_cli_rag_mode_selection(mock_vector_store, monkeypatch):
    """Test CLI main() with --mode rag and --provider mock."""
    store, emb = mock_vector_store

    mock_pipeline = RAGPipeline.create_mock_pipeline(vector_store=store, embedder=emb)
    monkeypatch.setattr("app.init_rag_pipeline", lambda **kwargs: mock_pipeline)

    mock_session = MagicMock()
    mock_session.current_dir = "/workspace"
    monkeypatch.setattr("app.SessionSandbox", lambda: mock_session)

    inputs = iter(["list files", "n", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main(["--mode", "rag", "--provider", "mock"], input_func=lambda _: next(inputs))

    mock_session.close.assert_called_once()


def test_main_cli_rag_init_error(capsys, monkeypatch):
    """Test CLI main() cleanly reports RAG initialization failure."""
    monkeypatch.setattr(
        "app.init_rag_pipeline",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("Vector index missing")),
    )
    main(["--mode", "rag"])

    captured = capsys.readouterr().err
    assert "RAG initialization error: Vector index missing" in captured


def test_init_rag_pipeline_groq_success(mock_vector_store):
    """Test initializing RAG pipeline with Groq provider."""
    store, emb = mock_vector_store
    pipeline = init_rag_pipeline(
        provider="groq",
        api_key="test-key",
        model_name="qwen/qwen3.8-27b",
        vector_store=store,
        embedder=emb,
    )
    assert pipeline.generator.provider_name == "groq"
    assert pipeline.generator.model_name == "qwen/qwen3.8-27b"


def test_init_rag_pipeline_groq_missing_key_error(mock_vector_store, monkeypatch):
    """Test initializing RAG pipeline with Groq provider fails cleanly if key is missing."""
    store, emb = mock_vector_store
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("RAG.generator.load_env_file", lambda *args, **kwargs: None)

    with pytest.raises(ValueError) as excinfo:
        init_rag_pipeline(
            provider="groq",
            api_key=None,
            vector_store=store,
            embedder=emb,
        )
    assert "GROQ_API_KEY is missing or empty" in str(excinfo.value)


def test_main_cli_groq_provider_selection(mock_vector_store, monkeypatch):
    """Test CLI main() with --mode rag and --provider groq."""
    store, emb = mock_vector_store

    mock_pipeline = MagicMock()
    mock_pipeline.generator.provider_name = "groq"
    monkeypatch.setattr("app.init_rag_pipeline", lambda **kwargs: mock_pipeline)

    mock_session = MagicMock()
    mock_session.current_dir = "/workspace"
    monkeypatch.setattr("app.SessionSandbox", lambda: mock_session)

    inputs = iter(["exit"])
    main(["--mode", "rag", "--provider", "groq", "--model", "qwen/qwen3.8-27b"], input_func=lambda _: next(inputs))

    mock_session.close.assert_called_once()



