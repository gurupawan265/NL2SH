"""
Natural Language to Shell (NL2SH) application interface.
Translates natural language instructions into shell commands and executes them
safely inside a secure Docker sandbox with mandatory human approval.

Supports:
- Context-Aware RAG Mode (--mode rag): Retrieves documentation, formats context,
  generates commands via pluggable providers (mock/ollama), validates safety,
  and executes inside SessionSandbox ONLY upon explicit user approval.
- Legacy Rule Mode (--mode rule): Direct rule-based converter for navigation and basic queries.
- One-shot execution (run_in_docker) for automated task evaluation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shlex
import sys
from typing import Any, Dict, List, Optional

from Sandbox.sandbox import CommandResult, DockerSandbox, SessionSandbox


def convert_to_shell(user_input: str) -> str:
    """
    Rule-based converter translating simple English queries to shell commands.
    Preserves basic file/folder queries and provides safe directory navigation.
    """
    trimmed = user_input.strip()
    lowered = trimmed.lower()

    if "list files" in lowered:
        return "ls"
    elif "current directory" in lowered:
        return "pwd"
    elif "show root folders" in lowered:
        return "ls /"
    elif "create folder" in lowered:
        return "mkdir new_folder"
    elif lowered in ("go back", "go to parent directory"):
        return "cd .."

    # Navigation patterns: "change directory [to] <dir>", "go to <dir>", "enter <dir>"
    nav_patterns = [
        r"^change directory (?:to )?(.+)$",
        r"^go to (.+)$",
        r"^enter (.+)$",
    ]
    for pattern in nav_patterns:
        match = re.match(pattern, trimmed, re.IGNORECASE)
        if match:
            target = match.group(1).strip()
            if target.lower() == "parent directory":
                return "cd .."
            # Strip user-supplied wrapping quotes if present
            if (target.startswith('"') and target.endswith('"')) or (target.startswith("'") and target.endswith("'")):
                target = target[1:-1].strip()
            if target:
                return f"cd {shlex.quote(target)}"

    return "echo Command not understood"


def run_in_docker(command: str, timeout: float = 10.0) -> CommandResult:
    """
    Execute a shell command inside the secure Docker sandbox (one-shot mode).
    Guarantees container cleanup via context manager.
    """
    try:
        with DockerSandbox() as sandbox:
            return sandbox.execute(command, timeout=timeout)
    except Exception as exc:
        return CommandResult(
            stdout="",
            stderr=f"Docker sandbox error: {exc}",
            exit_code=-1,
            timed_out=False,
            duration=0.0
        )


def init_rag_pipeline(
    provider: str = "mock",
    model_name: Optional[str] = None,
    base_url: str = "http://localhost:11434",
    api_key: Optional[str] = None,
    adapter_path: Optional[str] = None,
    embedder: Any = None,
    vector_store: Any = None,
    canned_responses: Optional[Dict[str, str]] = None,
    default_mock_response: str = "ls -la",
) -> Any:
    """
    Initialize and return an end-to-end RAGPipeline.
    Reuses existing FAISS index and embedding engine.
    """
    from RAG.config import RAGConfig
    from RAG.embedder import SentenceTransformerEmbedder
    from RAG.generator import (
        GroqCommandGenerator,
        MockCommandGenerator,
        OllamaCommandGenerator,
        QwenSFTCommandGenerator,
    )
    from RAG.pipeline import RAGPipeline
    from RAG.retriever import BashRetriever
    from RAG.vector_store import FAISSVectorStore

    cfg = RAGConfig()

    # Reuse or load vector store
    if vector_store is None:
        if not cfg.vectorstore_dir.is_dir():
            raise FileNotFoundError(
                f"RAG vector store directory not found at {cfg.vectorstore_dir}. "
                "Run 'py -3.11 Scripts/build_rag_index.py' to build the index."
            )
        vector_store = FAISSVectorStore.load(cfg.vectorstore_dir)

    # Reuse or load embedder
    if embedder is None:
        embedder = SentenceTransformerEmbedder(
            model_name=cfg.model_name,
            device=cfg.device,
            normalize=cfg.normalize_embeddings,
        )

    retriever = BashRetriever(
        vector_store=vector_store,
        embedder=embedder,
        default_top_k=cfg.default_top_k,
    )

    prov_clean = provider.strip().lower()
    if prov_clean == "mock":
        generator = MockCommandGenerator(
            canned_responses=canned_responses,
            default_response=default_mock_response,
            model_name=model_name or "mock-model-v1",
        )
    elif prov_clean == "ollama":
        generator = OllamaCommandGenerator(
            model_name=model_name or "qwen2.5-coder:1.5b",
            base_url=base_url,
            timeout=10.0,
        )
    elif prov_clean == "groq":
        generator = GroqCommandGenerator(
            api_key=api_key,
            model_name=model_name,
            timeout=15.0,
        )
    elif prov_clean in ("qwen-sft", "qwensft"):
        generator = QwenSFTCommandGenerator(
            adapter_path=adapter_path,
            base_model_name=model_name,
        )
    else:
        raise ValueError(
            f"Unsupported provider '{provider}'. Supported providers are: 'mock', 'ollama', 'groq', 'qwen-sft'."
        )

    return RAGPipeline(retriever=retriever, generator=generator)


def ask_human_approval(
    gen_result: Any,
    input_func=None,
    print_func=None,
) -> bool:
    """
    Display command and metadata, then require explicit user approval.
    Default behavior is strictly NO execution (pressing Enter or 'n' declines).
    Commands with destructive or privileged warnings require an additional explicit 'CONFIRM' input.
    """
    _input = input_func if input_func is not None else input
    _print = print_func if print_func is not None else print

    if not gen_result.is_valid:
        if gen_result.is_clarification:
            _print(f"\n[CLARIFICATION NEEDED]: {gen_result.rejection_reason}")
        else:
            _print(f"\n[GENERATION REJECTED]: {gen_result.rejection_reason}")
        return False

    _print("\n------------------------------------------------------------")
    _print(f"Generated Command: {gen_result.command}")
    _print(f"Provider:          {gen_result.provider_name} (model: {gen_result.model_name})")
    if gen_result.retrieved_doc_ids:
        _print(f"Retrieved Context: {', '.join(gen_result.retrieved_doc_ids)}")

    has_warnings = bool(gen_result.warnings)
    if has_warnings:
        _print("\n> [!WARNING] POTENTIAL SAFETY RISKS DETECTED:")
        for w in gen_result.warnings:
            _print(f">   - {w}")

    _print("------------------------------------------------------------")

    # Primary confirmation prompt (Default is NO execution)
    prompt_msg = "Execute this command in secure Docker sandbox? [y/N]: "
    try:
        user_choice = _input(prompt_msg).strip().lower()
    except (EOFError, KeyboardInterrupt):
        _print("\nExecution declined.")
        return False

    if user_choice not in ("y", "yes"):
        _print("Execution declined by user.")
        return False

    # Secondary confirmation if warnings exist
    if has_warnings:
        _print("\nATTENTION: This command triggered safety warnings.")
        try:
            confirm_choice = _input("Type 'CONFIRM' to proceed with sandbox execution: ").strip()
        except (EOFError, KeyboardInterrupt):
            _print("\nExecution cancelled.")
            return False

        if confirm_choice != "CONFIRM":
            _print("Execution cancelled: Confirmation phrase did not match.")
            return False

    return True


def run_interactive_session(
    session: SessionSandbox,
    mode: str = "rag",
    pipeline: Any = None,
    input_func=None,
    print_func=None,
) -> None:
    """
    Run the interactive REPL with SessionSandbox in either 'rag' or 'rule' mode.
    """
    _input = input_func if input_func is not None else input
    _print = print_func if print_func is not None else print

    _print("=" * 60)
    _print("NL2SH Interactive Shell")
    _print(f"Mode: {mode.upper()}")
    if mode == "rag" and pipeline:
        prov = getattr(pipeline.generator, "provider_name", "unknown")
        model = getattr(pipeline.generator, "model_name", "unknown")
        _print(f"Provider: {prov} ({model})")
        if prov == "mock":
            _print("[Note: Running with Mock generator (simulated responses for testing/offline)]")
    _print("Type your request, 'help' for info, or 'exit' / 'quit' to quit.")
    _print("=" * 60)

    while True:
        prompt = f"nl2sh:{session.current_dir}$ "
        try:
            user_input = _input(prompt)
        except (EOFError, KeyboardInterrupt):
            _print("\nExiting.")
            break

        cleaned_input = user_input.strip()
        if not cleaned_input:
            continue

        if cleaned_input.lower() in ("exit", "quit"):
            _print("Exiting.")
            break

        if cleaned_input.lower() == "help":
            _print("Commands:")
            _print("  exit / quit       - Exit the session")
            _print("  <natural language> - Translate and execute shell command")
            continue

        command_to_run: Optional[str] = None

        if mode == "rule":
            # Legacy rule-based mode
            command_to_run = convert_to_shell(cleaned_input)
            _print(f"Generated shell command: {command_to_run}")
        else:
            # Context-aware RAG mode
            if pipeline is None:
                _print("Error: RAG pipeline is not initialized.", file=sys.stderr)
                break

            _print("[Retrieving documentation and generating command...]")
            exec_ctx = {"current_dir": session.current_dir}
            gen_result = pipeline.generate(
                instruction=cleaned_input,
                execution_context=exec_ctx,
            )

            # Ask human approval before execution
            approved = ask_human_approval(gen_result, input_func=_input, print_func=_print)
            if approved:
                command_to_run = gen_result.command

        if command_to_run:
            result = session.execute(command_to_run)
            if result.stdout:
                _print("OUTPUT:\n" + result.stdout, end="" if result.stdout.endswith("\n") else "\n")
            if result.stderr:
                _print("ERROR:\n" + result.stderr, end="" if result.stderr.endswith("\n") else "\n")
            _print(f"EXIT CODE: {result.exit_code}")


def main(argv: Optional[List[str]] = None, input_func=None, print_func=None) -> None:
    """CLI entry point for NL2SH application."""
    parser = argparse.ArgumentParser(description="NL2SH: Natural Language to Shell Interface")
    default_mode = os.environ.get("NL2SH_MODE", "rule")
    parser.add_argument(
        "--mode",
        choices=["rag", "rule"],
        default=default_mode,
        help="Execution mode: 'rag' (retrieval-augmented) or 'rule' (legacy rule-based). Default: rule."
    )
    parser.add_argument(
        "--rag",
        action="store_const",
        dest="mode",
        const="rag",
        help="Shortcut to run in RAG mode."
    )
    parser.add_argument(
        "--rule",
        action="store_const",
        dest="mode",
        const="rule",
        help="Shortcut to run in rule mode."
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "ollama", "groq", "qwen-sft"],
        default="mock",
        help="Model provider in RAG mode ('mock', 'ollama', 'groq', or 'qwen-sft'). Default: mock."
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Path to fine-tuned LoRA adapter directory for 'qwen-sft' provider."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for provider. Defaults: qwen2.5-coder:1.5b (ollama), qwen/qwen3.8-27b (groq), Qwen/Qwen2.5-Coder-3B-Instruct (qwen-sft)."
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API base URL. Default: http://localhost:11434."
    )

    args, _ = parser.parse_known_args(argv)

    # Initialize RAG pipeline if in RAG mode
    pipeline = None
    if args.mode == "rag":
        try:
            pipeline = init_rag_pipeline(
                provider=args.provider,
                model_name=args.model,
                base_url=args.ollama_url,
                adapter_path=args.adapter_path,
            )
        except Exception as exc:
            print(f"RAG initialization error: {exc}", file=sys.stderr)
            return

    # Start SessionSandbox
    try:
        session = SessionSandbox()
    except Exception as exc:
        print(f"Docker sandbox error: {exc}", file=sys.stderr)
        return

    try:
        run_interactive_session(
            session,
            mode=args.mode,
            pipeline=pipeline,
            input_func=input_func,
            print_func=print_func,
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
