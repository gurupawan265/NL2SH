# NL2SH — Phase 4D: RAG Application Integration Report

**Date:** 2026-11-07  
**Component:** Interactive CLI Application (`app.py`)  
**Pipeline Integrated:** `RAG.pipeline.RAGPipeline` with `BashRetriever`, `PromptBuilder`, `CommandGenerator`, `CommandValidator`  
**Execution Environment:** `Sandbox.sandbox.SessionSandbox` (isolated Docker container)  
**Safety Protocol:** Strict Human Approval with Mandatory Two-Step `'CONFIRM'` verification for risky operations  
**Test Suite:** 22 tests in `tests/test_app.py`; **110 passing tests** repository-wide  

---

## 1. Executive Summary

Phase 4D completes the end-to-end integration of the NL2SH system, connecting the context-aware RAG pipeline implemented in Phase 4B/4C to the interactive terminal application (`app.py`) and persistent Docker sandbox (`SessionSandbox`).

```
                              User Natural Language Query
                                          │
                                          ▼
                             [app.py / Interactive REPL]
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   ▼                                             ▼
           [--mode rag (Default)]                         [--mode rule]
                   │                                             │
                   ▼                                             ▼
     [RAGPipeline.generate()]                         [Rule-based Generator]
     - BashRetriever (FAISS)                                     │
     - PromptBuilder                                             │
     - Mock / Ollama Generator                                   │
     - CommandValidator                                          │
                   │                                             │
                   └──────────────────────┬──────────────────────┘
                                          │
                                          ▼
                            [Ask Human Approval Workflow]
                              - Displays: Command, CWD,
                                Sources, Warnings
                              - Press Enter / 'n': DECLINE (Default)
                              - Destructive / Privileged: Requires 'CONFIRM'
                                          │
                                          ▼
                            [Execute in SessionSandbox]
                              - Persistent Docker container
                              - Real-time stdout, stderr, exit code
                              - Updates working directory (cwd)
```

### Safety and Isolation Guarantees
- **Strict Human Approval Before Execution:** Generated commands are never executed automatically. The user is presented with the proposed command, retrieved documentation context, model provider, and safety warnings.
- **Default Rejection:** Pressing Enter or typing anything other than `y` or `yes` immediately declines execution.
- **Mandatory Risk Confirmation:** When destructive operations (`rm`, `dd`, `mkfs`) or privileged operations (`sudo`, `su`, `chown`) are detected, the user is required to type the explicit confirmation phrase `'CONFIRM'`.
- **Zero Host Execution:** All commands execute strictly inside `SessionSandbox` within an isolated Linux container (`nl2sh-interactive-sandbox`). The host operating system (Windows) is never exposed to shell commands.
- **Fail-Fast Error Handling:** If `--provider ollama` is requested but the local daemon is unavailable or returns an error, the application halts cleanly with an informative diagnostic message rather than silently falling back to insecure defaults.

---

## 2. Interactive Application Architecture

### 2.1 Component Integration

`app.py` acts as the orchestrator connecting four decoupled components:

1. **Pipeline Initialization (`init_rag_pipeline`)**:
   - Lazily loads or reuses FAISS vector store and `SentenceTransformer` embedder (`all-MiniLM-L6-v2`).
   - Instantiates `BashRetriever`, `PromptBuilder`, `CommandValidator`, and the chosen `BaseCommandGenerator` subclass (`MockCommandGenerator` or `OllamaCommandGenerator`).
   - Assembles and returns a thread-safe `RAGPipeline`.

2. **Execution Sandbox Session (`SessionSandbox`)**:
   - Leverages persistent interactive Docker execution developed in Phase 1.
   - Maintains working directory state across commands (`current_dir = session.cwd`).
   - Injects `current_dir` into `RAGPipeline.generate(nl_instruction, execution_context=...)` so generated commands are aware of relative paths and active environment state.

3. **Approval Protocol (`ask_human_approval`)**:
   - Formats and displays command metadata:
     - Proposed Bash command.
     - Retrieved documentation sources and IDs.
     - Active provider and model name.
     - Prominent warning banners if risks are detected.
   - Evaluates input with safe fallback: any input other than `y`/`yes` returns `False`.
   - Flags with severity `CONFIRM` trigger a secondary prompt:
     `[!] High-risk command detected. Type 'CONFIRM' to proceed:`

4. **Interactive REPL (`run_interactive_session`)**:
   - Prompts user with natural language prompt (`nl2sh> `).
   - Supports built-in commands: `exit`, `quit`, `help`, `status`, `clear`.
   - Supports runtime mode switching between `--mode rag` and `--mode rule`.
   - Displays execution output cleanly separated by stdout, stderr, and non-zero exit code indicators.

---

## 3. CLI Usage & Options

The application CLI is invoked via `app.py`:

```bash
# Default: Context-aware RAG mode with offline mock generator
python app.py

# Explicit RAG mode selection
python app.py --mode rag
python app.py --rag

# Legacy rule-based mode
python app.py --mode rule
python app.py --rule

# Local Ollama LLM provider
python app.py --mode rag --provider ollama --model deepseek-r1:8b

# Custom Ollama endpoint
python app.py --mode rag --provider ollama --ollama-url http://127.0.0.1:11434
```

### Argument Reference

| Option | Values | Default | Description |
|---|---|---|---|
| `--mode` | `rag`, `rule` | `rag` | Mode selection: RAG generation or legacy rules |
| `--rag` | flag | — | Alias for `--mode rag` |
| `--rule` | flag | — | Alias for `--mode rule` |
| `--provider` | `mock`, `ollama` | `mock` | LLM generation provider |
| `--model` | string | `bash-generator-v1` (mock) / `qwen2.5-coder:7b` (ollama) | LLM model identifier |
| `--ollama-url` | URL | `http://localhost:11434` | Ollama daemon base URL |

---

## 4. Provider Behavior & Failure Modes

### Offline Mock Provider (`--provider mock`)
- Standard default for development, offline testing, and environments without local LLMs.
- Deterministic command generation for 15+ common command patterns (directory listing, searching, process inspection, file modification).
- Instant turnaround (< 2 ms) with zero external network or daemon dependencies.

### Local Ollama Provider (`--provider ollama`)
- Connects via HTTP to an existing local Ollama daemon (`http://localhost:11434`).
- Strict connection check before generation: validates endpoint reachability and reports specific connection errors (`ConnectionRefusedError`, HTTP 500, timeout).
- **No silent fallback:** If Ollama is requested but unavailable, the error is displayed prominently and the command is rejected, preserving user transparency and predictability.

---

## 5. Security & Human Approval Workflow

### Standard Verification Display
```
------------------------------------------------------------
[NL2SH] Generated Command:
  grep -rn "error" /var/log/
  Mode: rag | Provider: mock | Model: bash-generator-v1
  Retrieved Context: grep (grep_grep), find (find_find)
------------------------------------------------------------
Execute this command in sandbox? [y/N]: 
```
*Pressing Enter or typing `n` rejects execution.*

### High-Risk Warning & Two-Step Confirmation
```
------------------------------------------------------------
[NL2SH] Generated Command:
  rm -rf /tmp/scratch/*
  Mode: rag | Provider: mock | Model: bash-generator-v1
  [!] WARNINGS:
      - Command contains potentially destructive operation: 'rm'
------------------------------------------------------------
Execute this command in sandbox? [y/N]: y
[!] High-risk command detected. Type 'CONFIRM' to proceed: CONFIRM
[+] Execution approved by user.
```

---

## 6. Test Suite & Verification Results

The test suite in `tests/test_app.py` was expanded from 9 baseline tests to 22 comprehensive integration tests:

| Test Case | Scope | Status |
|---|---|---|
| `test_app_imports` | Module import integrity | Passed |
| `test_parse_instruction_rule_based` | Legacy rule-based parsing | Passed |
| `test_init_rag_pipeline_mock` | Mock pipeline initialization | Passed |
| `test_init_rag_pipeline_reuses_store_and_embedder` | Store & embedder reuse | Passed |
| `test_init_rag_pipeline_ollama` | Ollama provider wiring | Passed |
| `test_init_rag_pipeline_ollama_failure` | Ollama failure handling | Passed |
| `test_ask_human_approval_default_declines` | Default rejection on Enter | Passed |
| `test_ask_human_approval_declines_explicit_no` | Rejection on 'n' | Passed |
| `test_ask_human_approval_approves_yes` | Approval on 'y' | Passed |
| `test_ask_human_approval_invalid_generation` | Immediate rejection on invalid syntax | Passed |
| `test_ask_human_approval_destructive_warning_confirmed` | Confirmation of high-risk warning | Passed |
| `test_ask_human_approval_destructive_warning_declined` | Decline of high-risk warning | Passed |
| `test_repl_rag_mode_execution_declined` | REPL declined workflow | Passed |
| `test_repl_rag_mode_execution_approved` | REPL approved execution in sandbox | Passed |
| `test_repl_rag_mode_cwd_passed_to_generator` | Dynamic CWD propagation | Passed |
| `test_repl_rag_mode_sandbox_error_handling` | Container execution failure handling | Passed |
| `test_repl_rule_mode_execution_approved` | Rule mode fallback execution | Passed |
| `test_main_cli_rag_mode_selection` | CLI `--mode rag` argument dispatch | Passed |
| `test_main_cli_rag_init_failure` | Clean exit on RAG init failure | Passed |
| `test_app_main_interactive_exit` | Clean exit on REPL `exit` command | Passed |
| `test_app_main_cli_success` | Rule-based CLI execution | Passed |
| `test_app_main_cli_docker_error` | Docker daemon error handling | Passed |

### Full Test Suite Execution Summary
- **Total Tests Executed:** 110 tests across 12 test modules.
- **Results:** 110 passed, 0 failed, 0 errors.
- **Execution Time:** ~46.8 seconds.
- **Untouched Benchmark Data:** `data/raw/alfa_test.csv` verified 100% unaltered.

---

## 7. Resource Footprint

- **Host Memory Utilization:** ~150 MB (Python process + FAISS index + `all-MiniLM-L6-v2` embedding model in RAM).
- **GPU / VRAM Utilization:** 0 MB (NVIDIA GTX 1650 completely unutilized during mock mode).
- **Docker Container:** ~30 MB idle memory overhead for lightweight alpine/debian container.
- **Disk Footprint:** Added negligible Python code (~10 KB) without introducing new heavy dependencies.

---

## 8. Conclusion

Phase 4D establishes a complete, safe, and robust bridge between natural language instructions, context-aware RAG command generation, user governance, and isolated container execution. The system is now ready for end-to-end evaluation and model fine-tuning workflows in Phase 5.
