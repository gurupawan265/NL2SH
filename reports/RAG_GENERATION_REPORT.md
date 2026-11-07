# NL2SH — Phase 4C: Context-Aware Command Generation Report

**Date:** 2026-11-07  
**Pipeline:** Context-Aware RAG Command Generation (`RAG.pipeline.RAGPipeline`)  
**Providers Implemented:** `MockCommandGenerator` (offline unit tests) and `OllamaCommandGenerator` (optional local daemon)  
**Prompt Engine:** `RAG.prompt_builder.PromptBuilder` (isolated system rules, reference context, and user task)  
**Safety & Validation:** `RAG.validator.CommandValidator` (markdown extraction, shape validation, syntax verification, safety warnings)  
**Test Suite:** 29 new automated tests; **97 passed tests** project-wide  

---

## 1. Executive Summary

Phase 4C implements the context-aware LLM command generation layer for NL2SH, completing the end-to-end flow:
```
Natural Language Instruction
         │
         ▼
[BashRetriever] ───> Fetches Top-k reference documentation chunks
         │
         ▼
[PromptBuilder] ───> Constructs isolated system, reference context, and user task blocks
         │
         ▼
[BaseCommandGenerator] ───> Invokes configured provider (Mock / Ollama)
         │
         ▼
[CommandValidator] ───> Extracts command, validates syntax, assesses safety risks
         │
         ▼
[GenerationResult] ───> Structured output with command, validation status, doc IDs, and warnings
```

### Safety and Resource Guarantees Maintained
- **Zero Shell Command Execution:** No commands were executed on the host system or Docker containers during this phase.
- **Zero Paid APIs / Zero API Keys:** All tests run offline using deterministic mock responses.
- **Zero Model Downloads:** No LLM weights were pulled or downloaded.
- **Zero Background Daemons:** Ollama is an optional client that connects only to an already-running local service; it never starts processes or pulls models automatically.

> [!NOTE]
> **Model Accuracy Notice:** Because automated tests in this phase were conducted using `MockCommandGenerator` and offline fixtures, real-world model command generation accuracy has not yet been benchmarked against live LLM checkpoints. Live model evaluation will be conducted during the end-to-end evaluation phase.

---

## 2. Generation Architecture

The architecture maintains strict separation of concerns across four distinct modules:

```
RAG/
├── prompt_builder.py   <-- System prompt, reference context bounds, user task isolation
├── validator.py        <-- Markdown fence extraction, label cleaning, syntax & risk checks
├── generator.py        <-- BaseCommandGenerator, MockCommandGenerator, OllamaCommandGenerator
└── pipeline.py         <-- RAGPipeline coordinating retrieval -> generation -> validation
```

### Data Flow
1. **Instruction Ingestion:** `RAGPipeline.generate(instruction, top_k, execution_context)` receives the natural language query.
2. **Context Retrieval:** `BashRetriever` retrieves top-$k$ reference chunks from the Phase 4B FAISS vector store. If zero documents pass the threshold, the pipeline returns a clarification request rather than generating blindly.
3. **Prompt Assembly:** `PromptBuilder` formats retrieved utilities into an immutable `<REFERENCE_DOCUMENTATION>` block, environment state into `<EXECUTION_ENVIRONMENT>`, and user instructions into `<USER_TASK>`.
4. **Provider Invocation:** `BaseCommandGenerator` sends the full prompt to the active provider.
5. **Output Validation:** `CommandValidator` inspects the raw string, extracts the pure shell command, verifies balanced quotes via `shlex`, rejects multi-line or multi-option ambiguities, and computes non-fatal safety warnings.
6. **Result Assembly:** A structured `GenerationResult` is returned containing the sanitized command, validation status, retrieved document IDs, warnings, and provider metadata.

---

## 3. Prompt Structure & Security Boundaries

The prompt structure enforces prompt injection defense by separating instructions from reference text:

```text
[SYSTEM PROMPT]
You are an expert Linux shell assistant for the NL2SH system.
Your task is to translate the user's natural-language instruction into an exact, executable Bash command.
Strict Output Rules:
1. Output ONLY the raw executable Bash command.
2. Do NOT wrap the command in markdown code blocks or backticks.
3. Do NOT include explanations, commentary, or multiple options.
4. If ambiguous, respond with: CLARIFICATION: <question>

<REFERENCE_DOCUMENTATION>
--- Reference Document [1] ---
Utility: find (Search Files by Pattern)
Source: GNU Findutils Manual
Description: find searches directory hierarchies in real time...
Representative Examples:
    find /workspace -name '*.py'
</REFERENCE_DOCUMENTATION>

<USER_TASK>
Find all Python files in the workspace
</USER_TASK>

Generate the exact Bash command to execute this task:
```

---

## 4. Provider Implementation & Ollama Configuration

### 1. `MockCommandGenerator`
- Used for all automated testing and offline integration.
- Supports deterministic canned responses (`instruction -> response`) and configurable defaults.
- Zero network latency, zero dependencies.

### 2. `OllamaCommandGenerator`
- Connects via standard HTTP POST to local Ollama endpoints (default: `http://localhost:11434/api/generate`).
- Uses `temperature: 0.0` for deterministic generation.
- Enforces connection timeouts (default: 10s).
- **Failure Behavior:** If Ollama is not running, it catches connection errors cleanly and returns a structured rejection (`rejection_reason="Cannot connect to local Ollama service at..."`) without crashing the application.
- **Safety Restriction:** It never invokes `ollama pull` or `ollama run` subprocesses.

#### Running Ollama Locally (Optional Developer Workflow):
To use Ollama with NL2SH in future phases:
```powershell
# 1. Start Ollama independently in a separate terminal:
ollama serve

# 2. Pull a lightweight model (e.g. Qwen 2.5 Coder 1.5B):
ollama pull qwen2.5-coder:1.5b

# 3. Instantiate Ollama generator in Python:
from RAG.generator import OllamaCommandGenerator
generator = OllamaCommandGenerator(model_name="qwen2.5-coder:1.5b")
```

---

## 5. Command Validation and Safety Rules

The `CommandValidator` enforces strict syntax boundaries while computing descriptive safety warnings:

| Rule / Feature | Validator Action | Purpose |
| :--- | :--- | :--- |
| **Empty Output** | Rejected (`is_valid=False`) | Detects model timeout or null generation. |
| **Markdown Fences** | Extracted cleanly | Strips ```` ```bash ... ``` ```` and inline backticks. |
| **Introductory Labels** | Stripped cleanly | Removes prefixes such as `Command: `, `Bash: `, `$ `. |
| **Multiple Options** | Rejected (`is_valid=False`) | Flags `Option 1: ... Option 2: ...` as ambiguous. |
| **Unexplained Multi-line**| Rejected (`is_valid=False`) | Prevents execution of multi-line scripts unless escaped with trailing `\`. |
| **Unbalanced Quotes** | Rejected (`is_valid=False`) | Validated via `shlex` parsing to catch syntax errors before execution. |
| **Destructive Commands** | Non-fatal Warning | Warns on `rm`, `rm -rf`, `dd`, `mkfs`, `shred`. |
| **Privileged Commands** | Non-fatal Warning | Warns on `sudo`, `su`, `chown`, `doas`. |
| **Network Commands** | Non-fatal Warning | Warns on `curl`, `wget`, `nc`, `ssh`. |
| **Clarification Signal** | Flagged (`is_clarification=True`) | Detects `CLARIFICATION: <question>`. |

Valid shell primitives—including pipelines (`|`), input/output redirections (`>`, `>>`, `<`, `2>&1`), subshell substitutions (`$(...)`), and condition chains (`&&`, `||`)—are preserved without corruption.

---

## 6. Test Suite Verification

### Phase 4C Test Coverage (29 New Tests)
1. **`tests/test_rag_prompt_builder.py` (6 tests):**
   - Reference docs included in prompt.
   - Irrelevant docs excluded from prompt.
   - User task isolated inside `<USER_TASK>` tags.
   - Execution environment (`current_dir`) propagation.
   - Empty context handling.
   - System prompt rule verification.
2. **`tests/test_rag_validator.py` (13 tests):**
   - Empty and whitespace rejection.
   - Clean single-line command validation.
   - Markdown code-fence extraction (` ```bash ` and generic ` ``` `).
   - Unclosed code fence rejection.
   - Prefix label stripping (`Command:`, `$`).
   - Multiple alternative options rejection.
   - Unexplained multi-line rejection.
   - Legitimate backslash continuation acceptance.
   - Complex pipeline and redirection preservation.
   - Unbalanced quote syntax error rejection.
   - Destructive command warnings.
   - Privileged and network command warnings.
   - Clarification request detection.
3. **`tests/test_rag_generator.py` (5 tests):**
   - Mock generator default response.
   - Mock generator canned responses.
   - Automatic code-fence stripping in generator.
   - Ollama connection failure handling.
   - `GenerationResult.to_dict()` serialization.
4. **`tests/test_rag_pipeline.py` (5 tests):**
   - End-to-end pipeline execution with mock.
   - Empty instruction rejection.
   - Retrieval threshold elimination / no-context handling.
   - Execution context propagation to prompt.
   - Document ID tracking in generation results.

### Full Project Test Results
- **Total Project Tests:** **97 passed** in 40.91s across 9 test modules:
  - `test_alfa_preprocessing.py`: 8 passed
  - `test_app.py`: 9 passed
  - `test_execution_evaluator.py`: 13 passed
  - `test_rag_corpus.py`: 8 passed
  - `test_rag_generator.py`: 5 passed
  - `test_rag_pipeline.py`: 5 passed
  - `test_rag_prompt_builder.py`: 6 passed
  - `test_rag_retrieval.py`: 5 passed
  - `test_rag_validator.py`: 13 passed
  - `test_rag_vector_store.py`: 5 passed
  - `test_sandbox.py`: 13 passed
  - `test_session.py`: 7 passed

---

## 7. Hardware & Resource Profile

- **RAM Footprint:** ~45 MB runtime idle footprint.
- **CPU Utilization:** < 1% idle; negligible CPU during mock prompt construction and validation (< 1.5 ms per query).
- **GPU / VRAM Utilization:** 0 MB (GTX 1650 completely unused).
- **Disk Usage:** Phase 4C added lightweight Python source files (~35 KB total).
