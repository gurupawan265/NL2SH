"""
NL2SH Groq RAG Evaluation Runner.

Executes real-world benchmark tasks through:
1. BashRetriever (FAISS document retrieval)
2. PromptBuilder (Deterministic prompt structuring)
3. GroqCommandGenerator (Production inference with code-capable LLM)
4. CommandValidator (Command extraction and safety checks)
5. DockerSandbox (Isolated container execution with required fixtures)
6. Semantic Verification (Ground-truth filesystem & stdout validation)

Outputs comprehensive evaluation metrics and saves results separately to
data/evaluation/groq_evaluation_results.jsonl without overwriting Phase 3 results.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from RAG.config import RAGConfig
from RAG.embedder import SentenceTransformerEmbedder
from RAG.generator import GroqCommandGenerator, GenerationResult, load_env_file
from RAG.pipeline import RAGPipeline
from RAG.retriever import BashRetriever
from RAG.vector_store import FAISSVectorStore
from Sandbox.sandbox import DockerSandbox, CommandResult
from Scripts.evaluate_commands import setup_fixtures_in_sandbox, verify_semantic_correctness


def init_evaluation_pipeline(
    model_name: Optional[str] = None,
    timeout: float = 15.0,
) -> RAGPipeline:
    """Initialize RAG pipeline configured with GroqCommandGenerator."""
    load_env_file()
    cfg = RAGConfig()

    if not cfg.vectorstore_dir.is_dir():
        raise FileNotFoundError(
            f"Vector store directory not found at {cfg.vectorstore_dir}. "
            "Please build index first."
        )

    vector_store = FAISSVectorStore.load(cfg.vectorstore_dir)
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
    generator = GroqCommandGenerator(
        model_name=model_name,
        timeout=timeout,
    )
    return RAGPipeline(retriever=retriever, generator=generator)


def load_existing_results(results_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load existing evaluation results keyed by record_id to enable resume and reuse."""
    existing: Dict[str, Dict[str, Any]] = {}
    if results_path.is_file():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        rid = record.get("record_id")
                        if rid:
                            existing[rid] = record
        except Exception as exc:
            print(f"[Warning] Could not parse existing results at {results_path}: {exc}", file=sys.stderr)
    return existing


def evaluate_task(
    task: Dict[str, Any],
    pipeline: RAGPipeline,
    sandbox_timeout: float = 10.0,
    max_api_retries: int = 1,
) -> Dict[str, Any]:
    """
    Run a single benchmark task through RAG generation and Docker execution.
    """
    record_id = task.get("record_id", "unknown")
    instruction = task.get("instruction", "")
    ref_cmd = task.get("reference_command", "")
    category = task.get("command_category", "unknown")
    fixtures = task.get("required_fixtures", {})
    expected = task.get("expected_result", {})
    method = task.get("verification_method", "exit_code")

    task_result: Dict[str, Any] = {
        "record_id": record_id,
        "instruction": instruction,
        "reference_command": ref_cmd,
        "command_category": category,
        "verification_method": method,
        # Generation metadata
        "generated_command": "",
        "raw_response": "",
        "retrieved_doc_ids": [],
        "provider": pipeline.generator.provider_name,
        "model": pipeline.generator.model_name,
        "is_valid": False,
        "warnings": [],
        "rejection_reason": None,
        "is_clarification": False,
        "generation_latency": 0.0,
        "token_usage": None,
        "api_success": False,
        # Execution metadata
        "executed": False,
        "execution_success": False,
        "semantic_pass": False,
        "exit_code": None,
        "stdout": "",
        "stderr": "",
        "execution_duration": 0.0,
        "timed_out": False,
        "failure_reason": None,
        "quarantined": False,
        "exact_match": False,
    }

    # 1. Generation Step (with retry for transient API errors)
    gen_result: Optional[GenerationResult] = None
    last_err: Optional[str] = None

    for attempt in range(max_api_retries + 1):
        try:
            gen_result = pipeline.generate(instruction)
            break
        except (PermissionError, ValueError) as fatal_err:
            # Fatal authentication or configuration errors: DO NOT RETRY
            task_result["failure_reason"] = f"FATAL_API_ERROR: {fatal_err}"
            return task_result
        except Exception as trans_err:
            last_err = str(trans_err)
            if attempt < max_api_retries:
                time.sleep(1.0)

    if gen_result is None or not gen_result.is_valid:
        if gen_result:
            task_result["generated_command"] = gen_result.command
            task_result["raw_response"] = gen_result.raw_response
            task_result["retrieved_doc_ids"] = gen_result.retrieved_doc_ids
            task_result["warnings"] = gen_result.warnings
            task_result["rejection_reason"] = gen_result.rejection_reason
            task_result["is_clarification"] = gen_result.is_clarification
            task_result["generation_latency"] = gen_result.latency or 0.0
            task_result["token_usage"] = gen_result.token_usage
            task_result["api_success"] = bool(gen_result.raw_response)
        else:
            task_result["rejection_reason"] = f"API_CALL_FAILED: {last_err}"
            task_result["api_success"] = False

        task_result["failure_reason"] = f"GENERATION_REJECTED: {task_result['rejection_reason']}"
        return task_result

    # Record successful generation
    task_result["api_success"] = True
    task_result["is_valid"] = True
    task_result["generated_command"] = gen_result.command
    task_result["raw_response"] = gen_result.raw_response
    task_result["retrieved_doc_ids"] = gen_result.retrieved_doc_ids
    task_result["warnings"] = gen_result.warnings
    task_result["generation_latency"] = gen_result.latency or 0.0
    task_result["token_usage"] = gen_result.token_usage
    task_result["exact_match"] = (gen_result.command.strip() == ref_cmd.strip())

    # 2. Safety Policy Evaluation
    # If the command contains high-risk warnings or network operations not permitted, quarantine
    cmd_to_run = gen_result.command.strip()
    if not cmd_to_run:
        task_result["failure_reason"] = "EMPTY_COMMAND_EXTRACTED"
        return task_result

    # Benchmark safety check: network access is disabled
    if any("network" in w.lower() for w in gen_result.warnings) and not task.get("requires_network", False):
        task_result["quarantined"] = True
        task_result["failure_reason"] = "SAFETY_QUARANTINE: Network command generated for non-network task"
        return task_result

    # 3. Execution in isolated DockerSandbox
    task_result["executed"] = True
    try:
        with DockerSandbox() as sandbox:
            # Inject fixtures
            try:
                setup_fixtures_in_sandbox(sandbox, fixtures)
            except Exception as fix_exc:
                task_result["failure_reason"] = f"FIXTURE_SETUP_ERROR: {fix_exc}"
                return task_result

            # Execute generated command
            t_exec_start = time.perf_counter()
            cmd_res = sandbox.execute(cmd_to_run, timeout=sandbox_timeout)
            exec_duration = time.perf_counter() - t_exec_start

            task_result["exit_code"] = cmd_res.exit_code
            task_result["stdout"] = cmd_res.stdout
            task_result["stderr"] = cmd_res.stderr
            task_result["execution_duration"] = round(exec_duration, 4)
            task_result["timed_out"] = cmd_res.timed_out

            expected_exit = expected.get("exit_code", 0)
            task_result["execution_success"] = (
                cmd_res.exit_code == expected_exit and not cmd_res.timed_out
            )

            # Verify semantic correctness against expected outcome
            passed, failure_reason = verify_semantic_correctness(
                sandbox, cmd_res, expected, method
            )
            task_result["semantic_pass"] = passed
            task_result["failure_reason"] = failure_reason

    except Exception as container_exc:
        task_result["failure_reason"] = f"DOCKER_SANDBOX_ERROR: {container_exc}"

    return task_result


def run_groq_evaluation(
    benchmark_file: str = "data/evaluation/evaluation_benchmark.jsonl",
    results_file: str = "data/evaluation/groq_evaluation_results.jsonl",
    limit: Optional[int] = None,
    delay_between_requests: float = 0.5,
) -> Dict[str, Any]:
    """
    Run evaluation sequentially across tasks.
    Reuses existing completed tasks to avoid duplicate API requests.
    """
    bench_path = Path(benchmark_file)
    results_path = Path(results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if not bench_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")

    with open(bench_path, "r", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    if limit is not None:
        tasks = tasks[:limit]

    existing_results = load_existing_results(results_path)

    print(f"Loaded {len(tasks)} tasks to evaluate (Existing cached: {len(existing_results)})")

    # Initialize RAG Pipeline
    pipeline = init_evaluation_pipeline()
    print(f"RAG Pipeline initialized with Groq model: {pipeline.generator.model_name}")

    all_results: List[Dict[str, Any]] = []

    for idx, task in enumerate(tasks, start=1):
        rid = task.get("record_id", f"task_{idx}")
        instruction = task.get("instruction", "")

        # Check if already evaluated
        if rid in existing_results:
            print(f"[{idx:3d}/{len(tasks):3d}] {rid}: [CACHED] -> Semantic Pass: {existing_results[rid].get('semantic_pass')}")
            all_results.append(existing_results[rid])
            continue

        print(f"[{idx:3d}/{len(tasks):3d}] {rid}: '{instruction[:45]}...' -> Generating...", end="", flush=True)

        res = evaluate_task(task, pipeline)
        all_results.append(res)

        status_msg = "SEMANTIC_PASS" if res["semantic_pass"] else f"FAIL ({res.get('failure_reason')})"
        print(f" [{res['generation_latency']}s gen, {res['execution_duration']}s exec] -> {status_msg}")

        # Check for fatal auth error
        if "FATAL_API_ERROR" in str(res.get("failure_reason")):
            print(f"\n[FATAL] Stopping evaluation due to authentication/billing error: {res.get('failure_reason')}")
            break

        # Save incrementally to guarantee persistence
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

        # Update cache
        existing_results[rid] = res

        # Polite delay between API requests
        if delay_between_requests > 0:
            time.sleep(delay_between_requests)

    # Calculate summary metrics
    total = len(all_results)
    api_success_count = sum(1 for r in all_results if r.get("api_success"))
    valid_cmd_count = sum(1 for r in all_results if r.get("is_valid"))
    rejected_cmd_count = total - valid_cmd_count
    executed_count = sum(1 for r in all_results if r.get("executed"))
    exec_success_count = sum(1 for r in all_results if r.get("execution_success"))
    semantic_pass_count = sum(1 for r in all_results if r.get("semantic_pass"))
    exact_match_count = sum(1 for r in all_results if r.get("exact_match"))
    quarantined_count = sum(1 for r in all_results if r.get("quarantined"))
    timeout_count = sum(1 for r in all_results if r.get("timed_out"))

    total_tokens = sum(
        (r.get("token_usage") or {}).get("total_tokens", 0) for r in all_results
    )
    prompt_tokens = sum(
        (r.get("token_usage") or {}).get("prompt_tokens", 0) for r in all_results
    )
    completion_tokens = sum(
        (r.get("token_usage") or {}).get("completion_tokens", 0) for r in all_results
    )

    avg_gen_latency = (
        sum(r.get("generation_latency", 0.0) for r in all_results) / total if total else 0.0
    )
    avg_exec_duration = (
        sum(r.get("execution_duration", 0.0) for r in all_results if r.get("executed"))
        / executed_count
        if executed_count
        else 0.0
    )

    empty_context_count = sum(
        1 for r in all_results if not r.get("retrieved_doc_ids")
    )

    summary = {
        "model_name": pipeline.generator.model_name,
        "provider": pipeline.generator.provider_name,
        "total_tasks_attempted": total,
        "api_success_count": api_success_count,
        "api_failure_count": total - api_success_count,
        "valid_command_count": valid_cmd_count,
        "valid_command_rate": round(valid_cmd_count / total * 100, 2) if total else 0.0,
        "rejected_command_count": rejected_cmd_count,
        "rejected_command_rate": round(rejected_cmd_count / total * 100, 2) if total else 0.0,
        "executed_count": executed_count,
        "execution_success_count": exec_success_count,
        "execution_success_rate": round(exec_success_count / executed_count * 100, 2) if executed_count else 0.0,
        "semantic_pass_count": semantic_pass_count,
        "semantic_pass_rate": round(semantic_pass_count / total * 100, 2) if total else 0.0,
        "exact_match_count": exact_match_count,
        "exact_match_rate": round(exact_match_count / total * 100, 2) if total else 0.0,
        "quarantined_count": quarantined_count,
        "timeout_count": timeout_count,
        "empty_context_count": empty_context_count,
        "empty_context_rate": round(empty_context_count / total * 100, 2) if total else 0.0,
        "avg_generation_latency": round(avg_gen_latency, 4),
        "avg_execution_duration": round(avg_exec_duration, 4),
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "results_file": str(results_path),
    }

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NL2SH Groq RAG Evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks (e.g. 10 for pilot)")
    parser.add_argument("--results-file", default="data/evaluation/groq_evaluation_results.jsonl", help="Results file path")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API requests in seconds")
    args = parser.parse_args()

    summary = run_groq_evaluation(
        results_file=args.results_file,
        limit=args.limit,
        delay_between_requests=args.delay,
    )

    print("\n" + "=" * 60)
    print("GROQ RAG EVALUATION SUMMARY")
    print(f"Model:                   {summary['model_name']} ({summary['provider']})")
    print(f"Tasks Attempted:         {summary['total_tasks_attempted']}")
    print(f"API Success:             {summary['api_success_count']}/{summary['total_tasks_attempted']}")
    print(f"Valid Command Rate:      {summary['valid_command_rate']}% ({summary['valid_command_count']}/{summary['total_tasks_attempted']})")
    print(f"Executed Tasks:          {summary['executed_count']}")
    print(f"Execution Success Rate:  {summary['execution_success_rate']}%")
    print(f"Semantic Pass Rate:      {summary['semantic_pass_rate']}% ({summary['semantic_pass_count']}/{summary['total_tasks_attempted']})")
    print(f"Exact Match Rate:        {summary['exact_match_rate']}% ({summary['exact_match_count']}/{summary['total_tasks_attempted']})")
    print(f"Timeouts:                {summary['timeout_count']}")
    print(f"Quarantined Tasks:       {summary['quarantined_count']}")
    print(f"Avg Generation Latency:  {summary['avg_generation_latency']}s")
    print(f"Total Tokens Used:       {summary['total_tokens']} (prompt: {summary['prompt_tokens']}, comp: {summary['completion_tokens']})")
    print("=" * 60)

