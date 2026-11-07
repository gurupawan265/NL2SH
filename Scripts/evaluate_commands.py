"""
NL2SH Execution Evaluator.

Executes commands inside isolated DockerSandbox containers against deterministic fixtures,
captures execution metrics, verifies semantic correctness against independent ground truth,
and outputs comprehensive evaluation logs and reports.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Sandbox.sandbox import CommandResult, DockerSandbox


def setup_fixtures_in_sandbox(sandbox: DockerSandbox, fixtures: Dict[str, Any]) -> None:
    """Inject required directory hierarchy and files into container /workspace tmpfs."""
    # 1. Create directories
    for dir_path in fixtures.get("directories", []):
        cmd = f"mkdir -p {shlex.quote(dir_path)}"
        res = sandbox.execute(cmd, timeout=5.0)
        if res.exit_code != 0:
            raise RuntimeError(f"Failed to create fixture directory '{dir_path}': {res.stderr}")

    # 2. Inject files
    for file_info in fixtures.get("files", []):
        fpath = file_info["path"]
        content = file_info.get("content", "")

        # Ensure parent directory exists
        parent_dir = str(Path(fpath).parent).replace("\\", "/")
        sandbox.execute(f"mkdir -p {shlex.quote(parent_dir)}", timeout=5.0)

        # Write file content directly as UID 1000
        proc = subprocess.run(
            [
                "docker", "exec", "-i",
                "--user", "1000:1000",
                sandbox.container_id,
                "bash", "-c", f"cat > {shlex.quote(fpath)}"
            ],
            input=content.encode("utf-8"),
            capture_output=True,
            timeout=10,
            shell=False
        )
        if proc.returncode != 0:
            err = proc.stderr.decode() if proc.stderr else "Unknown error"
            raise RuntimeError(f"Failed to inject fixture file '{fpath}': {err}")


def verify_semantic_correctness(
    sandbox: DockerSandbox,
    result: CommandResult,
    expected_result: Dict[str, Any],
    verification_method: str
) -> Tuple[bool, Optional[str]]:
    """
    Verify whether the command's execution and resulting filesystem state
    semantically fulfill the expected outcome.
    """
    expected_exit = expected_result.get("exit_code", 0)

    # Check exit code
    if result.timed_out:
        return False, "TIMEOUT: Command exceeded maximum allowed execution time"

    if result.exit_code != expected_exit:
        return False, f"NON_ZERO_EXIT: Expected exit code {expected_exit}, got {result.exit_code}. Stderr: {result.stderr.strip()}"

    # 1. Verification Method: stdout_match
    if verification_method == "stdout_match":
        if "stdout" in expected_result:
            expected_out = expected_result["stdout"]
            actual_out = result.stdout
            if actual_out != expected_out and actual_out.strip() != expected_out.strip():
                return False, f"STDOUT_MISMATCH: Expected '{repr(expected_out)}', got '{repr(actual_out)}'"

        if "stdout_lines" in expected_result:
            expected_lines = [line.strip() for line in expected_result["stdout_lines"]]
            actual_lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
            # Compare lines (supporting line-order independence for search queries)
            if sorted(actual_lines) != sorted(expected_lines):
                return False, f"STDOUT_LINES_MISMATCH: Expected lines {expected_lines}, got {actual_lines}"

        if "stdout_contains" in expected_result:
            for substr in expected_result["stdout_contains"]:
                if substr not in result.stdout:
                    return False, f"STDOUT_MISSING_SUBSTRING: Expected substring '{substr}' in stdout"

    # 2. Verification Method: file_exists
    elif verification_method == "file_exists":
        for created_file in expected_result.get("created_files", []):
            check_res = sandbox.execute(f"test -f {shlex.quote(created_file)}", timeout=5.0)
            if check_res.exit_code != 0:
                return False, f"FILE_NOT_FOUND: Expected created file '{created_file}' does not exist"

    # 3. Verification Method: file_absence
    elif verification_method == "file_absence":
        for absent_path in expected_result.get("absent_paths", []):
            check_res = sandbox.execute(f"test ! -e {shlex.quote(absent_path)}", timeout=5.0)
            if check_res.exit_code != 0:
                return False, f"FILE_NOT_REMOVED: Path '{absent_path}' still exists after deletion command"

    # 4. Verification Method: file_content_match
    elif verification_method == "file_content_match":
        # First check created files
        for created_file in expected_result.get("created_files", []):
            check_res = sandbox.execute(f"test -f {shlex.quote(created_file)}", timeout=5.0)
            if check_res.exit_code != 0:
                return False, f"FILE_NOT_FOUND: Expected file '{created_file}' does not exist"

        # Check absent paths
        for absent_path in expected_result.get("absent_paths", []):
            check_res = sandbox.execute(f"test ! -e {shlex.quote(absent_path)}", timeout=5.0)
            if check_res.exit_code != 0:
                return False, f"FILE_NOT_REMOVED: Path '{absent_path}' was not removed"

        # Check content match
        for fpath, exp_content in expected_result.get("file_contents", {}).items():
            cat_res = sandbox.execute(f"cat {shlex.quote(fpath)}", timeout=5.0)
            if cat_res.exit_code != 0:
                return False, f"CANNOT_READ_FILE: Failed reading '{fpath}'"
            if cat_res.stdout != exp_content and cat_res.stdout.strip() != exp_content.strip():
                return False, f"CONTENT_MISMATCH in '{fpath}': Expected '{repr(exp_content)}', got '{repr(cat_res.stdout)}'"

    # 5. Verification Method: directory_structure
    elif verification_method == "directory_structure":
        target_dir = expected_result.get("directory")
        if target_dir:
            check_res = sandbox.execute(f"test -d {shlex.quote(target_dir)}", timeout=5.0)
            if check_res.exit_code != 0:
                return False, f"DIRECTORY_NOT_FOUND: Directory '{target_dir}' does not exist"

        for absent_path in expected_result.get("absent_paths", []):
            check_res = sandbox.execute(f"test ! -e {shlex.quote(absent_path)}", timeout=5.0)
            if check_res.exit_code != 0:
                return False, f"DIRECTORY_NOT_REMOVED: Directory '{absent_path}' still exists"

    # 6. Verification Method: permission_match
    elif verification_method == "permission_match":
        for fpath, exp_perm in expected_result.get("permissions", {}).items():
            stat_res = sandbox.execute(f"stat -c %a {shlex.quote(fpath)}", timeout=5.0)
            if stat_res.exit_code != 0:
                return False, f"STAT_FAILED: Cannot stat '{fpath}'"
            actual_perm = stat_res.stdout.strip()
            if actual_perm != exp_perm:
                return False, f"PERMISSION_MISMATCH on '{fpath}': Expected mode {exp_perm}, got {actual_perm}"

        if "permission_check" in expected_result:
            p_check_cmd = expected_result["permission_check"]
            p_res = sandbox.execute(p_check_cmd, timeout=5.0)
            if p_res.exit_code != 0:
                return False, f"PERMISSION_CHECK_FAILED: Command '{p_check_cmd}' failed with exit code {p_res.exit_code}"

    # 7. Verification Method: exit_code
    elif verification_method == "exit_code":
        # Exit code check already verified above
        pass

    return True, None


def evaluate_single_task(task: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
    """
    Execute a single task inside a fresh, isolated DockerSandbox container.
    Injects fixtures, executes command, checks semantic results, and guarantees container destruction.
    """
    record_id = task.get("record_id", "unknown")
    instruction = task.get("instruction", "")
    command = task.get("reference_command", "")
    category = task.get("command_category", "unknown")
    fixtures = task.get("required_fixtures", {})
    expected = task.get("expected_result", {})
    method = task.get("verification_method", "exit_code")

    eval_result: Dict[str, Any] = {
        "record_id": record_id,
        "instruction": instruction,
        "command": command,
        "command_category": category,
        "verification_method": method,
        "execution_success": False,
        "semantic_pass": False,
        "exit_code": None,
        "stdout": "",
        "stderr": "",
        "duration": 0.0,
        "timed_out": False,
        "failure_reason": None
    }

    try:
        with DockerSandbox() as sandbox:
            # 1. Setup workspace fixtures
            try:
                setup_fixtures_in_sandbox(sandbox, fixtures)
            except Exception as fixture_exc:
                eval_result["failure_reason"] = f"FIXTURE_SETUP_ERROR: {fixture_exc}"
                return eval_result

            # 2. Execute command
            start_time = time.perf_counter()
            cmd_res = sandbox.execute(command, timeout=timeout)
            duration = time.perf_counter() - start_time

            eval_result["exit_code"] = cmd_res.exit_code
            eval_result["stdout"] = cmd_res.stdout
            eval_result["stderr"] = cmd_res.stderr
            eval_result["duration"] = round(duration, 4)
            eval_result["timed_out"] = cmd_res.timed_out

            expected_exit = expected.get("exit_code", 0)
            eval_result["execution_success"] = (cmd_res.exit_code == expected_exit and not cmd_res.timed_out)

            # 3. Verify semantic correctness against ground truth
            passed, failure_reason = verify_semantic_correctness(sandbox, cmd_res, expected, method)
            eval_result["semantic_pass"] = passed
            eval_result["failure_reason"] = failure_reason

    except Exception as container_exc:
        eval_result["failure_reason"] = f"CONTAINER_ERROR: {container_exc}"

    return eval_result


def run_benchmark_evaluation(
    benchmark_file: str = "data/evaluation/evaluation_benchmark.jsonl",
    results_file: str = "data/evaluation/evaluation_results.jsonl",
    timeout: float = 10.0
) -> Dict[str, Any]:
    """
    Run execution evaluation across all tasks in the benchmark.
    Ensures that failures in individual tasks do not stop execution.
    """
    bench_path = Path(benchmark_file)
    if not bench_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")

    with open(bench_path, "r", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    results: List[Dict[str, Any]] = []
    print(f"Starting execution evaluation on {len(tasks)} benchmark tasks...")

    for i, t in enumerate(tasks, start=1):
        rid = t.get("record_id", f"task_{i}")
        cmd = t.get("reference_command", "")
        print(f"[{i:3d}/{len(tasks):3d}] Evaluating {rid}: {cmd[:45]}...", end="", flush=True)

        res = evaluate_single_task(t, timeout=timeout)
        results.append(res)

        status_str = "PASS" if res["semantic_pass"] else f"FAIL ({res['failure_reason']})"
        print(f" -> {status_str} ({res['duration']}s)")

    # Save results JSONL
    res_path = Path(results_file)
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(res_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Calculate summary metrics
    total_tasks = len(results)
    exec_success_count = sum(1 for r in results if r["execution_success"])
    semantic_pass_count = sum(1 for r in results if r["semantic_pass"])
    timeout_count = sum(1 for r in results if r["timed_out"])
    avg_duration = sum(r["duration"] for r in results) / total_tasks if total_tasks else 0.0

    category_stats: Dict[str, Dict[str, int]] = {}
    for r in results:
        cat = r["command_category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "exec_success": 0, "semantic_pass": 0}
        category_stats[cat]["total"] += 1
        if r["execution_success"]:
            category_stats[cat]["exec_success"] += 1
        if r["semantic_pass"]:
            category_stats[cat]["semantic_pass"] += 1

    summary = {
        "total_tasks": total_tasks,
        "execution_success_count": exec_success_count,
        "execution_success_rate": round(exec_success_count / total_tasks * 100, 2) if total_tasks else 0.0,
        "semantic_pass_count": semantic_pass_count,
        "semantic_pass_rate": round(semantic_pass_count / total_tasks * 100, 2) if total_tasks else 0.0,
        "timeout_count": timeout_count,
        "average_duration_seconds": round(avg_duration, 4),
        "category_stats": category_stats,
        "results_file": str(res_path)
    }

    return summary


if __name__ == "__main__":
    summary = run_benchmark_evaluation()
    print("\n" + "=" * 60)
    print("BENCHMARK EXECUTION EVALUATION SUMMARY")
    print(f"Total Tasks Evaluated:       {summary['total_tasks']}")
    print(f"Execution Success Rate:      {summary['execution_success_rate']}% ({summary['execution_success_count']}/{summary['total_tasks']})")
    print(f"Semantic Pass Rate:          {summary['semantic_pass_rate']}% ({summary['semantic_pass_count']}/{summary['total_tasks']})")
    print(f"Timeouts:                    {summary['timeout_count']}")
    print(f"Average Duration:            {summary['average_duration_seconds']}s")
    print("=" * 60)

