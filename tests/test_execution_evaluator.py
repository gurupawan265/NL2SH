"""
Unit and integration tests for NL2SH execution evaluation framework.
Verifies fixture injection, isolation, path traversal protection,
semantic verification methods, error detection, timeout handling, and container cleanup.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Sandbox.sandbox import DockerSandbox
from Scripts.generate_evaluation_fixtures import validate_task, generate_benchmark_tasks
from Scripts.evaluate_commands import (
    setup_fixtures_in_sandbox,
    verify_semantic_correctness,
    evaluate_single_task,
)


def test_path_traversal_rejection():
    """Verify that path traversal attempts or paths outside /workspace are rejected."""
    # Attempt with parent traversal '..'
    task_with_traversal = {
        "instruction": "Read outside",
        "reference_command": "cat /workspace/../etc/passwd",
        "command_category": "file_search",
        "required_fixtures": {
            "directories": ["/workspace/sub/.."],
            "files": []
        },
        "expected_result": {"exit_code": 0},
        "verification_method": "stdout_match",
        "safety_level": "safe"
    }
    is_valid, reason = validate_task(task_with_traversal)
    assert not is_valid
    assert "traversal" in reason.lower()

    # Attempt with absolute path outside /workspace
    task_outside_workspace = {
        "instruction": "Create file in root",
        "reference_command": "touch /tmp/test.txt",
        "command_category": "file_creation",
        "required_fixtures": {
            "directories": [],
            "files": [{"path": "/tmp/test.txt", "content": "data"}]
        },
        "expected_result": {"exit_code": 0},
        "verification_method": "file_exists",
        "safety_level": "safe"
    }
    is_valid_out, reason_out = validate_task(task_outside_workspace)
    assert not is_valid_out
    assert "/workspace" in reason_out


def test_quarantined_tasks_detection():
    """Verify that benchmark generation detects and quarantines risky candidate tasks."""
    benchmark_tasks, quarantined = generate_benchmark_tasks()
    assert len(benchmark_tasks) == 100
    assert len(quarantined) >= 3

    reasons = [q.get("quarantine_reason", "") for q in quarantined]
    assert any("traversal" in r.lower() for r in reasons)
    assert any("destructive" in r.lower() or "root" in r.lower() for r in reasons)
    assert any("network" in r.lower() for r in reasons)


def test_fixture_creation_and_isolation():
    """
    Verify fixtures are successfully created in one container
    and are completely absent in a subsequent container (isolation guarantee).
    """
    fixtures = {
        "directories": ["/workspace/custom_dir"],
        "files": [{"path": "/workspace/custom_dir/sample.txt", "content": "unique_fixture_data\n"}]
    }

    # Container 1: setup fixtures and verify
    with DockerSandbox() as sb1:
        setup_fixtures_in_sandbox(sb1, fixtures)
        res = sb1.execute("cat /workspace/custom_dir/sample.txt")
        assert res.exit_code == 0
        assert res.stdout == "unique_fixture_data\n"

    # Container 2: fresh container must NOT contain Container 1's fixtures
    with DockerSandbox() as sb2:
        res = sb2.execute("test -e /workspace/custom_dir")
        assert res.exit_code != 0


def test_evaluate_single_task_stdout_match():
    """Test evaluating a command with stdout_match verification."""
    task = {
        "record_id": "test_stdout_001",
        "instruction": "Print hello world",
        "reference_command": "echo 'Hello World'",
        "command_category": "file_creation",
        "required_fixtures": {"directories": [], "files": []},
        "expected_result": {
            "exit_code": 0,
            "stdout": "Hello World\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    }
    result = evaluate_single_task(task, timeout=5.0)
    assert result["execution_success"] is True
    assert result["semantic_pass"] is True
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "Hello World"
    assert result["failure_reason"] is None


def test_evaluate_single_task_file_exists():
    """Test evaluating a command with file_exists verification."""
    task = {
        "record_id": "test_create_001",
        "instruction": "Create an empty file /workspace/created.txt",
        "reference_command": "touch /workspace/created.txt",
        "command_category": "file_creation",
        "required_fixtures": {"directories": [], "files": []},
        "expected_result": {
            "exit_code": 0,
            "created_files": ["/workspace/created.txt"]
        },
        "verification_method": "file_exists",
        "safety_level": "file_create"
    }
    result = evaluate_single_task(task, timeout=5.0)
    assert result["execution_success"] is True
    assert result["semantic_pass"] is True
    assert result["failure_reason"] is None


def test_evaluate_single_task_file_content_match():
    """Test evaluating a command with file_content_match verification."""
    task = {
        "record_id": "test_content_001",
        "instruction": "Save 123 into /workspace/num.txt",
        "reference_command": "echo '123' > /workspace/num.txt",
        "command_category": "file_creation",
        "required_fixtures": {"directories": [], "files": []},
        "expected_result": {
            "exit_code": 0,
            "created_files": ["/workspace/num.txt"],
            "file_contents": {"/workspace/num.txt": "123\n"}
        },
        "verification_method": "file_content_match",
        "safety_level": "file_create"
    }
    result = evaluate_single_task(task, timeout=5.0)
    assert result["execution_success"] is True
    assert result["semantic_pass"] is True


def test_evaluate_single_task_file_deletion():
    """Test evaluating a command with file_absence verification."""
    task = {
        "record_id": "test_delete_001",
        "instruction": "Remove /workspace/temp.txt",
        "reference_command": "rm /workspace/temp.txt",
        "command_category": "file_copy_move",
        "required_fixtures": {
            "directories": [],
            "files": [{"path": "/workspace/temp.txt", "content": "discard"}]
        },
        "expected_result": {
            "exit_code": 0,
            "absent_paths": ["/workspace/temp.txt"]
        },
        "verification_method": "file_absence",
        "safety_level": "file_delete"
    }
    result = evaluate_single_task(task, timeout=5.0)
    assert result["execution_success"] is True
    assert result["semantic_pass"] is True


def test_evaluate_single_task_directory_structure():
    """Test directory structure creation and verification."""
    task = {
        "record_id": "test_dir_001",
        "instruction": "Create /workspace/nested/dir",
        "reference_command": "mkdir -p /workspace/nested/dir",
        "command_category": "directory_operations",
        "required_fixtures": {"directories": [], "files": []},
        "expected_result": {
            "exit_code": 0,
            "directory": "/workspace/nested/dir"
        },
        "verification_method": "directory_structure",
        "safety_level": "file_create"
    }
    result = evaluate_single_task(task, timeout=5.0)
    assert result["execution_success"] is True
    assert result["semantic_pass"] is True


def test_evaluate_single_task_permission_match():
    """Test file permission modification and verification using stat -c %a."""
    task = {
        "record_id": "test_perm_001",
        "instruction": "Set permissions of /workspace/config.cfg to 600",
        "reference_command": "chmod 600 /workspace/config.cfg",
        "command_category": "permissions",
        "required_fixtures": {
            "directories": [],
            "files": [{"path": "/workspace/config.cfg", "content": "key=val\n"}]
        },
        "expected_result": {
            "exit_code": 0,
            "permissions": {"/workspace/config.cfg": "600"}
        },
        "verification_method": "permission_match",
        "safety_level": "file_modify"
    }
    result = evaluate_single_task(task, timeout=5.0)
    assert result["execution_success"] is True
    assert result["semantic_pass"] is True


def test_incorrect_expected_result_detection():
    """Verify that semantic verification flags when a command fails to produce the expected output."""
    task = {
        "record_id": "test_fail_001",
        "instruction": "Print goodbye",
        "reference_command": "echo 'Hello'",  # Intentionally wrong command
        "command_category": "text_search",
        "required_fixtures": {"directories": [], "files": []},
        "expected_result": {
            "exit_code": 0,
            "stdout": "Goodbye\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    }
    result = evaluate_single_task(task, timeout=5.0)
    assert result["execution_success"] is True  # Command itself ran fine
    assert result["semantic_pass"] is False     # Semantic check detected mismatch
    assert "STDOUT_MISMATCH" in (result["failure_reason"] or "")


def test_timeout_handling():
    """Verify timeout handling marks the task as timed out and semantically failed."""
    task = {
        "record_id": "test_timeout_001",
        "instruction": "Sleep for 10 seconds",
        "reference_command": "sleep 10",
        "command_category": "file_search",
        "required_fixtures": {"directories": [], "files": []},
        "expected_result": {"exit_code": 0},
        "verification_method": "stdout_match",
        "safety_level": "safe"
    }
    result = evaluate_single_task(task, timeout=1.0)
    assert result["execution_success"] is False
    assert result["timed_out"] is True
    assert result["semantic_pass"] is False
    assert "TIMEOUT" in (result["failure_reason"] or "")


def test_expected_nonzero_exit_code():
    """Verify evaluator accepts expected non-zero exit codes (e.g. diff returning 1)."""
    task = {
        "record_id": "test_diff_001",
        "instruction": "Compare differing files",
        "reference_command": "diff /workspace/a.txt /workspace/b.txt",
        "command_category": "file_comparison",
        "required_fixtures": {
            "directories": [],
            "files": [
                {"path": "/workspace/a.txt", "content": "alpha\n"},
                {"path": "/workspace/b.txt", "content": "beta\n"}
            ]
        },
        "expected_result": {
            "exit_code": 1,
            "stdout_contains": ["< alpha", "> beta"]
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    }
    result = evaluate_single_task(task, timeout=5.0)
    assert result["execution_success"] is True
    assert result["exit_code"] == 1
    assert result["semantic_pass"] is True


def test_container_cleanup_after_task_failure():
    """Verify container is completely cleaned up and destroyed even if the task fails."""
    # List running containers before
    before = subprocess.run(
        ["docker", "ps", "-q"],
        capture_output=True, text=True, check=True
    ).stdout.split()

    failing_task = {
        "record_id": "test_cleanup_fail",
        "instruction": "Run invalid command",
        "reference_command": "nonexistent_command_12345",
        "command_category": "file_search",
        "required_fixtures": {"directories": [], "files": []},
        "expected_result": {"exit_code": 0},
        "verification_method": "stdout_match",
        "safety_level": "safe"
    }

    result = evaluate_single_task(failing_task, timeout=5.0)
    assert result["execution_success"] is False

    # List running containers after
    after = subprocess.run(
        ["docker", "ps", "-q"],
        capture_output=True, text=True, check=True
    ).stdout.split()

    # The failing task's container must NOT be in running containers
    assert len(after) <= len(before)

