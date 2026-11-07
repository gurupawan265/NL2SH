"""
NL2SH Execution Evaluation Benchmark and Fixture Generator.

Selects, validates, and builds deterministic execution-evaluation benchmark tasks
across 10 core Linux categories with path-traversal protected fixtures and
independently specified ground-truth verification assertions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
from typing import Any, Dict, List, Optional, Tuple


def normalize_workspace_path(path_str: str) -> str:
    """
    Validate and normalize path ensuring it resolves strictly within /workspace.
    Prevents path traversal attacks (e.g., ../ escaping /workspace).
    """
    clean = path_str.strip().replace("\\", "/")
    if clean.startswith("/workspace"):
        rel = clean[len("/workspace"):].lstrip("/")
    elif clean.startswith("/"):
        raise ValueError(f"Path traversal rejected: absolute path outside /workspace: '{path_str}'")
    else:
        rel = clean

    parts = [p for p in rel.split("/") if p and p != "."]
    resolved_parts: List[str] = []
    for p in parts:
        if p == "..":
            if not resolved_parts:
                raise ValueError(f"Path traversal rejected: '{path_str}' escapes /workspace")
            resolved_parts.pop()
        else:
            resolved_parts.append(p)

    return "/workspace/" + "/".join(resolved_parts) if resolved_parts else "/workspace"


def validate_fixtures(fixtures: Dict[str, Any]) -> Dict[str, Any]:
    """Validate all fixture paths to guarantee workspace containment."""
    validated: Dict[str, Any] = {
        "directories": [],
        "files": []
    }

    for d in fixtures.get("directories", []):
        norm_dir = normalize_workspace_path(d)
        if norm_dir not in validated["directories"]:
            validated["directories"].append(norm_dir)

    for f in fixtures.get("files", []):
        norm_path = normalize_workspace_path(f["path"])
        validated["files"].append({
            "path": norm_path,
            "content": f.get("content", "")
        })

    return validated


def validate_task(task: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate task structure, command, and ensure all fixture paths avoid traversal."""
    required_keys = ["instruction", "reference_command", "command_category", "required_fixtures", "expected_result", "verification_method"]
    for k in required_keys:
        if k not in task:
            return False, f"Missing required key: {k}"

    fixtures = task.get("required_fixtures", {})
    try:
        validate_fixtures(fixtures)
    except ValueError as e:
        return False, f"Path traversal rejected: {e}"

    # Check for forbidden patterns in command
    cmd = task["reference_command"]
    if "/../" in cmd or cmd.startswith("../") or " ../" in cmd:
        return False, "Path traversal pattern '../' in reference command"

    if cmd.strip() in ["rm -rf /", "rm -rf /*"]:
        return False, "Destructive root filesystem command"

    return True, None


def generate_benchmark_tasks() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return benchmark tasks and quarantined tasks."""
    tasks = get_curated_benchmark_tasks()
    quarantined = [
        {
            "candidate_id": "quar_eval_001",
            "instruction": "Fetch http://example.com/data.tar.gz",
            "command": "curl -O http://example.com/data.tar.gz",
            "quarantine_reason": "network_access_forbidden_in_sandbox"
        },
        {
            "candidate_id": "quar_eval_002",
            "instruction": "Restart nginx web service",
            "command": "sudo systemctl restart nginx",
            "quarantine_reason": "privileged_command_requires_root_and_systemd"
        },
        {
            "candidate_id": "quar_eval_003",
            "instruction": "Remove temporary files from parent system directory",
            "command": "rm -rf ../temp_data",
            "quarantine_reason": "path_traversal_escaping_workspace"
        },
        {
            "candidate_id": "quar_eval_004",
            "instruction": "Wipe all root directory files forcefully",
            "command": "rm -rf /",
            "quarantine_reason": "destructive_system_command_targeting_root"
        }
    ]
    return tasks, quarantined


def get_curated_benchmark_tasks() -> List[Dict[str, Any]]:
    """
    Define 100 diverse, high-quality evaluation tasks across 10 core command categories.
    Each task specifies its natural-language instruction, reference Bash command,
    isolated workspace fixtures, and independent ground-truth verification rules.
    """
    tasks: List[Dict[str, Any]] = []

    # =========================================================================
    # CATEGORY 1: file_search (10 tasks)
    # =========================================================================
    tasks.append({
        "instruction": "Find all '.jpg' files inside the directory /workspace/photos",
        "reference_command": "find /workspace/photos -name '*.jpg' | sort",
        "command_category": "file_search",
        "required_fixtures": {
            "directories": ["/workspace/photos", "/workspace/photos/sub"],
            "files": [
                {"path": "/workspace/photos/img1.jpg", "content": "jpeg-data-1"},
                {"path": "/workspace/photos/notes.txt", "content": "text-file"},
                {"path": "/workspace/photos/sub/img2.jpg", "content": "jpeg-data-2"},
                {"path": "/workspace/photos/sub/logo.png", "content": "png-data"}
            ]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout_lines": ["/workspace/photos/img1.jpg", "/workspace/photos/sub/img2.jpg"]
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Find all empty files in /workspace/project",
        "reference_command": "find /workspace/project -type f -empty | sort",
        "command_category": "file_search",
        "required_fixtures": {
            "directories": ["/workspace/project"],
            "files": [
                {"path": "/workspace/project/empty1.txt", "content": ""},
                {"path": "/workspace/project/nonempty.txt", "content": "content"},
                {"path": "/workspace/project/empty2.log", "content": ""}
            ]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout_lines": ["/workspace/project/empty1.txt", "/workspace/project/empty2.log"]
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "List all subdirectories directly inside /workspace/data",
        "reference_command": "find /workspace/data -mindepth 1 -maxdepth 1 -type d | sort",
        "command_category": "file_search",
        "required_fixtures": {
            "directories": ["/workspace/data/raw", "/workspace/data/processed", "/workspace/data/reports"],
            "files": [{"path": "/workspace/data/readme.txt", "content": "info"}]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout_lines": ["/workspace/data/processed", "/workspace/data/raw", "/workspace/data/reports"]
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Find files named 'config.json' in /workspace",
        "reference_command": "find /workspace -type f -name 'config.json' | sort",
        "command_category": "file_search",
        "required_fixtures": {
            "directories": ["/workspace/app", "/workspace/app/config", "/workspace/other"],
            "files": [
                {"path": "/workspace/app/config.json", "content": '{"app": true}'},
                {"path": "/workspace/app/config/config.json", "content": '{"sub": true}'},
                {"path": "/workspace/other/data.json", "content": "{}"}
            ]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout_lines": ["/workspace/app/config.json", "/workspace/app/config/config.json"]
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Find all '.sh' files in /workspace/scripts and display their paths",
        "reference_command": "find /workspace/scripts -name '*.sh' | sort",
        "command_category": "file_search",
        "required_fixtures": {
            "directories": ["/workspace/scripts"],
            "files": [
                {"path": "/workspace/scripts/deploy.sh", "content": "#!/bin/bash\n"},
                {"path": "/workspace/scripts/test.sh", "content": "#!/bin/bash\n"},
                {"path": "/workspace/scripts/setup.py", "content": "# python\n"}
            ]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout_lines": ["/workspace/scripts/deploy.sh", "/workspace/scripts/test.sh"]
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Find all regular files in /workspace/archive",
        "reference_command": "find /workspace/archive -type f | sort",
        "command_category": "file_search",
        "required_fixtures": {
            "directories": ["/workspace/archive", "/workspace/archive/old"],
            "files": [
                {"path": "/workspace/archive/doc1.txt", "content": "1"},
                {"path": "/workspace/archive/old/doc2.txt", "content": "2"}
            ]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout_lines": ["/workspace/archive/doc1.txt", "/workspace/archive/old/doc2.txt"]
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Find all files with '.log' extension case-insensitively in /workspace",
        "reference_command": "find /workspace -type f -iname '*.log' | sort",
        "command_category": "file_search",
        "required_fixtures": {
            "directories": ["/workspace/logs"],
            "files": [
                {"path": "/workspace/logs/app.LOG", "content": "log data"},
                {"path": "/workspace/logs/system.log", "content": "system data"},
                {"path": "/workspace/logs/app.txt", "content": "text"}
            ]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout_lines": ["/workspace/logs/app.LOG", "/workspace/logs/system.log"]
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Find files larger than 100 bytes in /workspace/storage",
        "reference_command": "find /workspace/storage -type f -size +100c | sort",
        "command_category": "file_search",
        "required_fixtures": {
            "directories": ["/workspace/storage"],
            "files": [
                {"path": "/workspace/storage/large.dat", "content": "x" * 200},
                {"path": "/workspace/storage/small.dat", "content": "x" * 50}
            ]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout_lines": ["/workspace/storage/large.dat"]
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Find all files in /workspace/code at a depth of exactly 1",
        "reference_command": "find /workspace/code -maxdepth 1 -mindepth 1 -type f | sort",
        "command_category": "file_search",
        "required_fixtures": {
            "directories": ["/workspace/code", "/workspace/code/sub"],
            "files": [
                {"path": "/workspace/code/main.py", "content": "pass"},
                {"path": "/workspace/code/sub/util.py", "content": "pass"}
            ]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout_lines": ["/workspace/code/main.py"]
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Find files not ending with '.txt' in /workspace/docs",
        "reference_command": "find /workspace/docs -type f ! -name '*.txt' | sort",
        "command_category": "file_search",
        "required_fixtures": {
            "directories": ["/workspace/docs"],
            "files": [
                {"path": "/workspace/docs/guide.txt", "content": "guide"},
                {"path": "/workspace/docs/manual.md", "content": "manual"},
                {"path": "/workspace/docs/sheet.csv", "content": "a,b"}
            ]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout_lines": ["/workspace/docs/manual.md", "/workspace/docs/sheet.csv"]
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    # =========================================================================
    # CATEGORY 2: text_search (10 tasks)
    # =========================================================================
    tasks.append({
        "instruction": "Search for lines containing 'ERROR' in /workspace/server.log",
        "reference_command": "grep 'ERROR' /workspace/server.log",
        "command_category": "text_search",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/server.log",
                "content": "INFO: connected\nERROR: database timeout\nWARN: high memory\nERROR: disk full\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "ERROR: database timeout\nERROR: disk full\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Search case-insensitively for 'failed' in /workspace/auth.log",
        "reference_command": "grep -i 'failed' /workspace/auth.log",
        "command_category": "text_search",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/auth.log",
                "content": "Login SUCCESS user1\nlogin FAILED user2\nFailed password for root\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "login FAILED user2\nFailed password for root\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Print lines with line numbers matching 'critical' in /workspace/app.log",
        "reference_command": "grep -n 'critical' /workspace/app.log",
        "command_category": "text_search",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/app.log",
                "content": "normal operation\ncritical system fault\nall clear\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "2:critical system fault\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Invert match and show all lines in /workspace/items.txt that do not contain 'apple'",
        "reference_command": "grep -v 'apple' /workspace/items.txt",
        "command_category": "text_search",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/items.txt",
                "content": "apple pie\nbanana bread\ncherry tart\ngreen apple\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "banana bread\ncherry tart\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Count the number of lines containing 'error' in /workspace/debug.log",
        "reference_command": "grep -c 'error' /workspace/debug.log",
        "command_category": "text_search",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/debug.log",
                "content": "error 1\ninfo\nerror 2\nwarning\nerror 3\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "3\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Search for lines starting with 'root:' in /workspace/users.txt",
        "reference_command": "grep '^root:' /workspace/users.txt",
        "command_category": "text_search",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/users.txt",
                "content": "root:x:0:0\nuser1:x:1000\nchroot:x:1001\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "root:x:0:0\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Search for lines ending with 'done' in /workspace/status.txt",
        "reference_command": "grep 'done$' /workspace/status.txt",
        "command_category": "text_search",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/status.txt",
                "content": "task1 done\ntask2 pending\ntask3 done\ndone already\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "task1 done\ntask3 done\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Recursively search for 'database_url' in /workspace/config",
        "reference_command": "grep -r 'database_url' /workspace/config | sort",
        "command_category": "text_search",
        "required_fixtures": {
            "directories": ["/workspace/config", "/workspace/config/env"],
            "files": [
                {"path": "/workspace/config/app.env", "content": "database_url=postgres://localhost\n"},
                {"path": "/workspace/config/env/prod.env", "content": "database_url=postgres://remote\n"},
                {"path": "/workspace/config/app.json", "content": '{"port": 8080}'}
            ]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout_lines": [
                "/workspace/config/app.env:database_url=postgres://localhost",
                "/workspace/config/env/prod.env:database_url=postgres://remote"
            ]
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Search for the whole word 'cat' in /workspace/animals.txt",
        "reference_command": "grep -w 'cat' /workspace/animals.txt",
        "command_category": "text_search",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/animals.txt",
                "content": "black cat\nscattered items\nwildcat\ncat food\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "black cat\ncat food\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Save grep matches of 'SUCCESS' from /workspace/results.txt to /workspace/passed.txt",
        "reference_command": "grep 'SUCCESS' /workspace/results.txt > /workspace/passed.txt",
        "command_category": "text_search",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/results.txt",
                "content": "test1: SUCCESS\ntest2: FAILED\ntest3: SUCCESS\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "created_files": ["/workspace/passed.txt"],
            "file_contents": {
                "/workspace/passed.txt": "test1: SUCCESS\ntest3: SUCCESS\n"
            }
        },
        "verification_method": "file_content_match",
        "safety_level": "file_modify"
    })

    # =========================================================================
    # CATEGORY 3: text_processing (10 tasks)
    # =========================================================================
    tasks.append({
        "instruction": "Print the second field of colon-separated /workspace/users.csv using awk",
        "reference_command": "awk -F':' '{print $2}' /workspace/users.csv",
        "command_category": "text_processing",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/users.csv",
                "content": "alice:admin:active\nbob:editor:active\ncharlie:viewer:inactive\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "admin\neditor\nviewer\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Extract the first field using comma as delimiter from /workspace/data.csv with cut",
        "reference_command": "cut -d',' -f1 /workspace/data.csv",
        "command_category": "text_processing",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/data.csv",
                "content": "101,John,Doe\n102,Jane,Smith\n103,Alice,Johnson\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "101\n102\n103\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Replace all occurrences of 'foo' with 'bar' in /workspace/sample.txt using sed",
        "reference_command": "sed 's/foo/bar/g' /workspace/sample.txt",
        "command_category": "text_processing",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/sample.txt",
                "content": "foo and foo are food\nsimply foo\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "bar and bar are bard\nsimply bar\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Translate lowercase letters to uppercase in /workspace/lowercase.txt using tr",
        "reference_command": "tr '[:lower:]' '[:upper:]' < /workspace/lowercase.txt",
        "command_category": "text_processing",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/lowercase.txt",
                "content": "hello world from nl2sh\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "HELLO WORLD FROM NL2SH\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Print lines 2 through 4 of /workspace/lines.txt using sed",
        "reference_command": "sed -n '2,4p' /workspace/lines.txt",
        "command_category": "text_processing",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/lines.txt",
                "content": "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "Line 2\nLine 3\nLine 4\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Extract the third column of space-separated /workspace/table.txt with awk",
        "reference_command": "awk '{print $3}' /workspace/table.txt",
        "command_category": "text_processing",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/table.txt",
                "content": "A B 100\nC D 200\nE F 300\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "100\n200\n300\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Replace 'temp' with 'final' in-place in /workspace/document.txt using sed",
        "reference_command": "sed -i 's/temp/final/g' /workspace/document.txt",
        "command_category": "text_processing",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/document.txt",
                "content": "This is a temp draft with temp notes.\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "file_contents": {
                "/workspace/document.txt": "This is a final draft with final notes.\n"
            }
        },
        "verification_method": "file_content_match",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Print lines with more than 15 characters from /workspace/words.txt using awk",
        "reference_command": "awk 'length > 15' /workspace/words.txt",
        "command_category": "text_processing",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/words.txt",
                "content": "short line\nthis is a significantly longer line\nmedium text\nanother very extended line\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "this is a significantly longer line\nanother very extended line\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Delete characters 1 through 5 from each line in /workspace/prefixes.txt using cut",
        "reference_command": "cut -c 6- /workspace/prefixes.txt",
        "command_category": "text_processing",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/prefixes.txt",
                "content": "12345apple\n12345banana\n12345cherry\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "apple\nbanana\ncherry\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Delete blank lines from /workspace/spaced.txt using sed",
        "reference_command": "sed '/^$/d' /workspace/spaced.txt",
        "command_category": "text_processing",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/spaced.txt",
                "content": "Line 1\n\nLine 2\n\nLine 3\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "Line 1\nLine 2\nLine 3\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    # =========================================================================
    # CATEGORY 4: file_creation (10 tasks)
    # =========================================================================
    tasks.append({
        "instruction": "Create an empty file named /workspace/report.txt",
        "reference_command": "touch /workspace/report.txt",
        "command_category": "file_creation",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": []
        },
        "expected_result": {
            "exit_code": 0,
            "created_files": ["/workspace/report.txt"],
            "file_contents": {"/workspace/report.txt": ""}
        },
        "verification_method": "file_exists",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Create two empty files /workspace/file1.txt and /workspace/file2.txt",
        "reference_command": "touch /workspace/file1.txt /workspace/file2.txt",
        "command_category": "file_creation",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": []
        },
        "expected_result": {
            "exit_code": 0,
            "created_files": ["/workspace/file1.txt", "/workspace/file2.txt"]
        },
        "verification_method": "file_exists",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Create an empty file /workspace/app/config.ini inside existing folder",
        "reference_command": "touch /workspace/app/config.ini",
        "command_category": "file_creation",
        "required_fixtures": {
            "directories": ["/workspace/app"],
            "files": []
        },
        "expected_result": {
            "exit_code": 0,
            "created_files": ["/workspace/app/config.ini"]
        },
        "verification_method": "file_exists",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Create a file /workspace/hello.txt containing 'Hello World'",
        "reference_command": "echo 'Hello World' > /workspace/hello.txt",
        "command_category": "file_creation",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": []
        },
        "expected_result": {
            "exit_code": 0,
            "created_files": ["/workspace/hello.txt"],
            "file_contents": {"/workspace/hello.txt": "Hello World\n"}
        },
        "verification_method": "file_content_match",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Append 'Status: OK' to /workspace/status.log",
        "reference_command": "echo 'Status: OK' >> /workspace/status.log",
        "command_category": "file_creation",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{"path": "/workspace/status.log", "content": "Initial Line\n"}]
        },
        "expected_result": {
            "exit_code": 0,
            "file_contents": {"/workspace/status.log": "Initial Line\nStatus: OK\n"}
        },
        "verification_method": "file_content_match",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Create an empty placeholder file /workspace/.keep",
        "reference_command": "touch /workspace/.keep",
        "command_category": "file_creation",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": []
        },
        "expected_result": {
            "exit_code": 0,
            "created_files": ["/workspace/.keep"]
        },
        "verification_method": "file_exists",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Create file /workspace/nums.txt with numbers 1, 2, 3 on separate lines",
        "reference_command": "printf '1\\n2\\n3\\n' > /workspace/nums.txt",
        "command_category": "file_creation",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": []
        },
        "expected_result": {
            "exit_code": 0,
            "created_files": ["/workspace/nums.txt"],
            "file_contents": {"/workspace/nums.txt": "1\n2\n3\n"}
        },
        "verification_method": "file_content_match",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Create multiple empty log files log1.log log2.log in /workspace/logs",
        "reference_command": "touch /workspace/logs/log1.log /workspace/logs/log2.log",
        "command_category": "file_creation",
        "required_fixtures": {
            "directories": ["/workspace/logs"],
            "files": []
        },
        "expected_result": {
            "exit_code": 0,
            "created_files": ["/workspace/logs/log1.log", "/workspace/logs/log2.log"]
        },
        "verification_method": "file_exists",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Create an empty script /workspace/run.sh",
        "reference_command": "touch /workspace/run.sh",
        "command_category": "file_creation",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": []
        },
        "expected_result": {
            "exit_code": 0,
            "created_files": ["/workspace/run.sh"]
        },
        "verification_method": "file_exists",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Truncate or create an empty file /workspace/reset.txt using redirection",
        "reference_command": "> /workspace/reset.txt",
        "command_category": "file_creation",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{"path": "/workspace/reset.txt", "content": "some previous text"}]
        },
        "expected_result": {
            "exit_code": 0,
            "file_contents": {"/workspace/reset.txt": ""}
        },
        "verification_method": "file_content_match",
        "safety_level": "file_modify"
    })

    # =========================================================================
    # CATEGORY 5: directory_operations (10 tasks)
    # =========================================================================
    tasks.append({
        "instruction": "Create a new directory named /workspace/project",
        "reference_command": "mkdir /workspace/project",
        "command_category": "directory_operations",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": []
        },
        "expected_result": {
            "exit_code": 0,
            "directory": "/workspace/project"
        },
        "verification_method": "directory_structure",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Recursively create directory path /workspace/a/b/c",
        "reference_command": "mkdir -p /workspace/a/b/c",
        "command_category": "directory_operations",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": []
        },
        "expected_result": {
            "exit_code": 0,
            "directory": "/workspace/a/b/c"
        },
        "verification_method": "directory_structure",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Create two directories /workspace/src and /workspace/build",
        "reference_command": "mkdir /workspace/src /workspace/build",
        "command_category": "directory_operations",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": []
        },
        "expected_result": {
            "exit_code": 0,
            "directory": "/workspace/src"
        },
        "verification_method": "directory_structure",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Remove the empty directory /workspace/empty_dir",
        "reference_command": "rmdir /workspace/empty_dir",
        "command_category": "directory_operations",
        "required_fixtures": {
            "directories": ["/workspace/empty_dir"],
            "files": []
        },
        "expected_result": {
            "exit_code": 0,
            "absent_paths": ["/workspace/empty_dir"]
        },
        "verification_method": "file_absence",
        "safety_level": "file_delete"
    })

    tasks.append({
        "instruction": "Remove nested empty directories parent/child using rmdir -p",
        "reference_command": "rmdir -p parent/child",
        "command_category": "directory_operations",
        "required_fixtures": {
            "directories": ["/workspace/parent/child"],
            "files": []
        },
        "expected_result": {
            "exit_code": 0,
            "absent_paths": ["/workspace/parent/child", "/workspace/parent"]
        },
        "verification_method": "file_absence",
        "safety_level": "file_delete"
    })

    tasks.append({
        "instruction": "Create directory /workspace/backup with parent dirs if needed",
        "reference_command": "mkdir -p /workspace/backup/daily",
        "command_category": "directory_operations",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": []
        },
        "expected_result": {
            "exit_code": 0,
            "directory": "/workspace/backup/daily"
        },
        "verification_method": "directory_structure",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Create a hidden directory /workspace/.config",
        "reference_command": "mkdir /workspace/.config",
        "command_category": "directory_operations",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": []
        },
        "expected_result": {
            "exit_code": 0,
            "directory": "/workspace/.config"
        },
        "verification_method": "directory_structure",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Create directories docs and tests under /workspace/pkg",
        "reference_command": "mkdir -p /workspace/pkg/docs /workspace/pkg/tests",
        "command_category": "directory_operations",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": []
        },
        "expected_result": {
            "exit_code": 0,
            "directory": "/workspace/pkg/docs"
        },
        "verification_method": "directory_structure",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Create /workspace/releases directory with mode 700",
        "reference_command": "mkdir -m 700 /workspace/releases",
        "command_category": "directory_operations",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": []
        },
        "expected_result": {
            "exit_code": 0,
            "directory": "/workspace/releases",
            "permissions": {"/workspace/releases": "700"}
        },
        "verification_method": "permission_match",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Safely remove directory /workspace/tmp_test if empty",
        "reference_command": "rmdir /workspace/tmp_test",
        "command_category": "directory_operations",
        "required_fixtures": {
            "directories": ["/workspace/tmp_test"],
            "files": []
        },
        "expected_result": {
            "exit_code": 0,
            "absent_paths": ["/workspace/tmp_test"]
        },
        "verification_method": "file_absence",
        "safety_level": "file_delete"
    })

    # =========================================================================
    # CATEGORY 6: sorting (10 tasks)
    # =========================================================================
    tasks.append({
        "instruction": "Sort the lines in /workspace/fruits.txt alphabetically",
        "reference_command": "sort /workspace/fruits.txt",
        "command_category": "sorting",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/fruits.txt",
                "content": "orange\napple\nbanana\npear\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "apple\nbanana\norange\npear\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Sort lines in /workspace/fruits.txt in reverse order",
        "reference_command": "sort -r /workspace/fruits.txt",
        "command_category": "sorting",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/fruits.txt",
                "content": "orange\napple\nbanana\npear\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "pear\norange\nbanana\napple\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Sort numbers in /workspace/numbers.txt numerically",
        "reference_command": "sort -n /workspace/numbers.txt",
        "command_category": "sorting",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/numbers.txt",
                "content": "100\n2\n25\n3\n1\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "1\n2\n3\n25\n100\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Deduplicate and sort lines in /workspace/duplicates.txt",
        "reference_command": "sort -u /workspace/duplicates.txt",
        "command_category": "sorting",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/duplicates.txt",
                "content": "b\na\nb\nc\na\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "a\nb\nc\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Show only duplicate lines from sorted /workspace/sorted_dups.txt using uniq -d",
        "reference_command": "uniq -d /workspace/sorted_dups.txt",
        "command_category": "sorting",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/sorted_dups.txt",
                "content": "apple\napple\nbanana\ncherry\ncherry\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "apple\ncherry\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Count duplicate occurrences in sorted /workspace/items.txt with uniq -c",
        "reference_command": "uniq -c /workspace/items.txt",
        "command_category": "sorting",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/items.txt",
                "content": "cat\ncat\ndog\nfish\nfish\nfish\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout_lines": ["2 cat", "1 dog", "3 fish"]
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Sort colon-separated file /workspace/passwd.txt by second field numerically",
        "reference_command": "sort -t':' -k2,2n /workspace/passwd.txt",
        "command_category": "sorting",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/passwd.txt",
                "content": "userA:100\nuserB:50\nuserC:200\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "userB:50\nuserA:100\nuserC:200\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Save sorted output of /workspace/words.txt into /workspace/sorted_words.txt",
        "reference_command": "sort /workspace/words.txt > /workspace/sorted_words.txt",
        "command_category": "sorting",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/words.txt",
                "content": "zebra\napple\nmango\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "file_contents": {"/workspace/sorted_words.txt": "apple\nmango\nzebra\n"}
        },
        "verification_method": "file_content_match",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Sort numbers in /workspace/scores.txt descending numerically",
        "reference_command": "sort -nr /workspace/scores.txt",
        "command_category": "sorting",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/scores.txt",
                "content": "45\n100\n12\n88\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "100\n88\n45\n12\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Pipe /workspace/unsorted.txt into sort and uniq",
        "reference_command": "sort /workspace/unsorted.txt | uniq",
        "command_category": "sorting",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/unsorted.txt",
                "content": "red\nblue\nred\ngreen\nblue\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "blue\ngreen\nred\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    # =========================================================================
    # CATEGORY 7: counting (10 tasks)
    # =========================================================================
    tasks.append({
        "instruction": "Count the number of lines in /workspace/file.txt",
        "reference_command": "wc -l < /workspace/file.txt",
        "command_category": "counting",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/file.txt",
                "content": "line1\nline2\nline3\nline4\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "4\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Count the number of words in /workspace/essay.txt",
        "reference_command": "wc -w < /workspace/essay.txt",
        "command_category": "counting",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/essay.txt",
                "content": "The quick brown fox jumps over the lazy dog\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "9\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Count total bytes in /workspace/data.bin",
        "reference_command": "wc -c < /workspace/data.bin",
        "command_category": "counting",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/data.bin",
                "content": "1234567890"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "10\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Count characters in /workspace/msg.txt with wc -m",
        "reference_command": "wc -m < /workspace/msg.txt",
        "command_category": "counting",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/msg.txt",
                "content": "Hello!\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "7\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Count lines of file /workspace/code.py and print filename",
        "reference_command": "wc -l /workspace/code.py",
        "command_category": "counting",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/code.py",
                "content": "import sys\nprint('hi')\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "2 /workspace/code.py\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Count number of regular files in /workspace/pool using find and wc",
        "reference_command": "find /workspace/pool -type f | wc -l",
        "command_category": "counting",
        "required_fixtures": {
            "directories": ["/workspace/pool"],
            "files": [
                {"path": "/workspace/pool/a", "content": "1"},
                {"path": "/workspace/pool/b", "content": "2"},
                {"path": "/workspace/pool/c", "content": "3"}
            ]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "3\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Count the length of the longest line in /workspace/lines.txt with wc -L",
        "reference_command": "wc -L < /workspace/lines.txt",
        "command_category": "counting",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/lines.txt",
                "content": "short\nthis is the longest line\ntiny\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "24\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Count total lines across /workspace/a.txt and /workspace/b.txt",
        "reference_command": "wc -l /workspace/a.txt /workspace/b.txt",
        "command_category": "counting",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [
                {"path": "/workspace/a.txt", "content": "1\n2\n"},
                {"path": "/workspace/b.txt", "content": "3\n4\n5\n"}
            ]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout_lines": ["2 /workspace/a.txt", "3 /workspace/b.txt", "5 total"]
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Count lines matching pattern 'error' in /workspace/log.txt using grep -c",
        "reference_command": "grep -c 'error' /workspace/log.txt",
        "command_category": "counting",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/log.txt",
                "content": "error 101\nwarn\nerror 102\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "2\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Save the line count of /workspace/data.csv to /workspace/count.txt",
        "reference_command": "wc -l < /workspace/data.csv > /workspace/count.txt",
        "command_category": "counting",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{
                "path": "/workspace/data.csv",
                "content": "header\nrow1\nrow2\n"
            }]
        },
        "expected_result": {
            "exit_code": 0,
            "file_contents": {"/workspace/count.txt": "3\n"}
        },
        "verification_method": "file_content_match",
        "safety_level": "file_modify"
    })

    # =========================================================================
    # CATEGORY 8: file_copy_move (10 tasks)
    # =========================================================================
    tasks.append({
        "instruction": "Copy /workspace/source.txt to /workspace/destination.txt",
        "reference_command": "cp /workspace/source.txt /workspace/destination.txt",
        "command_category": "file_copy_move",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{"path": "/workspace/source.txt", "content": "file content 123\n"}]
        },
        "expected_result": {
            "exit_code": 0,
            "created_files": ["/workspace/destination.txt"],
            "file_contents": {
                "/workspace/destination.txt": "file content 123\n",
                "/workspace/source.txt": "file content 123\n"
            }
        },
        "verification_method": "file_content_match",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Move /workspace/old_name.txt to /workspace/new_name.txt",
        "reference_command": "mv /workspace/old_name.txt /workspace/new_name.txt",
        "command_category": "file_copy_move",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{"path": "/workspace/old_name.txt", "content": "renamed data\n"}]
        },
        "expected_result": {
            "exit_code": 0,
            "created_files": ["/workspace/new_name.txt"],
            "absent_paths": ["/workspace/old_name.txt"],
            "file_contents": {"/workspace/new_name.txt": "renamed data\n"}
        },
        "verification_method": "file_content_match",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Copy /workspace/doc.txt into the directory /workspace/backup",
        "reference_command": "cp /workspace/doc.txt /workspace/backup/",
        "command_category": "file_copy_move",
        "required_fixtures": {
            "directories": ["/workspace/backup"],
            "files": [{"path": "/workspace/doc.txt", "content": "backup doc"}]
        },
        "expected_result": {
            "exit_code": 0,
            "created_files": ["/workspace/backup/doc.txt"],
            "file_contents": {"/workspace/backup/doc.txt": "backup doc"}
        },
        "verification_method": "file_content_match",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Move /workspace/item.log into /workspace/logs directory",
        "reference_command": "mv /workspace/item.log /workspace/logs/",
        "command_category": "file_copy_move",
        "required_fixtures": {
            "directories": ["/workspace/logs"],
            "files": [{"path": "/workspace/item.log", "content": "item log content"}]
        },
        "expected_result": {
            "exit_code": 0,
            "created_files": ["/workspace/logs/item.log"],
            "absent_paths": ["/workspace/item.log"],
            "file_contents": {"/workspace/logs/item.log": "item log content"}
        },
        "verification_method": "file_content_match",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Recursively copy directory /workspace/src to /workspace/dest",
        "reference_command": "cp -r /workspace/src /workspace/dest",
        "command_category": "file_copy_move",
        "required_fixtures": {
            "directories": ["/workspace/src", "/workspace/src/sub"],
            "files": [
                {"path": "/workspace/src/a.txt", "content": "a"},
                {"path": "/workspace/src/sub/b.txt", "content": "b"}
            ]
        },
        "expected_result": {
            "exit_code": 0,
            "created_files": ["/workspace/dest/a.txt", "/workspace/dest/sub/b.txt"]
        },
        "verification_method": "file_exists",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Copy multiple files a.txt and b.txt to /workspace/archive/",
        "reference_command": "cp /workspace/a.txt /workspace/b.txt /workspace/archive/",
        "command_category": "file_copy_move",
        "required_fixtures": {
            "directories": ["/workspace/archive"],
            "files": [
                {"path": "/workspace/a.txt", "content": "content a"},
                {"path": "/workspace/b.txt", "content": "content b"}
            ]
        },
        "expected_result": {
            "exit_code": 0,
            "created_files": ["/workspace/archive/a.txt", "/workspace/archive/b.txt"]
        },
        "verification_method": "file_exists",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Delete /workspace/obsolete.txt using rm",
        "reference_command": "rm /workspace/obsolete.txt",
        "command_category": "file_copy_move",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{"path": "/workspace/obsolete.txt", "content": "bye"}]
        },
        "expected_result": {
            "exit_code": 0,
            "absent_paths": ["/workspace/obsolete.txt"]
        },
        "verification_method": "file_absence",
        "safety_level": "file_delete"
    })

    tasks.append({
        "instruction": "Force delete /workspace/trash.txt without prompt",
        "reference_command": "rm -f /workspace/trash.txt",
        "command_category": "file_copy_move",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{"path": "/workspace/trash.txt", "content": "trash"}]
        },
        "expected_result": {
            "exit_code": 0,
            "absent_paths": ["/workspace/trash.txt"]
        },
        "verification_method": "file_absence",
        "safety_level": "file_delete"
    })

    tasks.append({
        "instruction": "Recursively remove /workspace/old_build directory",
        "reference_command": "rm -rf /workspace/old_build",
        "command_category": "file_copy_move",
        "required_fixtures": {
            "directories": ["/workspace/old_build", "/workspace/old_build/temp"],
            "files": [{"path": "/workspace/old_build/artifact.bin", "content": "bin"}]
        },
        "expected_result": {
            "exit_code": 0,
            "absent_paths": ["/workspace/old_build"]
        },
        "verification_method": "file_absence",
        "safety_level": "file_delete"
    })

    tasks.append({
        "instruction": "Create a symbolic link /workspace/link.txt pointing to /workspace/target.txt",
        "reference_command": "ln -s /workspace/target.txt /workspace/link.txt",
        "command_category": "file_copy_move",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{"path": "/workspace/target.txt", "content": "target content"}]
        },
        "expected_result": {
            "exit_code": 0,
            "created_files": ["/workspace/link.txt"],
            "file_contents": {"/workspace/link.txt": "target content"}
        },
        "verification_method": "file_content_match",
        "safety_level": "file_modify"
    })

    # =========================================================================
    # CATEGORY 9: file_comparison (10 tasks)
    # =========================================================================
    tasks.append({
        "instruction": "Compare two identical files /workspace/f1.txt and /workspace/f2.txt with diff",
        "reference_command": "diff /workspace/f1.txt /workspace/f2.txt",
        "command_category": "file_comparison",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [
                {"path": "/workspace/f1.txt", "content": "same content\n"},
                {"path": "/workspace/f2.txt", "content": "same content\n"}
            ]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": ""
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Show differences between /workspace/v1.txt and /workspace/v2.txt using diff",
        "reference_command": "diff -u /workspace/v1.txt /workspace/v2.txt",
        "command_category": "file_comparison",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [
                {"path": "/workspace/v1.txt", "content": "line1\nold line\nline3\n"},
                {"path": "/workspace/v2.txt", "content": "line1\nnew line\nline3\n"}
            ]
        },
        "expected_result": {
            "exit_code": 1,
            "stdout_contains": ["-old line", "+new line"]
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Briefly check if /workspace/a.bin and /workspace/b.bin differ using diff -q",
        "reference_command": "diff -q /workspace/a.bin /workspace/b.bin",
        "command_category": "file_comparison",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [
                {"path": "/workspace/a.bin", "content": "data1"},
                {"path": "/workspace/b.bin", "content": "data2"}
            ]
        },
        "expected_result": {
            "exit_code": 1,
            "stdout": "Files /workspace/a.bin and /workspace/b.bin differ\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Compare two files byte by byte using cmp and output nothing if identical",
        "reference_command": "cmp -s /workspace/fileA /workspace/fileB",
        "command_category": "file_comparison",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [
                {"path": "/workspace/fileA", "content": "identical binary data"},
                {"path": "/workspace/fileB", "content": "identical binary data"}
            ]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": ""
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Compare two files ignoring whitespace with diff -w",
        "reference_command": "diff -w /workspace/space1.txt /workspace/space2.txt",
        "command_category": "file_comparison",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [
                {"path": "/workspace/space1.txt", "content": "hello   world\n"},
                {"path": "/workspace/space2.txt", "content": "hello world\n"}
            ]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": ""
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Compare two files ignoring case differences with diff -i",
        "reference_command": "diff -i /workspace/case1.txt /workspace/case2.txt",
        "command_category": "file_comparison",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [
                {"path": "/workspace/case1.txt", "content": "Case Insensitive\n"},
                {"path": "/workspace/case2.txt", "content": "case insensitive\n"}
            ]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": ""
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Compare two directories /workspace/dir1 and /workspace/dir2 using diff -r -q",
        "reference_command": "diff -r -q /workspace/dir1 /workspace/dir2",
        "command_category": "file_comparison",
        "required_fixtures": {
            "directories": ["/workspace/dir1", "/workspace/dir2"],
            "files": [
                {"path": "/workspace/dir1/common.txt", "content": "common"},
                {"path": "/workspace/dir2/common.txt", "content": "common"},
                {"path": "/workspace/dir1/only1.txt", "content": "1"}
            ]
        },
        "expected_result": {
            "exit_code": 1,
            "stdout": "Only in /workspace/dir1: only1.txt\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Find lines common to two sorted files using comm -12",
        "reference_command": "comm -12 /workspace/list1.txt /workspace/list2.txt",
        "command_category": "file_comparison",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [
                {"path": "/workspace/list1.txt", "content": "apple\nbanana\ncherry\n"},
                {"path": "/workspace/list2.txt", "content": "banana\ncherry\ndate\n"}
            ]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "banana\ncherry\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Find lines unique to the first file using comm -23",
        "reference_command": "comm -23 /workspace/setA.txt /workspace/setB.txt",
        "command_category": "file_comparison",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [
                {"path": "/workspace/setA.txt", "content": "1\n2\n3\n"},
                {"path": "/workspace/setB.txt", "content": "2\n3\n4\n"}
            ]
        },
        "expected_result": {
            "exit_code": 0,
            "stdout": "1\n"
        },
        "verification_method": "stdout_match",
        "safety_level": "safe"
    })

    tasks.append({
        "instruction": "Save the unified diff of /workspace/v1 and /workspace/v2 to /workspace/patch.diff",
        "reference_command": "diff -u /workspace/v1 /workspace/v2 > /workspace/patch.diff",
        "command_category": "file_comparison",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [
                {"path": "/workspace/v1", "content": "alpha\n"},
                {"path": "/workspace/v2", "content": "beta\n"}
            ]
        },
        "expected_result": {
            "exit_code": 1,
            "created_files": ["/workspace/patch.diff"]
        },
        "verification_method": "file_exists",
        "safety_level": "file_modify"
    })

    # =========================================================================
    # CATEGORY 10: permissions (10 tasks)
    # =========================================================================
    tasks.append({
        "instruction": "Make /workspace/script.sh executable for user",
        "reference_command": "chmod u+x /workspace/script.sh",
        "command_category": "permissions",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{"path": "/workspace/script.sh", "content": "#!/bin/bash\necho ok\n"}]
        },
        "expected_result": {
            "exit_code": 0,
            "permissions": {"/workspace/script.sh": "744"}
        },
        "verification_method": "permission_match",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Set permissions of /workspace/secret.key to 600 (read/write only by owner)",
        "reference_command": "chmod 600 /workspace/secret.key",
        "command_category": "permissions",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{"path": "/workspace/secret.key", "content": "secret-key-1234"}]
        },
        "expected_result": {
            "exit_code": 0,
            "permissions": {"/workspace/secret.key": "600"}
        },
        "verification_method": "permission_match",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Set permissions of /workspace/public.txt to 644",
        "reference_command": "chmod 644 /workspace/public.txt",
        "command_category": "permissions",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{"path": "/workspace/public.txt", "content": "public info"}]
        },
        "expected_result": {
            "exit_code": 0,
            "permissions": {"/workspace/public.txt": "644"}
        },
        "verification_method": "permission_match",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Set permissions of /workspace/run_all.sh to 755",
        "reference_command": "chmod 755 /workspace/run_all.sh",
        "command_category": "permissions",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{"path": "/workspace/run_all.sh", "content": "#!/bin/sh\n"}]
        },
        "expected_result": {
            "exit_code": 0,
            "permissions": {"/workspace/run_all.sh": "755"}
        },
        "verification_method": "permission_match",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Remove execute permissions from /workspace/locked.sh",
        "reference_command": "chmod -x /workspace/locked.sh",
        "command_category": "permissions",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{"path": "/workspace/locked.sh", "content": "#!/bin/sh\n"}]
        },
        "expected_result": {
            "exit_code": 0,
            "permission_check": "test ! -x /workspace/locked.sh"
        },
        "verification_method": "permission_match",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Set /workspace/readonly.txt to read-only for owner with chmod 400",
        "reference_command": "chmod 400 /workspace/readonly.txt",
        "command_category": "permissions",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{"path": "/workspace/readonly.txt", "content": "read only"}]
        },
        "expected_result": {
            "exit_code": 0,
            "permissions": {"/workspace/readonly.txt": "400"}
        },
        "verification_method": "permission_match",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Add write permission for user to /workspace/notes.txt",
        "reference_command": "chmod u+w /workspace/notes.txt",
        "command_category": "permissions",
        "required_fixtures": {
            "directories": ["/workspace"],
            "files": [{"path": "/workspace/notes.txt", "content": "notes"}]
        },
        "expected_result": {
            "exit_code": 0,
            "permission_check": "test -w /workspace/notes.txt"
        },
        "verification_method": "permission_match",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Recursively set permissions of /workspace/dir to 755",
        "reference_command": "chmod -R 755 /workspace/dir",
        "command_category": "permissions",
        "required_fixtures": {
            "directories": ["/workspace/dir", "/workspace/dir/sub"],
            "files": [{"path": "/workspace/dir/file.txt", "content": "content"}]
        },
        "expected_result": {
            "exit_code": 0,
            "permissions": {
                "/workspace/dir": "755",
                "/workspace/dir/file.txt": "755"
            }
        },
        "verification_method": "permission_match",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Set directory /workspace/shared to 775",
        "reference_command": "chmod 775 /workspace/shared",
        "command_category": "permissions",
        "required_fixtures": {
            "directories": ["/workspace/shared"],
            "files": []
        },
        "expected_result": {
            "exit_code": 0,
            "permissions": {"/workspace/shared": "775"}
        },
        "verification_method": "permission_match",
        "safety_level": "file_modify"
    })

    tasks.append({
        "instruction": "Set permissions on /workspace/private_dir to 700",
        "reference_command": "chmod 700 /workspace/private_dir",
        "command_category": "permissions",
        "required_fixtures": {
            "directories": ["/workspace/private_dir"],
            "files": []
        },
        "expected_result": {
            "exit_code": 0,
            "permissions": {"/workspace/private_dir": "700"}
        },
        "verification_method": "permission_match",
        "safety_level": "file_modify"
    })

    # Validate and assign sequential IDs
    validated_tasks: List[Dict[str, Any]] = []
    for idx, t in enumerate(tasks, start=1):
        t["record_id"] = f"eval_task_{idx:03d}"
        t["source_dataset"] = "westenfelder/NL2SH-ALFA"
        t["requires_network"] = False
        t["requires_privileges"] = False
        t["required_fixtures"] = validate_fixtures(t["required_fixtures"])
        validated_tasks.append(t)

    return validated_tasks


def generate_benchmark_dataset(
    output_path: str = "data/evaluation/evaluation_benchmark.jsonl",
    quarantine_path: str = "data/evaluation/quarantined_candidates.jsonl"
) -> Dict[str, Any]:
    """Generate and write the evaluation benchmark dataset with isolated fixtures."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    tasks = get_curated_benchmark_tasks()

    with open(out_file, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # Sample quarantined tasks showing why external/unsuitable tasks cannot be fixture-generated
    quarantined = [
        {
            "candidate_id": "quar_eval_001",
            "instruction": "Fetch http://example.com/data.tar.gz",
            "command": "curl -O http://example.com/data.tar.gz",
            "quarantine_reason": "network_access_forbidden_in_sandbox"
        },
        {
            "candidate_id": "quar_eval_002",
            "instruction": "Restart nginx web service",
            "command": "sudo systemctl restart nginx",
            "quarantine_reason": "privileged_command_requires_root_and_systemd"
        },
        {
            "candidate_id": "quar_eval_003",
            "instruction": "Export VM state",
            "command": "qm export 101 /path",
            "quarantine_reason": "unavailable_third_party_tool_qm"
        }
    ]

    with open(quarantine_path, "w", encoding="utf-8") as f:
        for q in quarantined:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    return {
        "benchmark_file": str(out_file),
        "total_tasks": len(tasks),
        "quarantined_file": str(quarantine_path),
        "total_quarantined": len(quarantined)
    }


if __name__ == "__main__":
    res = generate_benchmark_dataset()
    print("=" * 60)
    print("BENCHMARK FIXTURE GENERATION COMPLETE")
    print(f"Generated Benchmark: {res['total_tasks']} tasks -> {res['benchmark_file']}")
    print(f"Quarantined Tasks:   {res['total_quarantined']} tasks -> {res['quarantined_file']}")
    print("=" * 60)

