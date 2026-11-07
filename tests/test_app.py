"""
Integration tests for app.py.
Verifies natural-language to shell translation, execution inside DockerSandbox,
exit code reporting, container cleanup, and graceful error handling.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import convert_to_shell, main, run_in_docker
from Sandbox.sandbox import CommandResult, DockerSandbox, SessionSandbox


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

