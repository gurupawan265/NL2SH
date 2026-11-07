"""
Integration tests for NL2SH SessionSandbox.
Verifies multi-command persistence:
- Working directory changes across commands
- Filesystem changes across commands
- Invalid directory handling
- Nested directories
- Session cleanup and idempotent teardown
- Security restrictions preservation
- Session reset
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Sandbox.sandbox import CommandResult, SessionSandbox


def test_directory_persistence():
    """
    Test A: Directory persistence
    - mkdir project
    - cd project
    - pwd
    Expected: /workspace/project
    """
    with SessionSandbox() as session:
        assert session.current_dir == "/workspace"

        res_mkdir = session.execute("mkdir project")
        assert res_mkdir.exit_code == 0

        res_cd = session.execute("cd project")
        assert res_cd.exit_code == 0
        assert session.current_dir == "/workspace/project"

        res_pwd = session.execute("pwd")
        assert res_pwd.exit_code == 0
        assert res_pwd.stdout.strip() == "/workspace/project"


def test_file_persistence():
    """
    Test B: File persistence
    - touch a.txt
    - ls
    Expected: a.txt appears in ls
    """
    with SessionSandbox() as session:
        res_touch = session.execute("touch a.txt")
        assert res_touch.exit_code == 0

        res_ls = session.execute("ls")
        assert res_ls.exit_code == 0
        assert "a.txt" in res_ls.stdout


def test_invalid_directory():
    """
    Test C: Invalid directory
    - cd nonexistent
    Verify nonzero exit code.
    Verify current_dir remains unchanged.
    """
    with SessionSandbox() as session:
        initial_dir = session.current_dir
        assert initial_dir == "/workspace"

        res_bad_cd = session.execute("cd nonexistent")
        assert res_bad_cd.exit_code != 0
        assert "no such file or directory" in res_bad_cd.stderr.lower()
        # Working directory must remain unchanged
        assert session.current_dir == initial_dir

        # Verify pwd inside container also confirms initial_dir
        res_pwd = session.execute("pwd")
        assert res_pwd.stdout.strip() == initial_dir


def test_nested_directories():
    """
    Test D: Nested directories
    - mkdir -p a/b
    - cd a/b
    - pwd
    Expected: /workspace/a/b
    """
    with SessionSandbox() as session:
        res_mkdir = session.execute("mkdir -p a/b")
        assert res_mkdir.exit_code == 0

        res_cd = session.execute("cd a/b")
        assert res_cd.exit_code == 0
        assert session.current_dir == "/workspace/a/b"

        res_pwd = session.execute("pwd")
        assert res_pwd.exit_code == 0
        assert res_pwd.stdout.strip() == "/workspace/a/b"

        # Also test moving back up with cd ..
        res_cd_up = session.execute("cd ..")
        assert res_cd_up.exit_code == 0
        assert session.current_dir == "/workspace/a"

        res_pwd_up = session.execute("pwd")
        assert res_pwd_up.exit_code == 0
        assert res_pwd_up.stdout.strip() == "/workspace/a"


def test_session_cleanup():
    """
    Test E: Session cleanup
    - Record container ID.
    - Close session.
    - Verify the container no longer exists.
    """
    container_id = None
    with SessionSandbox() as session:
        container_id = session.container_id
        assert container_id is not None
        # Verify container is alive
        check_proc = subprocess.run(
            ["docker", "inspect", container_id],
            capture_output=True,
            text=True
        )
        assert check_proc.returncode == 0

    # Outside context: verify container has been completely removed
    verify_proc = subprocess.run(
        ["docker", "inspect", container_id],
        capture_output=True,
        text=True
    )
    assert verify_proc.returncode != 0


def test_security_restrictions_preserved():
    """
    Test F: Security
    Inspect the persistent container.
    Verify that all 11 existing security restrictions remain active.
    """
    with SessionSandbox() as session:
        inspect_data = session.inspect()
        host_config = inspect_data.get("HostConfig", {})

        # Network none
        assert host_config.get("NetworkMode") == "none"

        # User 1000:1000
        assert inspect_data.get("Config", {}).get("User") == "1000:1000"

        # Memory 256m (268435456 bytes)
        assert host_config.get("Memory") == 256 * 1024 * 1024

        # Memory swap 256m (268435456 bytes)
        assert host_config.get("MemorySwap") == 256 * 1024 * 1024

        # CPUs 1.0 (NanoCpus = 1000000000)
        assert host_config.get("NanoCpus") == 1_000_000_000

        # PIDs limit 64
        assert host_config.get("PidsLimit") == 64

        # Cap drop ALL
        cap_drop = host_config.get("CapDrop", [])
        assert "ALL" in cap_drop or "all" in cap_drop

        # Security opt no-new-privileges
        sec_opts = host_config.get("SecurityOpt", [])
        assert any("no-new-privileges" in opt for opt in sec_opts)

        # Read-only rootfs
        assert host_config.get("ReadonlyRootfs") is True

        # Init process enabled
        assert host_config.get("Init") is True

        # Tmpfs mounts (/workspace, /tmp, and /run)
        tmpfs = host_config.get("Tmpfs", {})
        assert "/workspace" in tmpfs
        assert "/tmp" in tmpfs
        assert "/run" in tmpfs


def test_session_reset():
    """
    Test G: Reset
    - Reset the session.
    - Verify current_dir returns to /workspace.
    - Verify a fresh container is used.
    """
    with SessionSandbox() as session:
        first_cid = session.container_id

        # Navigate into a subfolder
        session.execute("mkdir testdir && cd testdir")
        assert session.current_dir == "/workspace/testdir"

        # Reset session
        session.reset()
        second_cid = session.container_id

        # Verify new container and directory reset
        assert second_cid != first_cid
        assert session.current_dir == "/workspace"

        # Verify testdir does not exist in fresh container
        res = session.execute("ls")
        assert "testdir" not in res.stdout

