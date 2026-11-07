"""
Tests for NL2SH DockerSandbox engine.
Verifies security restrictions, command execution, timeouts, network isolation,
container lifecycle, and resource constraints without running harmful commands.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

# Add project root to sys.path so Sandbox package can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Sandbox.sandbox import CommandResult, DockerSandbox


def test_basic_execution():
    """1. Basic execution: echo 'hello world'."""
    with DockerSandbox() as sb:
        result = sb.execute("echo 'hello world'")
        assert result.exit_code == 0
        assert result.stdout.strip() == "hello world"
        assert not result.timed_out
        assert result.duration >= 0.0


def test_non_root_execution():
    """2. Non-root execution: id -u should return 1000."""
    with DockerSandbox() as sb:
        result = sb.execute("id -u")
        assert result.exit_code == 0
        assert result.stdout.strip() == "1000"

        # Also verify group id
        result_gid = sb.execute("id -g")
        assert result_gid.exit_code == 0
        assert result_gid.stdout.strip() == "1000"


def test_command_timeout_reliable():
    """
    3. Reliable command timeout:
    - Runs sleep 30 with a 2-second timeout.
    - Confirms timed_out=True.
    - Confirms execution returns promptly.
    - Checks that the timed-out process is no longer running inside the container.
    """
    with DockerSandbox() as sb:
        result = sb.execute("sleep 30", timeout=2.0)
        assert result.timed_out is True
        assert result.exit_code in (124, 137, -1)
        assert "timed out" in result.stderr.lower()

        # Confirms execution returns promptly (around 2s, well under 6s)
        assert result.duration < 6.0

        # Check that the timed-out process is no longer running inside the container
        ps_res = sb.execute("ps -eo pid,comm,args")
        assert ps_res.exit_code == 0
        # sleep 30 should not appear in process list (only sleep infinity PID 1 may exist)
        assert "sleep 30" not in ps_res.stdout


def test_child_process_termination_on_timeout():
    """
    Confirm background/child processes in subshells are terminated when timeout triggers.
    """
    with DockerSandbox() as sb:
        # Spawn child processes in a subshell that waits on them
        result = sb.execute("bash -c '(sleep 30 & sleep 30 & wait)'", timeout=2.0)
        assert result.timed_out is True
        assert result.exit_code in (124, 137, -1)
        assert result.duration < 6.0

        # Verify no sleep processes survived in the container
        ps_res = sb.execute("ps -eo pid,comm,args")
        assert ps_res.exit_code == 0
        lines = [line for line in ps_res.stdout.splitlines() if "sleep 30" in line]
        assert len(lines) == 0, f"Found surviving child processes: {lines}"


def test_legitimate_nonzero_exit_codes_not_treated_as_timeout():
    """
    Verify commands returning exit codes 124, 137, or 1 are not falsely marked as timed out.
    """
    with DockerSandbox() as sb:
        res_1 = sb.execute("exit 1", timeout=5.0)
        assert res_1.exit_code == 1
        assert res_1.timed_out is False

        res_124 = sb.execute("exit 124", timeout=5.0)
        assert res_124.exit_code == 124
        assert res_124.timed_out is False

        res_137 = sb.execute("exit 137", timeout=5.0)
        assert res_137.exit_code == 137
        assert res_137.timed_out is False


def test_network_isolation():
    """4. Network isolation: attempt a short network connection and verify it fails."""
    with DockerSandbox() as sb:
        # Attempt to curl a public IP with 1 second timeout
        result = sb.execute("curl --connect-timeout 1 -s http://1.1.1.1 || exit 42")
        # Exit code should be non-zero (42 or curl connection failure exit code)
        assert result.exit_code != 0

        # Also verify loopback is present but no routable external network
        ip_res = sb.execute("ip route show")
        assert "default" not in ip_res.stdout


def test_working_directory():
    """5. Working directory: pwd should return /workspace."""
    with DockerSandbox() as sb:
        result = sb.execute("pwd")
        assert result.exit_code == 0
        assert result.stdout.strip() == "/workspace"


def test_workspace_writable_and_root_readonly():
    """
    6. Writable workspace with read-only root:
    - Create a file inside /workspace.
    - Write content to it.
    - Read the content back.
    - Delete the file.
    - Verify that the root filesystem remains read-only.
    """
    with DockerSandbox() as sb:
        # 1. Create a file inside /workspace and write content to it
        write_res = sb.execute("echo 'nl2sh test payload' > /workspace/sample.txt")
        assert write_res.exit_code == 0

        # Verify file exists
        ls_res = sb.execute("ls -la /workspace/sample.txt")
        assert ls_res.exit_code == 0
        assert "sample.txt" in ls_res.stdout
        assert "sandboxuser" in ls_res.stdout

        # 2. Append more content
        append_res = sb.execute("echo 'second line' >> /workspace/sample.txt")
        assert append_res.exit_code == 0

        # 3. Read the content back
        read_res = sb.execute("cat /workspace/sample.txt")
        assert read_res.exit_code == 0
        assert "nl2sh test payload" in read_res.stdout
        assert "second line" in read_res.stdout

        # 4. Delete the file
        del_res = sb.execute("rm /workspace/sample.txt")
        assert del_res.exit_code == 0

        # Verify file is deleted
        check_del = sb.execute("test -e /workspace/sample.txt")
        assert check_del.exit_code != 0

        # 5. Verify that the root filesystem remains read-only
        root_write = sb.execute("touch /etc/forbidden.txt")
        assert root_write.exit_code != 0
        assert "read-only file system" in root_write.stderr.lower()

        root_write2 = sb.execute("touch /forbidden.txt")
        assert root_write2.exit_code != 0
        assert "read-only file system" in root_write2.stderr.lower()


def test_cleanup():
    """7. Cleanup: verify the container is removed after exiting the context manager."""
    container_id = None
    with DockerSandbox() as sb:
        container_id = sb.container_id
        assert container_id is not None
        # Verify container is alive while in context
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


def test_repeated_close_idempotent():
    """Verify multiple close calls on the same sandbox do not error."""
    sb = DockerSandbox()
    cid = sb.container_id
    assert cid is not None
    sb.close()
    sb.close()  # Repeated call
    sb.close()  # Third call
    assert sb.container_id is None
    assert sb._closed is True


def test_reset():
    """8. Reset: verify the sandbox can execute commands after reset."""
    with DockerSandbox() as sb:
        first_cid = sb.container_id
        res1 = sb.execute("echo 'first session'")
        assert res1.exit_code == 0
        assert res1.stdout.strip() == "first session"

        # Perform reset
        sb.reset()
        second_cid = sb.container_id

        # New container should have been spawned
        assert second_cid is not None
        assert second_cid != first_cid

        # Verify command execution works in the fresh container
        res2 = sb.execute("echo 'second session'")
        assert res2.exit_code == 0
        assert res2.stdout.strip() == "second session"


def test_security_restrictions_inspection():
    """Verify resource limits and security constraints via Docker inspect (no fork bombs)."""
    with DockerSandbox() as sb:
        inspect_data = sb.inspect()
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

        # Verify workspace mount options include nosuid, nodev, size=64m
        ws_opts = tmpfs.get("/workspace", "")
        assert "nosuid" in ws_opts
        assert "nodev" in ws_opts
        assert "size=64m" in ws_opts


def test_command_validation():
    """Verify input validation for empty commands and invalid timeouts."""
    with DockerSandbox() as sb:
        with pytest.raises(ValueError, match="empty"):
            sb.execute("")

        with pytest.raises(ValueError, match="empty"):
            sb.execute("   ")

        with pytest.raises(ValueError, match="positive"):
            sb.execute("ls", timeout=-1)

        with pytest.raises(ValueError, match="positive"):
            sb.execute("ls", timeout=0)
