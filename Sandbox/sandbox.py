"""
Secure Docker Sandbox Engine for Natural Language to Shell (NL2SH) execution.
Provides isolated, resource-constrained container environments for executing
and verifying arbitrary Linux shell commands.

Supports both:
- One-shot execution (DockerSandbox): fresh container per task, destroyed after execution.
- Persistent session execution (SessionSandbox): single container maintaining filesystem
  and working directory state across multiple user commands.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass
import json
import subprocess
import time
from typing import Optional
import uuid


@dataclass
class CommandResult:
    """Result of command execution inside the Docker sandbox."""
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration: float


class DockerSandbox:
    """Isolated, resource-constrained Docker execution sandbox (one-shot mode)."""

    DEFAULT_IMAGE: str = "nl2sh-sandbox:latest"

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        auto_start: bool = True
    ) -> None:
        self.image = image
        self.container_id: Optional[str] = None
        self._closed: bool = False

        # Register secondary cleanup via atexit
        atexit.register(self.close)

        if auto_start:
            self.start()

    def start(self) -> None:
        """Start a new sandboxed Docker container with strict security restrictions."""
        if self.container_id and not self._closed:
            return

        cmd = [
            "docker", "run", "-d",
            "--init",
            "--network", "none",
            "--user", "1000:1000",
            "--memory", "256m",
            "--memory-swap", "256m",
            "--cpus", "1.0",
            "--pids-limit", "64",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true",
            "--read-only",
            "--tmpfs", "/workspace:rw,nosuid,nodev,size=64m,uid=1000,gid=1000",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--tmpfs", "/run:rw,noexec,nosuid,size=16m",
            self.image,
            "sleep", "infinity"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                shell=False
            )
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.strip() if exc.stderr else str(exc)
            raise RuntimeError(f"Failed to start Docker sandbox container: {err}") from exc
        except FileNotFoundError as exc:
            raise RuntimeError("Docker CLI executable not found in PATH.") from exc

        self.container_id = result.stdout.strip()
        if not self.container_id:
            raise RuntimeError("Docker did not return a valid container ID.")
        self._closed = False

    def execute(self, command: str, timeout: float = 10.0) -> CommandResult:
        """
        Execute a shell command inside the sandbox using Bash under GNU timeout.

        Args:
            command: The natural language translated shell command to execute.
            timeout: Maximum allowed execution time in seconds.

        Returns:
            CommandResult with stdout, stderr, exit code, timeout flag, and duration.
        """
        if not command or not command.strip():
            raise ValueError("Command cannot be empty or whitespace only.")

        if timeout is None or timeout <= 0:
            raise ValueError(f"Timeout must be a positive number, got: {timeout}")

        if not self.container_id or self._closed:
            raise RuntimeError("Sandbox container is not running or has been closed.")

        # Normalize Windows CRLF line endings to Linux LF
        normalized_command = command.replace("\r\n", "\n").replace("\r", "\n")

        # In-container timeout using GNU timeout with TERM signal and 1s SIGKILL grace period
        duration_str = f"{timeout}s"
        exec_cmd = [
            "docker", "exec", "-i",
            "--user", "1000:1000",
            self.container_id,
            "timeout", "--verbose", "--signal=TERM", "--kill-after=1s", duration_str,
            "bash", "-c", normalized_command
        ]

        # Host-side subprocess timeout as a secondary fallback limit
        host_timeout = timeout + 5.0
        start_time = time.perf_counter()

        try:
            proc = subprocess.run(
                exec_cmd,
                capture_output=True,
                text=True,
                timeout=host_timeout,
                shell=False
            )
            duration = time.perf_counter() - start_time
            stdout = proc.stdout
            stderr = proc.stderr
            exit_code = proc.returncode
            timed_out = False

            # Accurate timeout detection:
            # 1. GNU timeout with --verbose always prints "timeout: sending signal <SIG> to command 'bash'"
            # 2. Or exit_code is 124 (GNU timeout's returncode) and the actual duration met or exceeded the deadline
            if "timeout: sending signal" in stderr or (exit_code == 124 and duration >= timeout):
                timed_out = True
                if "command timed out" not in stderr.lower():
                    stderr = (
                        f"{stderr}\nCommand timed out after {timeout} seconds."
                        if stderr
                        else f"Command timed out after {timeout} seconds."
                    )

            return CommandResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                timed_out=timed_out,
                duration=duration
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start_time
            out = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout.decode() if exc.stdout else "")
            err = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode() if exc.stderr else "")
            err_msg = (
                f"{err}\nCommand timed out after {timeout} seconds."
                if err
                else f"Command timed out after {timeout} seconds."
            )
            return CommandResult(
                stdout=out or "",
                stderr=err_msg,
                exit_code=-1,
                timed_out=True,
                duration=duration
            )
        except subprocess.CalledProcessError as exc:
            duration = time.perf_counter() - start_time
            return CommandResult(
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                exit_code=exc.returncode,
                timed_out=False,
                duration=duration
            )

    def reset(self) -> DockerSandbox:
        """Reset the sandbox by destroying the existing container and starting a fresh one."""
        self.close()
        self.start()
        return self

    def close(self) -> None:
        """Idempotently stop and remove the sandboxed container."""
        try:
            atexit.unregister(self.close)
        except Exception:
            pass

        if self._closed or not self.container_id:
            return

        cid = self.container_id
        self.container_id = None
        self._closed = True

        try:
            subprocess.run(
                ["docker", "rm", "-f", cid],
                capture_output=True,
                text=True,
                shell=False,
                timeout=15
            )
        except Exception:
            # Idempotent cleanup: suppress errors if docker or container already stopped
            pass

    def inspect(self) -> dict:
        """Inspect the low-level container configuration."""
        if not self.container_id or self._closed:
            raise RuntimeError("Sandbox container is not running.")

        proc = subprocess.run(
            ["docker", "inspect", self.container_id],
            capture_output=True,
            text=True,
            check=True,
            shell=False
        )
        return json.loads(proc.stdout)[0]

    def __enter__(self) -> DockerSandbox:
        if not self.container_id or self._closed:
            self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class SessionSandbox(DockerSandbox):
    """
    Persistent session sandbox maintaining filesystem and working directory state
    across multiple shell command executions in the same container.

    Limitations:
    - Shell variables and aliases do not persist between commands (each command runs under a fresh subshell invocation).
    - The filesystem state and working directory persist across commands.
    """

    def __init__(
        self,
        image: str = DockerSandbox.DEFAULT_IMAGE,
        auto_start: bool = True
    ) -> None:
        self.current_dir: str = "/workspace"
        super().__init__(image=image, auto_start=auto_start)

    def reset(self) -> SessionSandbox:
        """Reset the session sandbox by destroying the container, spawning a fresh one, and resetting cwd to /workspace."""
        super().reset()
        self.current_dir = "/workspace"
        return self

    def execute(self, command: str, timeout: float = 10.0) -> CommandResult:
        """
        Execute a shell command inside the persistent session container.
        Tracks and updates current_dir when directory changes occur.

        Args:
            command: The natural language translated shell command to execute.
            timeout: Maximum allowed execution time in seconds.

        Returns:
            CommandResult with stdout, stderr, exit code, timeout flag, and duration.
        """
        if not command or not command.strip():
            raise ValueError("Command cannot be empty or whitespace only.")

        if timeout is None or timeout <= 0:
            raise ValueError(f"Timeout must be a positive number, got: {timeout}")

        if not self.container_id or self._closed:
            raise RuntimeError("Sandbox container is not running or has been closed.")

        # Normalize Windows CRLF line endings to Linux LF
        normalized_command = command.replace("\r\n", "\n").replace("\r", "\n")

        # Unique marker for directory tracking to prevent collisions with user output
        marker = f"__NL2SH_DIR_{uuid.uuid4().hex}__"
        duration_str = f"{timeout}s"

        # Safe wrapper executed inside the container:
        # - $1 is the normalized user command
        # - $2 is the unique marker
        # 1. eval "$1" executes the user command in the working directory
        # 2. __NL2SH_RET__=$? captures the command's exact exit code
        # 3. echo "" guarantees the marker begins on a fresh line
        # 4. echo "$2:$PWD" outputs the marker and updated directory
        # 5. exit $__NL2SH_RET__ returns the user command's true exit code
        wrapper_script = (
            'eval "$1"\n'
            '__NL2SH_RET__=$?\n'
            'echo ""\n'
            'echo "$2:$PWD"\n'
            'exit $__NL2SH_RET__'
        )

        exec_cmd = [
            "docker", "exec", "-i",
            "--user", "1000:1000",
            "-w", self.current_dir,
            self.container_id,
            "timeout", "--verbose", "--signal=TERM", "--kill-after=1s", duration_str,
            "bash", "-c", wrapper_script,
            "_",
            normalized_command,
            marker
        ]

        host_timeout = timeout + 5.0
        start_time = time.perf_counter()

        try:
            proc = subprocess.run(
                exec_cmd,
                capture_output=True,
                text=True,
                timeout=host_timeout,
                shell=False
            )
            duration = time.perf_counter() - start_time
            raw_stdout = proc.stdout
            stderr = proc.stderr
            exit_code = proc.returncode
            timed_out = False

            # Check timeout via GNU timeout diagnostic or returncode 124 deadline
            if "timeout: sending signal" in stderr or (exit_code == 124 and duration >= timeout):
                timed_out = True
                if "command timed out" not in stderr.lower():
                    stderr = (
                        f"{stderr}\nCommand timed out after {timeout} seconds."
                        if stderr
                        else f"Command timed out after {timeout} seconds."
                    )

            # Parse directory marker from stdout
            marker_pattern = f"\n{marker}:"
            if marker_pattern in raw_stdout:
                parts = raw_stdout.rsplit(marker_pattern, 1)
                user_stdout = parts[0]
                marker_rest = parts[1]
                new_dir = marker_rest.splitlines()[0].strip() if marker_rest.splitlines() else ""

                # Update current_dir only when command succeeded and resulting directory is valid
                if exit_code == 0 and not timed_out and new_dir and new_dir.startswith("/"):
                    self.current_dir = new_dir
            else:
                user_stdout = raw_stdout

            return CommandResult(
                stdout=user_stdout,
                stderr=stderr,
                exit_code=exit_code,
                timed_out=timed_out,
                duration=duration
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start_time
            out = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout.decode() if exc.stdout else "")
            err = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode() if exc.stderr else "")
            err_msg = (
                f"{err}\nCommand timed out after {timeout} seconds."
                if err
                else f"Command timed out after {timeout} seconds."
            )
            return CommandResult(
                stdout=out or "",
                stderr=err_msg,
                exit_code=-1,
                timed_out=True,
                duration=duration
            )
        except subprocess.CalledProcessError as exc:
            duration = time.perf_counter() - start_time
            return CommandResult(
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                exit_code=exc.returncode,
                timed_out=False,
                duration=duration
            )

    def __enter__(self) -> SessionSandbox:
        if not self.container_id or self._closed:
            self.start()
        return self


# Backward compatibility alias
Sandbox = DockerSandbox
