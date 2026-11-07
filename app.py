"""
Natural Language to Shell (NL2SH) application interface.
Translates natural language instructions into shell commands and executes them
safely inside a secure Docker sandbox.

Supports:
- Persistent interactive REPL session (SessionSandbox) with cwd and filesystem tracking.
- One-shot execution (run_in_docker) for isolated evaluation and automated tasks.
"""

from __future__ import annotations

import re
import shlex
import sys
from Sandbox.sandbox import CommandResult, DockerSandbox, SessionSandbox


def convert_to_shell(user_input: str) -> str:
    """
    Rule-based converter translating simple English queries to shell commands.
    Preserves basic file/folder queries and provides safe directory navigation.
    """
    trimmed = user_input.strip()
    lowered = trimmed.lower()

    if "list files" in lowered:
        return "ls"
    elif "current directory" in lowered:
        return "pwd"
    elif "show root folders" in lowered:
        return "ls /"
    elif "create folder" in lowered:
        return "mkdir new_folder"
    elif lowered in ("go back", "go to parent directory"):
        return "cd .."

    # Navigation patterns: "change directory [to] <dir>", "go to <dir>", "enter <dir>"
    nav_patterns = [
        r"^change directory (?:to )?(.+)$",
        r"^go to (.+)$",
        r"^enter (.+)$",
    ]
    for pattern in nav_patterns:
        match = re.match(pattern, trimmed, re.IGNORECASE)
        if match:
            target = match.group(1).strip()
            if target.lower() == "parent directory":
                return "cd .."
            # Strip user-supplied wrapping quotes if present
            if (target.startswith('"') and target.endswith('"')) or (target.startswith("'") and target.endswith("'")):
                target = target[1:-1].strip()
            if target:
                return f"cd {shlex.quote(target)}"

    return "echo Command not understood"


def run_in_docker(command: str, timeout: float = 10.0) -> CommandResult:
    """
    Execute a shell command inside the secure Docker sandbox (one-shot mode).
    Guarantees container cleanup via context manager.
    """
    try:
        with DockerSandbox() as sandbox:
            return sandbox.execute(command, timeout=timeout)
    except Exception as exc:
        return CommandResult(
            stdout="",
            stderr=f"Docker sandbox error: {exc}",
            exit_code=-1,
            timed_out=False,
            duration=0.0
        )


def main() -> None:
    """Interactive CLI REPL for NL2SH using a persistent SessionSandbox."""
    try:
        session = SessionSandbox()
    except Exception as exc:
        print(f"Docker sandbox error: {exc}", file=sys.stderr)
        return

    try:
        while True:
            prompt = f"nl2sh:{session.current_dir}$ "
            try:
                user_input = input(prompt)
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            cleaned_input = user_input.strip()
            if not cleaned_input:
                continue

            if cleaned_input.lower() in ("exit", "quit"):
                print("Exiting.")
                break

            shell_command = convert_to_shell(cleaned_input)
            print("Generated shell command:", shell_command)

            result = session.execute(shell_command)

            if result.stdout:
                print("OUTPUT:\n" + result.stdout, end="" if result.stdout.endswith("\n") else "\n")
            if result.stderr:
                print("ERROR:\n" + result.stderr, end="" if result.stderr.endswith("\n") else "\n")
            print(f"EXIT CODE: {result.exit_code}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
