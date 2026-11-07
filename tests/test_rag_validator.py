"""
Unit tests for NL2SH CommandValidator.
Tests extraction, markdown fence handling, label stripping, alternative rejection,
multi-line validation, and safety risk warnings.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RAG.validator import CommandValidator, ValidationResult


@pytest.fixture
def validator():
    return CommandValidator()


def test_empty_and_whitespace_responses(validator):
    """Test empty, whitespace, or None responses are rejected."""
    r1 = validator.validate("")
    assert r1.is_valid is False
    assert "Empty response" in r1.rejection_reason

    r2 = validator.validate("   \n\t  ")
    assert r2.is_valid is False
    assert "Empty response" in r2.rejection_reason

    r3 = validator.validate(None)
    assert r3.is_valid is False


def test_valid_single_line_command(validator):
    """Test clean single-line commands are accepted without alteration."""
    res = validator.validate("ls -la /workspace")
    assert res.is_valid is True
    assert res.command == "ls -la /workspace"
    assert res.rejection_reason is None
    assert len(res.warnings) == 0


def test_markdown_code_fence_extraction(validator):
    """Test extracting commands from markdown ```bash and ``` blocks."""
    # Standard ```bash block
    r1 = validator.validate("```bash\nfind /workspace -name '*.txt'\n```")
    assert r1.is_valid is True
    assert r1.command == "find /workspace -name '*.txt'"

    # Generic ``` block
    r2 = validator.validate("```\ngrep -i 'error' /workspace/server.log\n```")
    assert r2.is_valid is True
    assert r2.command == "grep -i 'error' /workspace/server.log"

    # Inline backticks: `chmod 755 script.sh`
    r3 = validator.validate("`chmod 755 script.sh`")
    assert r3.is_valid is True
    assert r3.command == "chmod 755 script.sh"


def test_malformed_unclosed_markdown_fence(validator):
    """Test that unclosed code fences are detected and rejected."""
    res = validator.validate("```bash\nls -la")
    assert res.is_valid is False
    assert "Unclosed code fence" in res.rejection_reason


def test_introductory_label_stripping(validator):
    """Test stripping simple label prefixes like 'Command:', 'Bash:', or '$ '."""
    r1 = validator.validate("Command: ls -l")
    assert r1.is_valid is True
    assert r1.command == "ls -l"

    r2 = validator.validate("Bash: grep 'foo' file.txt")
    assert r2.is_valid is True
    assert r2.command == "grep 'foo' file.txt"

    r3 = validator.validate("$ pwd")
    assert r3.is_valid is True
    assert r3.command == "pwd"


def test_multiple_alternative_commands_rejected(validator):
    """Test that responses offering multiple options are rejected."""
    # Numbered options
    alt1 = "Option 1: ls -l\nOption 2: ls -la"
    r1 = validator.validate(alt1)
    assert r1.is_valid is False
    assert "Multiple alternative commands" in r1.rejection_reason

    # Conversational alternatives
    alt2 = "ls -l or you can use find . -maxdepth 1"
    r2 = validator.validate(alt2)
    assert r2.is_valid is False
    assert "Multiple alternative commands" in r2.rejection_reason


def test_unexplained_multiline_response_rejected(validator):
    """Test multi-line responses without continuation or heredoc are rejected."""
    multi = "cd /workspace\nls -l\npwd"
    res = validator.validate(multi)
    assert res.is_valid is False
    assert "Unexplained multi-line response" in res.rejection_reason


def test_legitimate_multiline_continuation_accepted(validator):
    """Test valid multiline commands using trailing backslashes are accepted."""
    cmd = "find /workspace \\\n  -type f \\\n  -name '*.py'"
    res = validator.validate(cmd)
    assert res.is_valid is True
    assert "find /workspace" in res.command
    assert "-name '*.py'" in res.command


def test_pipes_redirection_and_quotes_preserved(validator):
    """Test that complex valid shell pipelines and redirections are preserved intact."""
    complex_cmd = "find /workspace -type f | grep -E '\\.log$' | sort | wc -l > /workspace/count.txt"
    res = validator.validate(complex_cmd)
    assert res.is_valid is True
    assert res.command == complex_cmd
    assert len(res.warnings) == 0


def test_unbalanced_quotes_syntax_error(validator):
    """Test that commands with mismatched unclosed quotes are rejected."""
    bad_quote = "find /workspace -name 'unclosed.txt"
    res = validator.validate(bad_quote)
    assert res.is_valid is False
    assert "Unbalanced quotes" in res.rejection_reason


def test_destructive_operation_warnings(validator):
    """Test that destructive commands generate warnings but remain valid."""
    r1 = validator.validate("rm -rf /workspace/build")
    assert r1.is_valid is True
    assert any("Destructive" in w and "rm -r" in w for w in r1.warnings)

    r2 = validator.validate("rm -f /workspace/temp.txt")
    assert r2.is_valid is True
    assert any("Destructive" in w for w in r2.warnings)

    r3 = validator.validate("dd if=/dev/zero of=/workspace/disk.img bs=1M count=10")
    assert r3.is_valid is True
    assert any("dd" in w for w in r3.warnings)


def test_privileged_and_network_operation_warnings(validator):
    """Test that sudo, su, and network tools (curl, wget) generate warnings."""
    r1 = validator.validate("sudo chmod 755 /etc/hosts")
    assert r1.is_valid is True
    assert any("Privileged" in w for w in r1.warnings)

    r2 = validator.validate("curl -s https://example.com/api | grep 'status'")
    assert r2.is_valid is True
    assert any("Network" in w for w in r2.warnings)


def test_clarification_request_detection(validator):
    """Test that model clarification requests are recognized."""
    clarif = "CLARIFICATION: Which directory would you like to search in?"
    res = validator.validate(clarif)
    assert res.is_valid is False
    assert res.is_clarification is True
    assert "Which directory" in res.rejection_reason

