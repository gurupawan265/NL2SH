"""
Validation and extraction engine for NL2SH generated shell commands.
Performs deterministic output cleaning, markdown extraction, safety risk analysis,
and ambiguity detection without changing command semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import shlex
from typing import List, Optional, Tuple


@dataclass
class ValidationResult:
    """Structured result of command validation and risk assessment."""

    is_valid: bool
    command: str = ""
    raw_extracted: str = ""
    warnings: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    is_clarification: bool = False


class CommandValidator:
    """
    Validates, sanitizes, and evaluates safety risks of LLM-generated shell commands.
    Distinguishes syntax extraction from execution safety proofs.
    """

    # Operations requiring caution warnings
    DESTRUCTIVE_PATTERNS = [
        (r"\brm\s+-[a-zA-Z]*r", "Recursive directory deletion (rm -r)"),
        (r"\brm\s+-[a-zA-Z]*f", "Forced deletion without confirmation (rm -f)"),
        (r"\brm\b", "File deletion (rm)"),
        (r"\bdd\b", "Low-level bitstream copy (dd)"),
        (r"\bmkfs\b", "Filesystem creation/format (mkfs)"),
        (r"\bshred\b", "Secure file shredding (shred)"),
        (r"\bwipefs\b", "Filesystem signature wiping (wipefs)"),
        (r">\s*/dev/sd[a-z]", "Direct block device write"),
    ]

    PRIVILEGED_PATTERNS = [
        (r"\bsudo\b", "Elevated superuser execution (sudo)"),
        (r"\bsu\b", "User identity switch (su)"),
        (r"\bchown\b", "File ownership alteration (chown)"),
        (r"\bchmod\s+-[a-zA-Z]*R", "Recursive permission modification (chmod -R)"),
        (r"\bdoas\b", "Privileged execution (doas)"),
    ]

    NETWORK_PATTERNS = [
        (r"\bcurl\b", "Network data transfer (curl)"),
        (r"\bwget\b", "Network file download (wget)"),
        (r"\bnc\b", "Arbitrary network connection (nc / netcat)"),
        (r"\bssh\b", "Secure shell remote access (ssh)"),
        (r"\bscp\b", "Secure remote file copy (scp)"),
        (r"\bftp\b", "File transfer protocol (ftp)"),
    ]

    # Patterns indicating multiple alternative options
    ALTERNATIVE_PATTERNS = [
        r"(?i)\b(?:option\s*\d|alternative\s*\d|choice\s*\d):",
        r"(?i)\bor\s+you\s+can\s+use\b",
        r"(?i)\beither\s+.*?\s+or\b",
        r"(?i)^\s*(?:1\.|2\.)\s+[a-zA-Z0-9_\-]+",
    ]

    def validate(self, raw_response: str) -> ValidationResult:
        """
        Validate and extract a single Bash command from model output.
        """
        if not raw_response or not raw_response.strip():
            return ValidationResult(
                is_valid=False,
                rejection_reason="Empty response from model",
            )

        trimmed = raw_response.strip()

        # Check for model clarification request
        if trimmed.startswith("CLARIFICATION:"):
            clarification_msg = trimmed[len("CLARIFICATION:"):].strip()
            return ValidationResult(
                is_valid=False,
                command="",
                raw_extracted=trimmed,
                rejection_reason=f"Model requested clarification: {clarification_msg}",
                is_clarification=True,
            )

        # 1. Extract command from markdown code fences if present
        extracted, extract_err = self._extract_from_markdown(trimmed)
        if extract_err:
            return ValidationResult(is_valid=False, rejection_reason=extract_err)

        # 2. Strip single introductory command labels (e.g. "Command: ls", "$ ls")
        cleaned = self._strip_introductory_labels(extracted)

        # 3. Check for multiple alternative choices
        for alt_pat in self.ALTERNATIVE_PATTERNS:
            if re.search(alt_pat, trimmed, re.MULTILINE):
                return ValidationResult(
                    is_valid=False,
                    raw_extracted=cleaned,
                    rejection_reason="Multiple alternative commands or options detected in model response",
                )

        # 4. Check for unexplained multi-line responses
        # Valid multiline: line continuation with trailing '\', or heredoc (<< EOF)
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if len(lines) > 1:
            is_valid_multiline = self._is_valid_multiline_script(cleaned)
            if not is_valid_multiline:
                return ValidationResult(
                    is_valid=False,
                    raw_extracted=cleaned,
                    rejection_reason=f"Unexplained multi-line response ({len(lines)} lines detected without valid continuation or heredoc)",
                )

        # Single normalized command string
        final_command = cleaned.strip()

        # 5. Output shape validation
        if not final_command:
            return ValidationResult(
                is_valid=False,
                rejection_reason="Empty command extracted after stripping formatting",
            )

        # Check for unbalanced quotes (shlex check)
        quote_valid, quote_err = self._check_quotes(final_command)
        if not quote_valid:
            return ValidationResult(
                is_valid=False,
                raw_extracted=final_command,
                rejection_reason=f"Syntax error: Unbalanced quotes detected ({quote_err})",
            )

        # 6. Safety risk assessment (adds non-fatal warnings)
        warnings = self._assess_safety_warnings(final_command)

        return ValidationResult(
            is_valid=True,
            command=final_command,
            raw_extracted=cleaned,
            warnings=warnings,
            rejection_reason=None,
        )

    def _extract_from_markdown(self, text: str) -> Tuple[str, Optional[str]]:
        """
        Extract code from markdown code fences (```bash ... ``` or ``` ... ```).
        Rejects responses with multiple disjoint code fences.
        """
        fence_matches = list(re.finditer(r"```(?:bash|sh)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE))

        if len(fence_matches) > 1:
            return "", "Multiple distinct markdown code blocks found; ambiguous command extraction"

        if len(fence_matches) == 1:
            extracted = fence_matches[0].group(1).strip()
            # If there is substantial text outside the code block, reject ambiguous commentary
            before = text[:fence_matches[0].start()].strip()
            after = text[fence_matches[0].end():].strip()
            # Allow minor introductory words like "Here is the command:" but reject paragraphs
            if len(before.split()) > 10 or len(after.split()) > 10:
                return "", "Excessive conversational text outside markdown code block"
            return extracted, None

        # Check for unclosed backticks
        if text.count("```") % 2 != 0:
            return "", "Malformed markdown: Unclosed code fence detected"

        # Check for inline single backticks wrapping whole command: `ls -la`
        if text.startswith("`") and text.endswith("`") and text.count("`") == 2:
            return text[1:-1].strip(), None

        return text, None

    def _strip_introductory_labels(self, text: str) -> str:
        """Strip simple label prefixes such as 'Command:', 'Bash:', or prompt symbols '$ '."""
        lines = text.splitlines()
        if not lines:
            return text

        first_line = lines[0].strip()
        label_patterns = [
            r"^(?:command|bash|sh|shell|output):\s*(.+)$",
            r"^\$\s*(.+)$",
            r"^#\s*(?:command|bash):\s*(.+)$",
        ]
        for pat in label_patterns:
            m = re.match(pat, first_line, re.IGNORECASE)
            if m:
                lines[0] = m.group(1).strip()
                break

        return "\n".join(lines).strip()

    def _is_valid_multiline_script(self, text: str) -> bool:
        """Check whether multiline text represents a legitimate chained command or heredoc."""
        lines = [l for l in text.splitlines() if l.strip()]
        # Check trailing backslashes for line continuation
        if all(l.rstrip().endswith("\\") for l in lines[:-1]):
            return True

        # Check heredoc: contains << EOF or similar delimiter
        if re.search(r"<<\s*['\"]?([A-Za-z0-9_]+)['\"]?", text):
            return True

        return False

    def _check_quotes(self, cmd: str) -> Tuple[bool, str]:
        """Validate that all single and double quotes are properly paired."""
        try:
            # shlex.split validates quote closures
            shlex.split(cmd)
            return True, ""
        except ValueError as exc:
            return False, str(exc)

    def _assess_safety_warnings(self, cmd: str) -> List[str]:
        """Inspect command for destructive, privileged, or network operations."""
        warnings: List[str] = []

        for pat, desc in self.DESTRUCTIVE_PATTERNS:
            if re.search(pat, cmd):
                warnings.append(f"Destructive operation: {desc}")

        for pat, desc in self.PRIVILEGED_PATTERNS:
            if re.search(pat, cmd):
                warnings.append(f"Privileged operation: {desc}")

        for pat, desc in self.NETWORK_PATTERNS:
            if re.search(pat, cmd):
                warnings.append(f"Network operation: {desc}")

        return warnings

