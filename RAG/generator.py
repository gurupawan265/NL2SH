"""
Provider-independent LLM command generation interface for NL2SH.
Defines BaseCommandGenerator, MockCommandGenerator for tests, and an optional Ollama provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

from .prompt_builder import PromptBuilder
from .retriever import RetrievalResult
from .validator import CommandValidator, ValidationResult


@dataclass
class GenerationResult:
    """Structured result returned by command generator providers and the RAG pipeline."""

    instruction: str
    command: str = ""
    raw_response: str = ""
    retrieved_doc_ids: List[str] = field(default_factory=list)
    provider_name: str = "unknown"
    model_name: str = "unknown"
    is_valid: bool = False
    warnings: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    is_clarification: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize generation result to dictionary."""
        return {
            "instruction": self.instruction,
            "command": self.command,
            "raw_response": self.raw_response,
            "retrieved_doc_ids": self.retrieved_doc_ids,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "is_valid": self.is_valid,
            "warnings": self.warnings,
            "rejection_reason": self.rejection_reason,
            "is_clarification": self.is_clarification,
        }


class BaseCommandGenerator(ABC):
    """Abstract base class for all LLM command generation providers."""

    def __init__(
        self,
        provider_name: str,
        model_name: str,
        prompt_builder: Optional[PromptBuilder] = None,
        validator: Optional[CommandValidator] = None,
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.validator = validator or CommandValidator()

    @abstractmethod
    def _call_model(self, prompt: str) -> str:
        """Call underlying provider with formatted prompt and return raw string response."""

    def generate(
        self,
        instruction: str,
        context_documents: List[RetrievalResult],
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        """
        Generate, extract, and validate a Bash command for natural-language instruction.
        """
        doc_ids = [d.doc_id for d in context_documents]

        # 1. Build prompt
        prompt = self.prompt_builder.build_full_prompt(
            instruction=instruction,
            context_documents=context_documents,
            execution_context=execution_context,
        )

        # 2. Call model
        try:
            raw_response = self._call_model(prompt)
        except Exception as exc:
            return GenerationResult(
                instruction=instruction,
                command="",
                raw_response="",
                retrieved_doc_ids=doc_ids,
                provider_name=self.provider_name,
                model_name=self.model_name,
                is_valid=False,
                rejection_reason=f"Model invocation failed: {exc}",
            )

        # 3. Validate and extract command
        val: ValidationResult = self.validator.validate(raw_response)

        return GenerationResult(
            instruction=instruction,
            command=val.command,
            raw_response=raw_response,
            retrieved_doc_ids=doc_ids,
            provider_name=self.provider_name,
            model_name=self.model_name,
            is_valid=val.is_valid,
            warnings=val.warnings,
            rejection_reason=val.rejection_reason,
            is_clarification=val.is_clarification,
        )


class MockCommandGenerator(BaseCommandGenerator):
    """
    Deterministic mock provider for automated tests and offline verification.
    Requires zero network access, downloads, or external services.
    """

    def __init__(
        self,
        canned_responses: Optional[Dict[str, str]] = None,
        default_response: str = "ls -la",
        model_name: str = "mock-model-v1",
        prompt_builder: Optional[PromptBuilder] = None,
        validator: Optional[CommandValidator] = None,
    ) -> None:
        super().__init__(
            provider_name="mock",
            model_name=model_name,
            prompt_builder=prompt_builder,
            validator=validator,
        )
        self.canned_responses = canned_responses or {}
        self.default_response = default_response
        self.invocation_count = 0
        self.last_prompt = ""

    def _call_model(self, prompt: str) -> str:
        self.invocation_count += 1
        self.last_prompt = prompt

        # Check for specific instruction matches in canned responses
        for key, resp in self.canned_responses.items():
            if key.lower() in prompt.lower():
                return resp

        return self.default_response


class OllamaCommandGenerator(BaseCommandGenerator):
    """
    Optional provider connecting strictly to an already-running local Ollama daemon.
    Never pulls models, never starts local processes, and enforces strict connection timeouts.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5-coder:1.5b",
        base_url: str = "http://localhost:11434",
        timeout: float = 10.0,
        prompt_builder: Optional[PromptBuilder] = None,
        validator: Optional[CommandValidator] = None,
    ) -> None:
        super().__init__(
            provider_name="ollama",
            model_name=model_name,
            prompt_builder=prompt_builder,
            validator=validator,
        )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _call_model(self, prompt: str) -> str:
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Ollama returned HTTP error status {resp.status}")
                body = json.loads(resp.read().decode("utf-8"))
                return body.get("response", "").strip()
        except urllib.error.URLError as err:
            raise ConnectionError(
                f"Cannot connect to local Ollama service at {endpoint}: {err.reason}. "
                "Ensure Ollama is running or use MockCommandGenerator."
            ) from err
        except Exception as exc:
            raise RuntimeError(f"Ollama request error: {exc}") from exc

