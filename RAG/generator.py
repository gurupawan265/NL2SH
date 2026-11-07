"""
Provider-independent LLM command generation interface for NL2SH.
Defines BaseCommandGenerator, MockCommandGenerator for tests, and an optional Ollama provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

from .prompt_builder import PromptBuilder
from .retriever import RetrievalResult
from .validator import CommandValidator, ValidationResult


def load_env_file(env_path: Optional[Path] = None) -> None:
    """Load key-value pairs from .env without external dependencies or exposing secrets."""
    if env_path is None:
        candidates = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parent.parent / ".env",
        ]
        for cand in candidates:
            if cand.is_file():
                env_path = cand
                break
    if env_path and env_path.is_file():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass


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
    token_usage: Optional[Dict[str, int]] = None
    latency: Optional[float] = None

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
            "token_usage": self.token_usage,
            "latency": self.latency,
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
        self.last_token_usage: Optional[Dict[str, int]] = None
        self.last_latency: Optional[float] = None

    def _sanitize_error(self, msg: str) -> str:
        """Strip sensitive secrets or API keys from error messages."""
        msg = re.sub(r"gsk_[a-zA-Z0-9]{20,}", "[REDACTED_API_KEY]", msg)
        msg = re.sub(r"Bearer\s+[a-zA-Z0-9_.-]+", "Bearer [REDACTED]", msg)
        return msg

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
        t0 = time.perf_counter()
        try:
            raw_response = self._call_model(prompt)
            latency = getattr(self, "last_latency", None)
            if latency is None:
                latency = round(time.perf_counter() - t0, 4)
        except Exception as exc:
            clean_err = self._sanitize_error(str(exc))
            return GenerationResult(
                instruction=instruction,
                command="",
                raw_response="",
                retrieved_doc_ids=doc_ids,
                provider_name=self.provider_name,
                model_name=self.model_name,
                is_valid=False,
                rejection_reason=f"Model invocation failed: {clean_err}",
                latency=round(time.perf_counter() - t0, 4),
            )

        # 3. Validate and extract command
        val: ValidationResult = self.validator.validate(raw_response)

        token_usage = getattr(self, "last_token_usage", None)

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
            token_usage=token_usage,
            latency=latency,
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


class GroqCommandGenerator(BaseCommandGenerator):
    """
    Production LLM command generation provider powered by Groq API.
    Supports ultra-low-latency inference with code-capable models (e.g. qwen/qwen3.8-27b).
    Strictly redacts sensitive keys from errors and logs, and never falls back silently.
    """

    DEFAULT_MODEL = "qwen/qwen3.8-27b"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 15.0,
        prompt_builder: Optional[PromptBuilder] = None,
        validator: Optional[CommandValidator] = None,
        client: Optional[Any] = None,
    ) -> None:
        load_env_file()
        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key or not resolved_key.strip():
            raise ValueError(
                "GROQ_API_KEY is missing or empty. Please set GROQ_API_KEY in .env or the environment."
            )

        resolved_model = (
            model_name
            or os.environ.get("GROQ_MODEL")
            or self.DEFAULT_MODEL
        )

        super().__init__(
            provider_name="groq",
            model_name=resolved_model,
            prompt_builder=prompt_builder,
            validator=validator,
        )
        self.api_key = resolved_key.strip()
        self.timeout = timeout

        if client is not None:
            self._client = client
        else:
            try:
                import groq
                self._client = groq.Groq(api_key=self.api_key, timeout=self.timeout)
            except ImportError:
                self._client = None

    def _call_model(self, prompt: str) -> str:
        self.last_token_usage = None
        start_time = time.perf_counter()

        if self._client is not None:
            return self._call_via_sdk(prompt, start_time)
        return self._call_via_http(prompt, start_time)

    def _call_via_sdk(self, prompt: str, start_time: float) -> str:
        import groq

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            self.last_latency = round(time.perf_counter() - start_time, 4)

            if getattr(response, "usage", None):
                usage = response.usage
                usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else (usage if isinstance(usage, dict) else vars(usage))
                self.last_token_usage = {
                    "prompt_tokens": usage_dict.get("prompt_tokens", 0),
                    "completion_tokens": usage_dict.get("completion_tokens", 0),
                    "total_tokens": usage_dict.get("total_tokens", 0),
                }

            if not response.choices:
                raise RuntimeError("Groq API returned an empty choices list.")

            content = response.choices[0].message.content or ""
            return content.strip()

        except groq.AuthenticationError as err:
            clean_msg = self._sanitize_error(str(err))
            raise PermissionError(f"Groq API authentication failed: {clean_msg}") from err
        except groq.RateLimitError as err:
            clean_msg = self._sanitize_error(str(err))
            raise RuntimeError(f"Groq API rate limit reached: {clean_msg}") from err
        except groq.APITimeoutError as err:
            clean_msg = self._sanitize_error(str(err))
            raise TimeoutError(f"Groq API request timed out after {self.timeout}s: {clean_msg}") from err
        except groq.APIConnectionError as err:
            clean_msg = self._sanitize_error(str(err))
            raise ConnectionError(f"Groq API connection failed: {clean_msg}") from err
        except Exception as exc:
            clean_msg = self._sanitize_error(str(exc))
            raise RuntimeError(f"Groq API error: {clean_msg}") from exc

    def _call_via_http(self, prompt: str, start_time: float) -> str:
        endpoint = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "NL2SH/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Groq API returned HTTP status {resp.status}")
                body = json.loads(resp.read().decode("utf-8"))
                self.last_latency = round(time.perf_counter() - start_time, 4)

                usage = body.get("usage", {})
                if usage:
                    self.last_token_usage = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }

                choices = body.get("choices", [])
                if not choices:
                    raise RuntimeError("Groq API returned no choices in response.")
                content = choices[0].get("message", {}).get("content", "")
                return content.strip()
        except urllib.error.HTTPError as err:
            err_body = err.read().decode("utf-8", errors="replace")
            clean_body = self._sanitize_error(err_body)
            if err.code in (401, 403):
                raise PermissionError(f"Groq API authentication failed (HTTP {err.code}): {clean_body}") from err
            elif err.code == 429:
                raise RuntimeError(f"Groq API rate limit exceeded (HTTP 429): {clean_body}") from err
            else:
                raise RuntimeError(f"Groq API HTTP error {err.code}: {clean_body}") from err
        except urllib.error.URLError as err:
            clean_reason = self._sanitize_error(str(err.reason))
            raise ConnectionError(f"Failed to connect to Groq API endpoint: {clean_reason}") from err
        except Exception as exc:
            clean_msg = self._sanitize_error(str(exc))
            raise RuntimeError(f"Groq API request failed: {clean_msg}") from exc


