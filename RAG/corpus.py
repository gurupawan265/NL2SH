"""
Corpus document schema, validation, and loading for NL2SH RAG.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


REQUIRED_FIELDS = [
    "doc_id",
    "command",
    "title",
    "section",
    "content",
    "examples",
    "source_url",
    "source_name",
    "license",
    "command_category",
    "safety_level",
]


@dataclass
class CorpusDocument:
    """Represents a validated documentation chunk in the Bash corpus."""

    doc_id: str
    command: str
    title: str
    section: str
    content: str
    examples: List[str]
    source_url: str
    source_name: str
    license: str
    command_category: str
    safety_level: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return asdict(self)

    def to_search_text(self) -> str:
        """
        Format document into a rich, representative text representation
        optimized for dense embedding and semantic retrieval.
        """
        example_str = " ; ".join(self.examples) if self.examples else "None"
        return (
            f"Command: {self.command}\n"
            f"Title: {self.title}\n"
            f"Category: {self.command_category}\n"
            f"Description: {self.content}\n"
            f"Examples: {example_str}"
        )


def validate_corpus_record(record: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate a single corpus dictionary against required schema rules.
    Returns (is_valid, error_message).
    """
    if not isinstance(record, dict):
        return False, "Record is not a dictionary"

    for field in REQUIRED_FIELDS:
        if field not in record:
            return False, f"Missing required field: '{field}'"
        val = record[field]
        if field == "examples":
            if not isinstance(val, list):
                return False, f"Field 'examples' must be a list of strings, got {type(val).__name__}"
            if not all(isinstance(x, str) and x.strip() for x in val):
                return False, "All elements in 'examples' must be non-empty strings"
        else:
            if not isinstance(val, str) or not val.strip():
                return False, f"Field '{field}' must be a non-empty string"

    return True, ""


def load_corpus(file_path: str | Path) -> List[CorpusDocument]:
    """
    Load, validate, and parse a JSONL corpus file.
    Raises ValueError or FileNotFoundError if the file is missing or invalid.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file does not exist: {path}")

    documents: List[CorpusDocument] = []
    seen_ids = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {line_num} in {path}: {exc}") from exc

            is_valid, err_msg = validate_corpus_record(record)
            if not is_valid:
                raise ValueError(f"Schema validation failure on line {line_num}: {err_msg}")

            doc_id = record["doc_id"]
            if doc_id in seen_ids:
                raise ValueError(f"Duplicate doc_id '{doc_id}' detected on line {line_num}")
            seen_ids.add(doc_id)

            doc = CorpusDocument(
                doc_id=doc_id,
                command=record["command"],
                title=record["title"],
                section=record["section"],
                content=record["content"],
                examples=record["examples"],
                source_url=record["source_url"],
                source_name=record["source_name"],
                license=record["license"],
                command_category=record["command_category"],
                safety_level=record["safety_level"],
            )
            documents.append(doc)

    if not documents:
        raise ValueError(f"Corpus file '{path}' is empty or contains no valid documents")

    return documents

