"""
Unit tests for NL2SH RAG corpus schema, validation, and loading.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RAG.corpus import (
    REQUIRED_FIELDS,
    CorpusDocument,
    load_corpus,
    validate_corpus_record,
)


@pytest.fixture
def sample_valid_record():
    return {
        "doc_id": "test_ls_001",
        "command": "ls",
        "title": "List directory contents",
        "section": "File Listing",
        "content": "List files with options.",
        "examples": ["ls -l", "ls -a"],
        "source_url": "https://example.com/ls",
        "source_name": "Test Manual",
        "license": "MIT",
        "command_category": "navigation",
        "safety_level": "safe",
    }


def test_validate_valid_corpus_record(sample_valid_record):
    """Test validation succeeds on schema-compliant dictionary."""
    is_valid, err = validate_corpus_record(sample_valid_record)
    assert is_valid is True
    assert err == ""


def test_missing_required_fields(sample_valid_record):
    """Test validation fails when any required field is missing."""
    for field in REQUIRED_FIELDS:
        incomplete = dict(sample_valid_record)
        del incomplete[field]
        is_valid, err = validate_corpus_record(incomplete)
        assert is_valid is False
        assert f"Missing required field: '{field}'" in err


def test_empty_string_fields(sample_valid_record):
    """Test validation fails when a string field is blank or whitespace."""
    for field in REQUIRED_FIELDS:
        if field == "examples":
            continue
        invalid = dict(sample_valid_record)
        invalid[field] = "   "
        is_valid, err = validate_corpus_record(invalid)
        assert is_valid is False
        assert f"Field '{field}' must be a non-empty string" in err


def test_invalid_examples_type(sample_valid_record):
    """Test validation fails when examples is not a list of non-empty strings."""
    # Not a list
    invalid1 = dict(sample_valid_record)
    invalid1["examples"] = "ls -l"
    is_valid1, err1 = validate_corpus_record(invalid1)
    assert is_valid1 is False
    assert "must be a list of strings" in err1

    # List with empty string
    invalid2 = dict(sample_valid_record)
    invalid2["examples"] = ["ls -l", ""]
    is_valid2, err2 = validate_corpus_record(invalid2)
    assert is_valid2 is False
    assert "must be non-empty strings" in err2


def test_load_corpus_from_valid_file(sample_valid_record):
    """Test loading and parsing records from a valid JSONL file."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".jsonl") as f:
        f.write(json.dumps(sample_valid_record) + "\n")
        tmp_path = f.name

    try:
        docs = load_corpus(tmp_path)
        assert len(docs) == 1
        doc = docs[0]
        assert isinstance(doc, CorpusDocument)
        assert doc.doc_id == "test_ls_001"
        assert doc.command == "ls"
        assert doc.examples == ["ls -l", "ls -a"]
        assert "Command: ls" in doc.to_search_text()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_load_corpus_malformed_json():
    """Test loading fails with ValueError on malformed JSON line."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".jsonl") as f:
        f.write('{"doc_id": "bad", broken json...\n')
        tmp_path = f.name

    try:
        with pytest.raises(ValueError, match="Malformed JSON"):
            load_corpus(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_load_corpus_duplicate_doc_id(sample_valid_record):
    """Test loading rejects duplicate doc_id entries."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".jsonl") as f:
        f.write(json.dumps(sample_valid_record) + "\n")
        f.write(json.dumps(sample_valid_record) + "\n")
        tmp_path = f.name

    try:
        with pytest.raises(ValueError, match="Duplicate doc_id"):
            load_corpus(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_official_corpus_file_validity():
    """Verify that the project's data/corpus/bash_reference.jsonl is valid and loads cleanly."""
    corpus_path = Path(__file__).resolve().parent.parent / "data" / "corpus" / "bash_reference.jsonl"
    assert corpus_path.is_file(), f"Official corpus file missing at {corpus_path}"

    docs = load_corpus(corpus_path)
    assert len(docs) >= 50
    assert all(isinstance(d, CorpusDocument) for d in docs)

