"""
Automated unit and regression tests for ALFA dataset preprocessing and cleaning.

Verifies:
- Raw dataset immutability and provenance
- Schema compliance of processed JSONL files
- Absence of documentation placeholders in clean splits
- Absence of train/val instruction leakage
- Unique record IDs
- Complete accounting of all raw records across clean, quarantined, and removed sets
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import pytest

RAW_TRAIN_FILE = Path("data/raw/alfa_train.csv")
RAW_TEST_FILE = Path("data/raw/alfa_test.csv")
RAW_METADATA_FILE = Path("data/raw/metadata.json")

PROCESSED_DIR = Path("data/processed")
TRAIN_FILE = PROCESSED_DIR / "train.jsonl"
VAL_FILE = PROCESSED_DIR / "val.jsonl"
QUARANTINE_FILE = PROCESSED_DIR / "quarantined.jsonl"
REMOVED_FILE = PROCESSED_DIR / "removed.jsonl"

REQUIRED_CLEAN_FIELDS = {
    "record_id",
    "source_row",
    "instruction",
    "command",
    "command_category",
    "source_dataset",
    "requires_network",
    "requires_privileges",
    "safety_level",
}

VALID_CATEGORIES = {
    "file_search",
    "text_processing",
    "file_management",
    "file_deletion",
    "permissions",
    "archive_compression",
    "process_management",
    "system_info",
    "network_remote",
    "shell_control",
    "utility_cli",
}

VALID_SAFETY_LEVELS = {
    "safe",
    "file_modify",
    "file_delete",
    "process_kill",
    "network_restricted",
    "privileged",
}


def test_raw_dataset_files_exist_and_unaltered():
    """Verify raw files exist with exact expected splits and headers."""
    assert RAW_TRAIN_FILE.exists(), "Raw training CSV missing!"
    assert RAW_TEST_FILE.exists(), "Raw test CSV missing!"
    assert RAW_METADATA_FILE.exists(), "Raw metadata JSON missing!"

    # Verify train raw split
    with open(RAW_TRAIN_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        train_rows = list(reader)
    assert header == ["nl", "bash"]
    assert len(train_rows) == 40639, f"Expected 40639 raw train rows, got {len(train_rows)}"

    # Verify official test set is exactly 300 tasks
    with open(RAW_TEST_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        test_rows = list(reader)
    assert header == ["nl", "bash", "bash2", "difficulty"]
    assert len(test_rows) == 300, f"Expected 300 official test rows, got {len(test_rows)}"

    # Verify metadata
    with open(RAW_METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["dataset_id"] == "westenfelder/NL2SH-ALFA"
    assert meta["revision"] == "a99cb5784cf5c2a42b1cc26c1903d9c3b35206ba"
    assert meta["license"] == "MIT"


def test_processed_files_exist():
    """Verify all 4 processed JSONL files exist."""
    assert TRAIN_FILE.exists(), "data/processed/train.jsonl missing!"
    assert VAL_FILE.exists(), "data/processed/val.jsonl missing!"
    assert QUARANTINE_FILE.exists(), "data/processed/quarantined.jsonl missing!"
    assert REMOVED_FILE.exists(), "data/processed/removed.jsonl missing!"


def test_complete_accounting_of_raw_records():
    """Verify that every single raw record is accounted for across clean, quarantined, and removed sets."""
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        n_train = sum(1 for _ in f)
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        n_val = sum(1 for _ in f)
    with open(QUARANTINE_FILE, "r", encoding="utf-8") as f:
        n_quarantine = sum(1 for _ in f)
    with open(REMOVED_FILE, "r", encoding="utf-8") as f:
        n_removed = sum(1 for _ in f)

    total_accounted = n_train + n_val + n_quarantine + n_removed
    assert total_accounted == 40639, f"Total accounted {total_accounted} != 40639 raw records!"
    assert n_train > 25000, f"Clean train size {n_train} lower than expected"
    assert n_val > 2500, f"Clean val size {n_val} lower than expected"


def test_schema_compliance_of_clean_splits():
    """Verify all clean records conform to the required schema and value types."""
    for filepath, split_name in [(TRAIN_FILE, "train"), (VAL_FILE, "val")]:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                rec = json.loads(line)
                missing = REQUIRED_CLEAN_FIELDS - set(rec.keys())
                assert not missing, f"{split_name} line {line_no} missing fields: {missing}"

                assert isinstance(rec["record_id"], str) and rec["record_id"].startswith(f"alfa_{split_name}_")
                assert isinstance(rec["source_row"], int) and rec["source_row"] >= 1
                assert isinstance(rec["instruction"], str) and rec["instruction"].strip()
                assert isinstance(rec["command"], str) and rec["command"].strip()
                assert rec["command_category"] in VALID_CATEGORIES, f"Invalid category: {rec['command_category']}"
                assert rec["source_dataset"] == "westenfelder/NL2SH-ALFA"
                assert isinstance(rec["requires_network"], bool)
                assert isinstance(rec["requires_privileges"], bool)
                assert rec["requires_privileges"] is False, "Clean split should not contain privileged commands!"
                assert rec["safety_level"] in VALID_SAFETY_LEVELS, f"Invalid safety level: {rec['safety_level']}"


def test_unique_record_ids():
    """Verify all record IDs in train and val are globally unique."""
    seen_ids = set()
    for filepath in [TRAIN_FILE, VAL_FILE]:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                rid = rec["record_id"]
                assert rid not in seen_ids, f"Duplicate record_id found: {rid}"
                seen_ids.add(rid)


def test_no_placeholder_leakage_in_clean_splits():
    """Verify clean records contain no documentation placeholders."""
    placeholder_patterns = [
        re.compile(r"path/to/", re.IGNORECASE),
        re.compile(r"<[^>]+>"),
        re.compile(r"(?:\s|^)\.\.\.(?:\s|$)"),
        re.compile(r"--?[a-zA-Z0-9_\-]+\|[a-zA-Z0-9_\-]+"),
    ]

    for filepath in [TRAIN_FILE, VAL_FILE]:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                rec = json.loads(line)
                cmd = rec["command"]
                for pat in placeholder_patterns:
                    assert not pat.search(cmd), f"{filepath.name} line {line_no} contains placeholder: '{cmd}'"


def test_no_prompt_leakage_between_train_and_val():
    """Verify there is zero instruction leakage between train and val."""
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_prompts = set(json.loads(line)["instruction"].strip().lower() for line in f)

    with open(VAL_FILE, "r", encoding="utf-8") as f:
        val_prompts = set(json.loads(line)["instruction"].strip().lower() for line in f)

    overlap = train_prompts.intersection(val_prompts)
    assert not overlap, f"Train and validation have {len(overlap)} overlapping instructions!"


def test_quarantined_and_removed_records_have_reasons():
    """Verify all quarantined and removed records record explicit reasons."""
    with open(QUARANTINE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            assert "quarantine_reason" in rec and rec["quarantine_reason"].strip()
            assert rec["record_id"].startswith("alfa_quarantine_")

    with open(REMOVED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            assert "removal_reason" in rec and rec["removal_reason"].strip()
            assert rec["record_id"].startswith("alfa_removed_")

