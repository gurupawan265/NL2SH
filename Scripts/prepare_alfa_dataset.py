"""
Preprocessing and Cleaning Pipeline for NL2SH-ALFA.

Loads raw ALFA dataset, validates schema, applies strict filtering and quarantine
rules, categorizes commands, normalizes records to JSONL schema, and creates
leak-free train and validation splits.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import random
import re
import shlex
from typing import Any, Dict, List, Set, Tuple


# Specialized proprietary or distro-specific CLIs not standard in Ubuntu / Debian
UNAVAILABLE_TOOLS: Set[str] = {
    "adscript", "b2", "bw", "distrobox-export", "gam", "loc", "pvecm",
    "qm", "samtools", "urpme", "urpmi", "urpmi.update", "zfs", "zpool"
}

# Network-dependent utilities and patterns
NETWORK_PATTERN = re.compile(
    r"\b(curl|wget|ping|ssh|scp|sftp|ftp|nc|netcat|nmap|traceroute|nslookup|dig|"
    r"git\s+clone|git\s+pull|git\s+push|git\s+fetch|docker\s+pull|"
    r"pip\s+install|npm\s+install|cargo\s+install|apt|apt-get|yum|dnf|pacman|brew)\b",
    re.IGNORECASE
)

# Privileged command patterns
PRIVILEGED_PATTERN = re.compile(
    r"(?:^\s*|\b|;|\&|\|)(?:sudo|su)\s+",
    re.IGNORECASE
)

# Destructive command patterns
DESTRUCTIVE_PATTERN = re.compile(
    r"\b(rm|rmdir|shred|mkfs|dd|fdisk|parted|reboot|shutdown|poweroff|init|kill|killall|pkill)\b",
    re.IGNORECASE
)

# File modifying patterns
FILE_MODIFY_PATTERN = re.compile(
    r"\b(touch|mkdir|cp|mv|chmod|chown|sed\s+-i)\b|(?<![<>&])>(?![>])|>>",
    re.IGNORECASE
)


def extract_primary_binary(command: str) -> str:
    """Extract the primary executable binary from a shell command string."""
    c = re.sub(r"^\s*sudo\s+", "", command.strip(), flags=re.IGNORECASE)
    tokens = re.split(r"[\s|;&]+", c)
    return tokens[0].lower() if tokens and tokens[0] else ""


def determine_command_category(command: str) -> str:
    """Classify the command into a functional category."""
    binary = extract_primary_binary(command)

    if binary in {"find", "locate", "which", "whereis", "fd"}:
        return "file_search"
    elif binary in {"grep", "awk", "sed", "cut", "sort", "uniq", "tr", "wc", "cat", "head", "tail",
                    "diff", "cmp", "tee", "tac", "nl", "fold", "paste", "column", "comm", "rev",
                    "join", "expand", "unexpand"}:
        return "text_processing"
    elif binary in {"ls", "cd", "pwd", "mkdir", "rmdir", "touch", "cp", "mv", "ln", "readlink",
                    "basename", "dirname", "stat", "file", "tree"}:
        return "file_management"
    elif binary in {"rm", "shred", "unlink"}:
        return "file_deletion"
    elif binary in {"chmod", "chown", "chgrp", "umask"}:
        return "permissions"
    elif binary in {"tar", "gzip", "gunzip", "bzip2", "bunzip2", "xz", "unxz", "zip", "unzip", "7z", "cpio", "zcat"}:
        return "archive_compression"
    elif binary in {"ps", "top", "htop", "kill", "killall", "pkill", "pgrep", "nohup", "nice", "renice", "jobs", "fg", "bg"}:
        return "process_management"
    elif binary in {"df", "du", "free", "uptime", "uname", "who", "whoami", "id", "w", "last",
                    "date", "cal", "env", "printenv", "export", "set", "alias", "history", "echo", "printf"}:
        return "system_info"
    elif binary in {"curl", "wget", "ping", "ssh", "scp", "sftp", "rsync", "nc", "netcat", "nmap",
                    "traceroute", "nslookup", "dig", "host", "ip", "ifconfig", "netstat", "route", "git"}:
        return "network_remote"
    elif binary in {"sh", "bash", "source", "exec", "eval", "exit", "test", "sleep", "xargs"}:
        return "shell_control"
    else:
        return "utility_cli"


def determine_safety_level(command: str, requires_network: bool, requires_privileges: bool) -> str:
    """Determine the safety and operational constraint level of the command."""
    if requires_privileges:
        return "privileged"
    if re.search(r"\b(rm|rmdir|shred)\b", command, re.IGNORECASE):
        return "file_delete"
    if re.search(r"\b(kill|killall|pkill)\b", command, re.IGNORECASE):
        return "process_kill"
    if requires_network:
        return "network_restricted"
    if FILE_MODIFY_PATTERN.search(command):
        return "file_modify"
    return "safe"


def abstract_task_pattern(instruction: str, command: str) -> str:
    """Abstract instruction and command structure to group related task patterns."""
    binary = extract_primary_binary(command)
    # Normalize words in instruction
    words = [w for w in re.sub(r"[^\w\s]", "", instruction.lower()).split() if len(w) > 3]
    key_words = "_".join(words[:4])
    return f"{binary}:{key_words}"


def process_dataset(
    raw_train_csv: str = "data/raw/alfa_train.csv",
    output_dir: str = "data/processed",
    val_ratio: float = 0.10,
    random_seed: int = 42
) -> Dict[str, Any]:
    """Execute the full cleaning and normalization pipeline."""
    random.seed(random_seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(raw_train_csv):
        raise FileNotFoundError(f"Raw ALFA training file not found at: {raw_train_csv}")

    with open(raw_train_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        raw_rows = list(reader)

    if header != ["nl", "bash"]:
        raise ValueError(f"Unexpected header format in raw file: {header}. Expected ['nl', 'bash']")

    total_records = len(raw_rows)

    # First pass: conflict and duplicate detection
    inst_to_rows: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
    for idx, (nl, cmd) in enumerate(raw_rows):
        inst_to_rows[nl.strip().lower()].append((idx, nl, cmd))

    conflicting_instructions: Set[str] = set()
    for inst, occurrences in inst_to_rows.items():
        if len(occurrences) > 1:
            cmds = set(c.strip() for _, _, c in occurrences)
            if len(cmds) > 1:
                first_bins = set(extract_primary_binary(c) for c in cmds)
                if len(first_bins) > 1:
                    conflicting_instructions.add(inst)

    removed_records: List[Dict[str, Any]] = []
    quarantined_records: List[Dict[str, Any]] = []
    clean_records: List[Dict[str, Any]] = []

    seen_pairs: Set[Tuple[str, str]] = set()

    # Track template frequency for find capping
    find_sig_counts: Counter[str] = Counter()

    for idx, (nl_raw, cmd_raw) in enumerate(raw_rows):
        nl = nl_raw.strip()
        cmd = cmd_raw.strip()
        source_row_id = idx + 1

        # 1. Empty or whitespace
        if not nl or not cmd:
            removed_records.append({
                "record_id": f"alfa_removed_{len(removed_records)+1:05d}",
                "source_row": source_row_id,
                "instruction": nl,
                "command": cmd,
                "removal_reason": "empty_instruction_or_command"
            })
            continue

        # 2. Degenerate or gibberish instruction (<8 chars and not standard command)
        if len(nl) < 8 and nl.lower() not in {"ls", "pwd", "top", "who", "date", "cal", "df", "du", "ps", "id"}:
            removed_records.append({
                "record_id": f"alfa_removed_{len(removed_records)+1:05d}",
                "source_row": source_row_id,
                "instruction": nl,
                "command": cmd,
                "removal_reason": "degenerate_instruction"
            })
            continue

        # 3. Shell syntax validity
        try:
            shlex.split(cmd)
        except Exception as exc:
            removed_records.append({
                "record_id": f"alfa_removed_{len(removed_records)+1:05d}",
                "source_row": source_row_id,
                "instruction": nl,
                "command": cmd,
                "removal_reason": f"syntax_error: {exc}"
            })
            continue

        # 4. Documentation placeholders:
        # a) path/to/
        if re.search(r"path/to/", cmd, re.IGNORECASE):
            removed_records.append({
                "record_id": f"alfa_removed_{len(removed_records)+1:05d}",
                "source_row": source_row_id,
                "instruction": nl,
                "command": cmd,
                "removal_reason": "documentation_placeholder_path_to"
            })
            continue

        # b) <...> (documentation angle bracket placeholders)
        if re.search(r"<[^>]+>", cmd):
            removed_records.append({
                "record_id": f"alfa_removed_{len(removed_records)+1:05d}",
                "source_row": source_row_id,
                "instruction": nl,
                "command": cmd,
                "removal_reason": "documentation_placeholder_angle_brackets"
            })
            continue

        # c) standalone ellipsis (...)
        if re.search(r"(?:\s|^)\.\.\.(?:\s|$)", cmd):
            removed_records.append({
                "record_id": f"alfa_removed_{len(removed_records)+1:05d}",
                "source_row": source_row_id,
                "instruction": nl,
                "command": cmd,
                "removal_reason": "documentation_placeholder_ellipsis"
            })
            continue

        # d) documentation pipe in option syntax: -s|--shell or optionA|optionB
        if re.search(r"--?[a-zA-Z0-9_\-]+\|[a-zA-Z0-9_\-]+", cmd):
            removed_records.append({
                "record_id": f"alfa_removed_{len(removed_records)+1:05d}",
                "source_row": source_row_id,
                "instruction": nl,
                "command": cmd,
                "removal_reason": "documentation_placeholder_option_pipe"
            })
            continue

        # 5. Quarantine: Concurrently conflicting instructions
        if nl.lower() in conflicting_instructions:
            quarantined_records.append({
                "record_id": f"alfa_quarantine_{len(quarantined_records)+1:05d}",
                "source_row": source_row_id,
                "instruction": nl,
                "command": cmd,
                "quarantine_reason": "conflicting_prompt_mapping_different_utilities"
            })
            continue

        # 6. Quarantine: Exact duplicate pair (nl, cmd)
        pair_key = (nl.lower(), cmd)
        if pair_key in seen_pairs:
            quarantined_records.append({
                "record_id": f"alfa_quarantine_{len(quarantined_records)+1:05d}",
                "source_row": source_row_id,
                "instruction": nl,
                "command": cmd,
                "quarantine_reason": "exact_duplicate_instruction_command_pair"
            })
            continue
        seen_pairs.add(pair_key)

        # 7. Quarantine: Privileged commands requiring root/sudo
        requires_privs = bool(PRIVILEGED_PATTERN.search(cmd))
        if requires_privs:
            quarantined_records.append({
                "record_id": f"alfa_quarantine_{len(quarantined_records)+1:05d}",
                "source_row": source_row_id,
                "instruction": nl,
                "command": cmd,
                "quarantine_reason": "privileged_command_requires_root"
            })
            continue

        # 8. Quarantine: Unavailable third-party tools
        first_bin = extract_primary_binary(cmd)
        if first_bin in UNAVAILABLE_TOOLS:
            quarantined_records.append({
                "record_id": f"alfa_quarantine_{len(quarantined_records)+1:05d}",
                "source_row": source_row_id,
                "instruction": nl,
                "command": cmd,
                "quarantine_reason": f"unavailable_third_party_tool_{first_bin}"
            })
            continue

        # 9. Find balancing: cap excessive near-duplicate template signatures to max 10
        if first_bin == "find":
            # Normalize signature
            c_sig = re.sub(r"\b[a-zA-Z0-9_\-\./]+\.[a-zA-Z0-9]+\b", "<FILE>", cmd)
            c_sig = re.sub(r'"[^"]*"', '"<STR>"', c_sig)
            c_sig = re.sub(r"'[^']*'", "'<STR>'", c_sig)
            c_sig = re.sub(r"\b\d+\b", "<NUM>", c_sig).strip()
            if find_sig_counts[c_sig] >= 10:
                quarantined_records.append({
                    "record_id": f"alfa_quarantine_{len(quarantined_records)+1:05d}",
                    "source_row": source_row_id,
                    "instruction": nl,
                    "command": cmd,
                    "quarantine_reason": "excessive_find_template_repetition"
                })
                continue
            find_sig_counts[c_sig] += 1

        # Record is clean and retained
        req_net = bool(NETWORK_PATTERN.search(cmd))
        category = determine_command_category(cmd)
        safety = determine_safety_level(cmd, req_net, False)

        clean_records.append({
            "source_row": source_row_id,
            "instruction": nl,
            "command": cmd,
            "command_category": category,
            "source_dataset": "westenfelder/NL2SH-ALFA",
            "requires_network": req_net,
            "requires_privileges": False,
            "safety_level": safety,
            "task_pattern": abstract_task_pattern(nl, cmd)
        })

    # Grouped Train / Validation Splitting to prevent leakage
    # Cluster records by task pattern
    pattern_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in clean_records:
        pattern_groups[rec["task_pattern"]].append(rec)

    group_keys = list(pattern_groups.keys())
    random.shuffle(group_keys)

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []

    target_val_count = int(len(clean_records) * val_ratio)
    current_val_count = 0

    for key in group_keys:
        group = pattern_groups[key]
        if current_val_count + len(group) <= target_val_count or current_val_count == 0:
            val_records.extend(group)
            current_val_count += len(group)
        else:
            train_records.extend(group)

    # Assign sequential record_ids
    for i, r in enumerate(train_records, start=1):
        r["record_id"] = f"alfa_train_{i:05d}"
        del r["task_pattern"]

    for i, r in enumerate(val_records, start=1):
        r["record_id"] = f"alfa_val_{i:05d}"
        del r["task_pattern"]

    # Write output JSONL files
    train_file = out_path / "train.jsonl"
    val_file = out_path / "val.jsonl"
    quarantine_file = out_path / "quarantined.jsonl"
    removed_file = out_path / "removed.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(quarantine_file, "w", encoding="utf-8") as f:
        for r in quarantined_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(removed_file, "w", encoding="utf-8") as f:
        for r in removed_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Command distribution analysis before and after
    raw_dist = Counter(extract_primary_binary(r[1]) for r in raw_rows)
    clean_dist = Counter(extract_primary_binary(r["command"]) for r in train_records + val_records)

    stats = {
        "total_raw": total_records,
        "clean_retained_total": len(train_records) + len(val_records),
        "train_size": len(train_records),
        "val_size": len(val_records),
        "quarantined_total": len(quarantined_records),
        "removed_total": len(removed_records),
        "removal_reasons": dict(Counter(r["removal_reason"] for r in removed_records)),
        "quarantine_reasons": dict(Counter(r["quarantine_reason"] for r in quarantined_records)),
        "raw_distribution_top10": raw_dist.most_common(10),
        "clean_distribution_top10": clean_dist.most_common(10)
    }

    return stats


if __name__ == "__main__":
    results = process_dataset()
    print("=" * 60)
    print("ALFA DATASET PREPROCESSING COMPLETE")
    print(f"Total Raw Records:        {results['total_raw']}")
    print(f"Clean Records Retained:   {results['clean_retained_total']}")
    print(f"  - Training split:       {results['train_size']}")
    print(f"  - Validation split:     {results['val_size']}")
    print(f"Quarantined Records:      {results['quarantined_total']}")
    print(f"Removed Records:          {results['removed_total']}")
    print("=" * 60)
