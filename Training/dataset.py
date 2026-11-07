"""
Dataset loading, validation, and chat-template formatting for NL2SH SFT.
Ensures zero data leakage from official test set and formats pure Bash training examples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert Linux shell assistant. Your task is to translate natural language "
    "instructions into exact, executable Bash commands. Output only the command with no markdown, "
    "explanation, or preamble."
)


def validate_sft_record(record: Dict[str, Any], record_idx: int = 0) -> Tuple[bool, Optional[str]]:
    """
    Validate that an SFT record contains non-empty natural language instruction and Bash command.
    """
    if not isinstance(record, dict):
        return False, f"Record {record_idx} is not a valid JSON object"

    inst = record.get("instruction")
    if not inst or not isinstance(inst, str) or not inst.strip():
        return False, f"Record {record_idx} has empty or missing 'instruction'"

    cmd = record.get("command")
    if not cmd or not isinstance(cmd, str) or not cmd.strip():
        return False, f"Record {record_idx} has empty or missing 'command'"

    return True, None


def load_jsonl_dataset(
    file_path: Union[str, Path],
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load and validate records from a JSONL file.
    Rejects any attempt to load test benchmark files to protect evaluation integrity.
    """
    p = Path(file_path).resolve()

    # Safety check: Prevent loading raw official test sets
    if "alfa_test" in p.name.lower():
        raise PermissionError(
            f"Security Violation: Attempted to load official holdout test set '{p.name}'. "
            "alfa_test.csv is strictly reserved for final evaluation and must never be loaded for training!"
        )

    if not p.is_file():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    records: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Malformed JSON on line {idx} of {p}: {err}") from err

            is_valid, reason = validate_sft_record(row, idx)
            if not is_valid:
                raise ValueError(f"Invalid record in {p}: {reason}")

            records.append(row)
            if max_samples and len(records) >= max_samples:
                break

    return records


def format_chat_messages(
    instruction: str,
    command: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    """
    Construct chat message list compliant with Qwen2.5 chat template.
    If command is None, format for inference generation prompt.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction.strip()},
    ]
    if command is not None:
        messages.append({"role": "assistant", "content": command.strip()})
    return messages


def format_qwen_text(
    instruction: str,
    command: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    add_generation_prompt: bool = False,
) -> str:
    """
    Render exact Qwen2.5 ChatML format text without requiring transformers tokenizer.
    <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{command}<|im_end|>
    """
    parts = [
        f"<|im_start|>system\n{system_prompt}<|im_end|>",
        f"<|im_start|>user\n{instruction.strip()}<|im_end|>",
    ]
    if command is not None:
        parts.append(f"<|im_start|>assistant\n{command.strip()}<|im_end|>")
    elif add_generation_prompt:
        parts.append("<|im_start|>assistant\n")

    return "\n".join(parts)


def prepare_hf_dataset(
    train_file: Union[str, Path] = "data/processed/train.jsonl",
    val_file: Union[str, Path] = "data/processed/val.jsonl",
    tokenizer: Optional[Any] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Tuple[Any, Any]:
    """
    Load train and validation datasets and format them into Hugging Face Dataset objects
    with a 'text' field containing the complete ChatML formatted conversation.
    """
    from datasets import Dataset

    raw_train = load_jsonl_dataset(train_file, max_samples=max_train_samples)
    raw_val = load_jsonl_dataset(val_file, max_samples=max_val_samples)

    def _to_text_rows(records: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        texts = []
        instructions = []
        commands = []
        record_ids = []

        for r in records:
            inst = r["instruction"]
            cmd = r["command"]
            rid = r.get("record_id", "")

            if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
                messages = format_chat_messages(inst, cmd, system_prompt=system_prompt)
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                text = format_qwen_text(inst, cmd, system_prompt=system_prompt)

            texts.append(text)
            instructions.append(inst)
            commands.append(cmd)
            record_ids.append(rid)

        return {
            "text": texts,
            "instruction": instructions,
            "command": commands,
            "record_id": record_ids,
        }

    train_data = _to_text_rows(raw_train)
    val_data = _to_text_rows(raw_val)

    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)

    return train_dataset, val_dataset


def get_dataset_statistics(
    train_file: Union[str, Path] = "data/processed/train.jsonl",
    val_file: Union[str, Path] = "data/processed/val.jsonl",
) -> Dict[str, Any]:
    """
    Compute dataset audit statistics: counts, lengths, unique instructions, and validation.
    """
    train_records = load_jsonl_dataset(train_file)
    val_records = load_jsonl_dataset(val_file)

    train_instructions = {r["instruction"].strip() for r in train_records}
    val_instructions = {r["instruction"].strip() for r in val_records}
    overlap = train_instructions.intersection(val_instructions)

    inst_lengths = [len(r["instruction"].split()) for r in train_records]
    cmd_lengths = [len(r["command"].split()) for r in train_records]

    return {
        "train_count": len(train_records),
        "val_count": len(val_records),
        "total_clean_samples": len(train_records) + len(val_records),
        "unique_train_instructions": len(train_instructions),
        "unique_val_instructions": len(val_instructions),
        "train_val_overlap_count": len(overlap),
        "avg_instruction_word_length": round(sum(inst_lengths) / len(inst_lengths), 2),
        "max_instruction_word_length": max(inst_lengths),
        "avg_command_word_length": round(sum(cmd_lengths) / len(cmd_lengths), 2),
        "max_command_word_length": max(cmd_lengths),
    }
