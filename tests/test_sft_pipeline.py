"""
Unit tests for NL2SH Qwen Supervised Fine-Tuning (SFT) pipeline.
Tests configuration, dataset loading, chat template formatting, adapter loading interface,
and evaluation comparison framework without downloading heavy model weights.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import MagicMock
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from RAG.generator import GenerationResult, QwenSFTCommandGenerator
from RAG.retriever import RetrievalResult
from Scripts.compare_evaluation_models import (
    format_comparison_table,
    load_evaluation_summary,
)
from Training.config import SFTTrainingConfig
from Training.dataset import (
    DEFAULT_SYSTEM_PROMPT,
    format_chat_messages,
    format_qwen_text,
    get_dataset_statistics,
    load_jsonl_dataset,
    prepare_hf_dataset,
    validate_sft_record,
)
from app import init_rag_pipeline


def test_sft_config_defaults_and_yaml_roundtrip(tmp_path):
    """Test SFTTrainingConfig defaults and YAML serialization roundtrip."""
    cfg = SFTTrainingConfig()
    assert cfg.model_name_or_path == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert cfg.max_seq_length == 384
    assert cfg.response_template == "<|im_start|>assistant\n"
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert "q_proj" in cfg.lora_target_modules
    assert cfg.load_in_4bit is True
    assert cfg.bf16 is True

    # Memory estimation
    mem = cfg.estimate_vram_requirement()
    assert "total_estimated_vram" in mem
    assert "T4" in mem["recommended_hardware"]

    # YAML roundtrip
    yaml_file = tmp_path / "test_sft_config.yaml"
    cfg.to_yaml(yaml_file)
    assert yaml_file.exists()

    loaded_cfg = SFTTrainingConfig.from_yaml(yaml_file)
    assert loaded_cfg.model_name_or_path == cfg.model_name_or_path
    assert loaded_cfg.learning_rate == cfg.learning_rate
    assert loaded_cfg.lora_target_modules == cfg.lora_target_modules


def test_sft_dataset_validation():
    """Test validation of SFT dataset records."""
    valid_rec = {"instruction": "List files", "command": "ls -l"}
    ok, err = validate_sft_record(valid_rec)
    assert ok is True
    assert err is None

    empty_inst = {"instruction": "  ", "command": "ls -l"}
    ok, err = validate_sft_record(empty_inst)
    assert ok is False
    assert "empty or missing 'instruction'" in err

    empty_cmd = {"instruction": "List files", "command": " "}
    ok, err = validate_sft_record(empty_cmd)
    assert ok is False
    assert "empty or missing 'command'" in err


def test_sft_dataset_rejects_alfa_test_csv():
    """Security check: Loading official alfa_test.csv must be strictly rejected."""
    with pytest.raises(PermissionError) as excinfo:
        load_jsonl_dataset("data/raw/alfa_test.csv")
    assert "Security Violation" in str(excinfo.value)
    assert "alfa_test.csv" in str(excinfo.value)


def test_sft_dataset_counts_and_zero_leakage():
    """Verify clean dataset counts and confirm zero overlap between train and validation."""
    stats = get_dataset_statistics("data/processed/train.jsonl", "data/processed/val.jsonl")
    assert stats["train_count"] == 28167
    assert stats["val_count"] == 3129
    assert stats["total_clean_samples"] == 31296
    # 100% strict zero leakage between train and val
    assert stats["train_val_overlap_count"] == 0


def test_chat_template_formatting():
    """Test ChatML formatting for Qwen instruction tuning."""
    inst = "Find all empty files in /workspace/project"
    cmd = "find /workspace/project -type f -empty"

    # Construction of messages
    msgs = format_chat_messages(inst, cmd)
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == inst
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["content"] == cmd

    # ChatML Text Rendering
    text = format_qwen_text(inst, cmd)
    assert "<|im_start|>system" in text
    assert "<|im_start|>user\nFind all empty files in /workspace/project<|im_end|>" in text
    assert f"<|im_start|>assistant\n{cmd}<|im_end|>" in text
    # No markdown code blocks in command
    assert "```" not in text


def test_prepare_hf_dataset_subset():
    """Test loading and preparing small subset of train and val datasets."""
    train_ds, val_ds = prepare_hf_dataset(
        train_file="data/processed/train.jsonl",
        val_file="data/processed/val.jsonl",
        max_train_samples=5,
        max_val_samples=3,
    )
    assert len(train_ds) == 5
    assert len(val_ds) == 3
    assert "text" in train_ds.column_names
    assert "instruction" in train_ds.column_names
    assert "command" in train_ds.column_names
    assert "<|im_start|>assistant" in train_ds[0]["text"]


def test_qwen_sft_generator_lazy_initialization():
    """Test that QwenSFTCommandGenerator initializes lazily without downloading weights."""
    gen = QwenSFTCommandGenerator(
        adapter_path="adapters/test_adapter",
        base_model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
    )
    assert gen.provider_name == "qwen-sft"
    assert gen.model_name == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert gen._model is None  # Weights are not loaded in __init__
    assert gen._tokenizer is None


def test_qwen_sft_generator_mocked_inference():
    """Test inference output formatting and validation with a mocked Qwen model."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Setup tokenizer mock
    import torch
    dummy_input_ids = torch.tensor([[101, 102, 103]])
    mock_tokenizer.return_value = {"input_ids": dummy_input_ids}
    mock_tokenizer.pad_token_id = 0
    mock_model.device = "cpu"

    # Setup model generate output mock
    dummy_output_ids = torch.tensor([[101, 102, 103, 201, 202]])
    mock_model.generate.return_value = dummy_output_ids

    # Decode mock returns pure bash command
    mock_tokenizer.decode.return_value = "grep -rn 'ERROR' /var/log/"

    gen = QwenSFTCommandGenerator(
        base_model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
        model=mock_model,
        tokenizer=mock_tokenizer,
    )

    ctx = [
        RetrievalResult(
            doc_id="doc_grep",
            command="grep",
            title="Search text",
            section="Inspection",
            content="Searches for patterns.",
            examples=["grep -rn pattern dir/"],
            score=0.92,
            command_category="text_search",
            safety_level="safe",
            metadata={"source_name": "Linux Coreutils"},
        )
    ]

    res = gen.generate("Search for ERROR in logs", ctx)
    assert isinstance(res, GenerationResult)
    assert res.is_valid is True
    assert res.command == "grep -rn 'ERROR' /var/log/"
    assert res.provider_name == "qwen-sft"
    assert res.retrieved_doc_ids == ["doc_grep"]
    assert res.token_usage is not None
    assert res.token_usage["prompt_tokens"] == 3
    assert res.token_usage["completion_tokens"] == 2


def test_app_init_rag_pipeline_qwen_sft(tmp_path):
    """Test that app.init_rag_pipeline supports provider='qwen-sft'."""
    from RAG.retriever import BashRetriever
    from RAG.vector_store import FAISSVectorStore

    mock_store = MagicMock()
    mock_emb = MagicMock()

    pipeline = init_rag_pipeline(
        provider="qwen-sft",
        adapter_path="adapters/qwen_adapter",
        vector_store=mock_store,
        embedder=mock_emb,
    )
    assert pipeline.generator.provider_name == "qwen-sft"
    assert pipeline.generator.adapter_path == "adapters/qwen_adapter"


def test_compare_evaluation_models_baseline_summary():
    """Test loading and summarizing existing Groq baseline evaluation results."""
    summary = load_evaluation_summary("data/evaluation/groq_evaluation_results.jsonl")
    assert summary is not None
    assert summary["total_tasks"] == 100
    assert summary["semantic_pass_rate"] == 92.0
    assert summary["semantic_pass_count"] == 92
    assert summary["exact_match_rate"] == 79.0

    table = format_comparison_table([
        {"label": "1. Groq Baseline", "architecture": "LPU API + RAG", **summary}
    ])
    assert "92.0%" in table
    assert "79.0%" in table
    assert "Groq Baseline" in table
