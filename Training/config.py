"""
SFT Training Configuration for NL2SH Qwen Fine-Tuning.
Provides strongly-typed hyperparameters, YAML serialization, and memory estimation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


@dataclass
class SFTTrainingConfig:
    """Supervised Fine-Tuning configuration for Qwen2.5-Coder models."""

    # Model & Data Paths
    model_name_or_path: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    train_file: str = "data/processed/train.jsonl"
    val_file: str = "data/processed/val.jsonl"
    output_dir: str = "checkpoints/qwen2.5-coder-3b-nl2sh-sft"
    adapter_output_dir: str = "adapters/qwen2.5-coder-3b-nl2sh-sft"

    # Context & Tokenizer Limits
    max_seq_length: int = 256
    padding_side: str = "right"

    # PEFT / LoRA Hyperparameters
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Batching & Compute
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01

    # Precision & Quantization
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"

    # Logging, Evaluation & Checkpoints
    logging_steps: int = 50
    eval_strategy: str = "steps"
    eval_steps: int = 200
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 2
    seed: int = 42
    report_to: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SFTTrainingConfig:
        """Create configuration instance from dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration as a YAML file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> SFTTrainingConfig:
        """Load configuration from a YAML file."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def estimate_vram_requirement(self) -> Dict[str, str]:
        """Estimate VRAM footprint under various hardware setups."""
        if self.load_in_4bit:
            return {
                "base_model_weights": "~2.1 GB (4-bit NF4 quantized)",
                "lora_trainable_params": "~64 MB (r=16, 7 target modules)",
                "activations_and_gradients": "~3.5 GB (batch_size=4, seq_len=256, grad_ckpt=True)",
                "optimizer_states": "~300 MB (paged_adamw_8bit)",
                "total_estimated_vram": "~6.0 - 7.5 GB VRAM",
                "recommended_hardware": "NVIDIA T4 (15-16 GB, free Google Colab), V100 (16 GB), L4 (24 GB)",
            }
        else:
            return {
                "base_model_weights": "~6.2 GB (16-bit bfloat16)",
                "lora_trainable_params": "~128 MB (r=16)",
                "activations_and_gradients": "~5.0 GB (batch_size=4, seq_len=256, grad_ckpt=True)",
                "optimizer_states": "~600 MB (AdamW 8-bit)",
                "total_estimated_vram": "~12.0 - 14.0 GB VRAM",
                "recommended_hardware": "NVIDIA A100 (40/80 GB), L4 (24 GB), or A10G (24 GB)",
            }
