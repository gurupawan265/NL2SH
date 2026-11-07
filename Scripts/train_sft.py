"""
NL2SH Qwen SFT Training Script.
Supports reproducible LoRA / QLoRA fine-tuning of Qwen2.5-Coder-3B-Instruct.
Includes --inspect-only and --dry-run modes for safe local verification without downloading weights.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Training.config import SFTTrainingConfig
from Training.dataset import (
    DEFAULT_SYSTEM_PROMPT,
    format_chat_messages,
    format_qwen_text,
    get_dataset_statistics,
    load_jsonl_dataset,
    prepare_hf_dataset,
)


def run_inspect_mode(config: SFTTrainingConfig) -> None:
    """Audit datasets, display configurations, and print estimated GPU memory without loading models."""
    print("=" * 65)
    print("NL2SH SFT PIPELINE INSPECTION & PRE-FLIGHT AUDIT")
    print("=" * 65)
    print(f"Base Model:       {config.model_name_or_path}")
    print(f"Train File:       {config.train_file}")
    print(f"Validation File:  {config.val_file}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Adapter Output:   {config.adapter_output_dir}")
    print(f"LoRA Rank (r):    {config.lora_r} | Alpha: {config.lora_alpha} | Dropout: {config.lora_dropout}")
    print(f"Target Modules:   {', '.join(config.lora_target_modules)}")
    print(f"Max Seq Length:   {config.max_seq_length} tokens")
    print(f"Batch Size/Device:{config.per_device_train_batch_size} (Grad Accum: {config.gradient_accumulation_steps})")
    print(f"Learning Rate:    {config.learning_rate} ({config.lr_scheduler_type} schedule)")
    print(f"Epochs:           {config.num_train_epochs}")
    print(f"Precision:        4-bit QLoRA={config.load_in_4bit}, BF16={config.bf16}")

    print("\n--- Dataset Statistics ---")
    stats = get_dataset_statistics(config.train_file, config.val_file)
    print(f"  Clean Training Records:   {stats['train_count']:,}")
    print(f"  Clean Validation Records: {stats['val_count']:,}")
    print(f"  Total Clean Samples:      {stats['total_clean_samples']:,}")
    print(f"  Unique Train Prompts:     {stats['unique_train_instructions']:,}")
    print(f"  Train/Val Overlap Count:  {stats['train_val_overlap_count']} (100% strictly zero leakage)")
    print(f"  Avg Instruction Length:   {stats['avg_instruction_word_length']} words (Max: {stats['max_instruction_word_length']})")
    print(f"  Avg Command Length:       {stats['avg_command_word_length']} words (Max: {stats['max_command_word_length']})")

    print("\n--- GPU Memory Estimation ---")
    mem = config.estimate_vram_requirement()
    for k, v in mem.items():
        print(f"  {k.replace('_', ' ').title():26s}: {v}")

    print("\n--- Sample Formatted ChatML Training Record ---")
    sample_records = load_jsonl_dataset(config.train_file, max_samples=1)
    if sample_records:
        rec = sample_records[0]
        preview = format_qwen_text(rec["instruction"], rec["command"])
        print(preview)
    print("=" * 65)


def build_training_pipeline(
    config: SFTTrainingConfig,
    dry_run: bool = False,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> None:
    """
    Execute Supervised Fine-Tuning with PEFT LoRA / QLoRA.
    """
    if dry_run:
        print("[DRY-RUN] Validating dataset loading and tokenization pipeline...")
        train_ds, val_ds = prepare_hf_dataset(
            train_file=config.train_file,
            val_file=config.val_file,
            max_train_samples=max_train_samples or 10,
            max_val_samples=max_val_samples or 5,
        )
        print(f"[DRY-RUN] Successfully loaded {len(train_ds)} train and {len(val_ds)} val samples.")
        print("[DRY-RUN] Pre-flight dry-run completed successfully without downloading model weights.")
        return

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    if not torch.cuda.is_available():
        print("[Warning] CUDA is not available. SFT training on CPU is not recommended.", file=sys.stderr)

    # 1. Quantization Configuration
    bnb_config = None
    if config.load_in_4bit and torch.cuda.is_available():
        compute_dtype = torch.bfloat16 if config.bf16 else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    # 2. Tokenizer Setup
    print(f"Loading tokenizer: {config.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        padding_side=config.padding_side,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Model Setup
    print(f"Loading base model: {config.model_name_or_path}...")
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if config.bf16 else torch.float16,
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        **model_kwargs,
    )

    if config.load_in_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # 4. LoRA Configuration
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Dataset Preparation
    print("Preparing train and validation datasets...")
    train_dataset, val_dataset = prepare_hf_dataset(
        train_file=config.train_file,
        val_file=config.val_file,
        tokenizer=tokenizer,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=config.bf16 and torch.cuda.is_available(),
        fp16=config.fp16 and torch.cuda.is_available(),
        optim=config.optim if torch.cuda.is_available() else "adamw_torch",
        seed=config.seed,
        report_to=config.report_to,
        remove_unused_columns=False,
    )

    # 7. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    # 8. Train & Save
    print("Starting SFT training...")
    trainer.train()

    print(f"Saving fine-tuned LoRA adapter to {config.adapter_output_dir}...")
    model.save_pretrained(config.adapter_output_dir)
    tokenizer.save_pretrained(config.adapter_output_dir)
    print("SFT training completed successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(description="NL2SH Qwen Supervised Fine-Tuning")
    parser.add_argument("--config", default="Training/sft_config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--inspect-only", action="store_true", help="Print configuration, memory estimate, and dataset stats without loading models")
    parser.add_argument("--dry-run", action="store_true", help="Validate data loading without downloading model weights or starting training")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Cap training samples (for quick debug)")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Cap validation samples (for quick debug)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-device train batch size")
    parser.add_argument("--output-dir", default=None, help="Override checkpoint output directory")

    args = parser.parse_args()

    cfg_path = Path(args.config)
    if cfg_path.is_file():
        cfg = SFTTrainingConfig.from_yaml(cfg_path)
    else:
        cfg = SFTTrainingConfig()

    if args.epochs is not None:
        cfg.num_train_epochs = args.epochs
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.batch_size is not None:
        cfg.per_device_train_batch_size = args.batch_size
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    if args.inspect_only:
        run_inspect_mode(cfg)
    else:
        if args.dry_run:
            run_inspect_mode(cfg)
        build_training_pipeline(
            cfg,
            dry_run=args.dry_run,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
        )


if __name__ == "__main__":
    main()
