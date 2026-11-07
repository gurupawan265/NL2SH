"""
NL2SH Model Evaluation Comparison Framework.
Compares performance across:
1. Groq Baseline (LPU-hosted production API)
2. Fine-tuned Qwen SFT without RAG (Direct instruction-to-command generation)
3. Fine-tuned Qwen SFT with RAG (Documentation retrieval + SFT model)

Guarantees zero access to official holdout benchmark (data/raw/alfa_test.csv).
Uses only development benchmark (data/evaluation/evaluation_benchmark.jsonl).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_evaluation_summary(results_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load and aggregate evaluation metrics from a JSONL results file."""
    p = Path(results_path)
    if not p.is_file():
        return None

    records: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return None

    total = len(records)
    valid_count = sum(1 for r in records if r.get("is_valid"))
    executed_count = sum(1 for r in records if r.get("executed"))
    exec_success_count = sum(1 for r in records if r.get("execution_success"))
    semantic_pass_count = sum(1 for r in records if r.get("semantic_pass"))
    exact_match_count = sum(1 for r in records if r.get("exact_match"))
    timeouts = sum(1 for r in records if r.get("timed_out"))

    latencies = [r.get("generation_latency", 0.0) for r in records]
    avg_latency = sum(latencies) / total if total else 0.0

    total_tokens = sum((r.get("token_usage") or {}).get("total_tokens", 0) for r in records)
    avg_tokens = total_tokens / total if total else 0.0

    provider = records[0].get("provider", "unknown")
    model = records[0].get("model", "unknown")

    return {
        "file": str(p),
        "total_tasks": total,
        "provider": provider,
        "model": model,
        "valid_command_rate": round(valid_count / total * 100, 2),
        "execution_success_rate": round(exec_success_count / executed_count * 100, 2) if executed_count else 0.0,
        "semantic_pass_rate": round(semantic_pass_count / total * 100, 2),
        "semantic_pass_count": semantic_pass_count,
        "exact_match_rate": round(exact_match_count / total * 100, 2),
        "exact_match_count": exact_match_count,
        "timeout_count": timeouts,
        "avg_latency": round(avg_latency, 4),
        "avg_tokens": round(avg_tokens, 1),
    }


def format_comparison_table(models_data: List[Dict[str, Any]]) -> str:
    """Format comparative evaluation statistics into a clean markdown table."""
    headers = [
        "Metric",
        *(f"{m['label']}" for m in models_data),
    ]

    rows = [
        ("Architecture", *(f"{m.get('architecture', 'N/A')}" for m in models_data)),
        ("Model Identifier", *(f"`{m.get('model', 'N/A')}`" for m in models_data)),
        ("Tasks Evaluated", *(f"{m.get('total_tasks', 'N/A')}" for m in models_data)),
        ("Valid Command Rate", *(f"{m.get('valid_command_rate', 'N/A')}%" for m in models_data)),
        ("Execution Success Rate", *(f"{m.get('execution_success_rate', 'N/A')}%" for m in models_data)),
        ("**Semantic Pass Rate**", *(f"**{m.get('semantic_pass_rate', 'N/A')}%** ({m.get('semantic_pass_count', 0)}/{m.get('total_tasks', 0)})" for m in models_data)),
        ("Exact Match Rate", *(f"{m.get('exact_match_rate', 'N/A')}%" for m in models_data)),
        ("Timeouts", *(f"{m.get('timeout_count', 'N/A')}" for m in models_data)),
        ("Avg Generation Latency", *(f"{m.get('avg_latency', 'N/A')}s" for m in models_data)),
        ("Avg Tokens / Query", *(f"{m.get('avg_tokens', 'N/A')}" for m in models_data)),
    ]

    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(val)))

    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"

    table_lines = [header_line, sep_line]
    for row in rows:
        table_lines.append("| " + " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)) + " |")

    return "\n".join(table_lines)


def generate_comparison_report(
    groq_results_file: str = "data/evaluation/groq_evaluation_results.jsonl",
    qwen_norag_file: str = "data/evaluation/qwen_sft_norag_results.jsonl",
    qwen_rag_file: str = "data/evaluation/qwen_sft_rag_results.jsonl",
) -> str:
    """Load available results and generate full comparison report text."""
    models_to_compare = []

    # 1. Groq Baseline
    groq_stats = load_evaluation_summary(groq_results_file)
    if groq_stats:
        groq_stats["label"] = "1. Groq Baseline"
        groq_stats["architecture"] = "LPU API + RAG"
        models_to_compare.append(groq_stats)

    # 2. Qwen SFT No-RAG
    norag_stats = load_evaluation_summary(qwen_norag_file)
    if norag_stats:
        norag_stats["label"] = "2. Qwen SFT (No-RAG)"
        norag_stats["architecture"] = "Local LoRA Direct"
        models_to_compare.append(norag_stats)
    else:
        models_to_compare.append({
            "label": "2. Qwen SFT (No-RAG)",
            "architecture": "Local LoRA Direct",
            "model": "Qwen2.5-Coder-3B-Instruct (Adapter)",
            "total_tasks": "Pending Colab SFT",
            "valid_command_rate": "Target >95",
            "execution_success_rate": "Target >90",
            "semantic_pass_rate": "Target >88",
            "semantic_pass_count": "-",
            "exact_match_rate": "Target >75",
            "timeout_count": 0,
            "avg_latency": "~0.08",
            "avg_tokens": "~28",
        })

    # 3. Qwen SFT with RAG
    rag_stats = load_evaluation_summary(qwen_rag_file)
    if rag_stats:
        rag_stats["label"] = "3. Qwen SFT + RAG"
        rag_stats["architecture"] = "Local LoRA + RAG Pipeline"
        models_to_compare.append(rag_stats)
    else:
        models_to_compare.append({
            "label": "3. Qwen SFT + RAG",
            "architecture": "Local LoRA + RAG Pipeline",
            "model": "Qwen2.5-Coder-3B-Instruct (Adapter)",
            "total_tasks": "Pending Colab SFT",
            "valid_command_rate": "Target 100",
            "execution_success_rate": "Target >95",
            "semantic_pass_rate": "Target >92",
            "semantic_pass_count": "-",
            "exact_match_rate": "Target >80",
            "timeout_count": 0,
            "avg_latency": "~0.12",
            "avg_tokens": "~350",
        })

    return format_comparison_table(models_to_compare)


def main() -> None:
    parser = argparse.ArgumentParser(description="NL2SH Model Evaluation Comparison Framework")
    parser.add_argument("--groq-file", default="data/evaluation/groq_evaluation_results.jsonl")
    parser.add_argument("--qwen-norag-file", default="data/evaluation/qwen_sft_norag_results.jsonl")
    parser.add_argument("--qwen-rag-file", default="data/evaluation/qwen_sft_rag_results.jsonl")

    args = parser.parse_args()

    print("=" * 70)
    print("NL2SH TRI-MODEL EVALUATION COMPARISON BENCHMARK")
    print("=" * 70)
    table = generate_comparison_report(
        groq_results_file=args.groq_file,
        qwen_norag_file=args.qwen_norag_file,
        qwen_rag_file=args.qwen_rag_file,
    )
    print(table)
    print("=" * 70)


if __name__ == "__main__":
    main()
