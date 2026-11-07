"""
Evaluation script for NL2SH RAG retrieval engine.
Evaluates data/evaluation/retrieval_queries.jsonl against the FAISS index,
calculates Recall@1, Recall@3, Recall@5, MRR, and average query latency.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RAG.config import RAGConfig
from RAG.embedder import SentenceTransformerEmbedder
from RAG.retriever import BashRetriever
from RAG.vector_store import FAISSVectorStore


def evaluate_retrieval(
    queries_file: str | Path = "data/evaluation/retrieval_queries.jsonl",
    index_dir: str | Path = "data/vectorstore/bash_index",
) -> Dict[str, Any]:
    cfg = RAGConfig()
    q_path = Path(queries_file)
    idx_dir = Path(index_dir)

    if not q_path.is_file():
        raise FileNotFoundError(f"Queries file missing: {q_path}")
    if not idx_dir.is_dir():
        raise FileNotFoundError(f"Vector store directory missing: {idx_dir}")

    # Load queries
    queries: List[Dict[str, Any]] = []
    with open(q_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                queries.append(json.loads(stripped))

    # Initialize store and retriever
    print(f"Loading FAISS vector store from {idx_dir}...")
    store = FAISSVectorStore.load(idx_dir)
    embedder = SentenceTransformerEmbedder(model_name=cfg.model_name, device=cfg.device)
    retriever = BashRetriever(vector_store=store, embedder=embedder, default_top_k=5)

    print(f"Evaluating {len(queries)} retrieval queries (k=5)...")
    print("-" * 75)

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks: List[float] = []
    latencies: List[float] = []

    for q in queries:
        qid = q["query_id"]
        instruction = q["instruction"]
        relevant = set(cmd.lower() for cmd in q["relevant_commands"])

        t0 = time.perf_counter()
        results = retriever.retrieve(instruction, top_k=5)
        latency = (time.perf_counter() - t0) * 1000  # ms
        latencies.append(latency)

        retrieved_commands = [res.command.lower() for res in results]

        # Calculate hits
        r1 = any(cmd in relevant for cmd in retrieved_commands[:1])
        r3 = any(cmd in relevant for cmd in retrieved_commands[:3])
        r5 = any(cmd in relevant for cmd in retrieved_commands[:5])

        if r1:
            hits_at_1 += 1
        if r3:
            hits_at_3 += 1
        if r5:
            hits_at_5 += 1

        # Calculate reciprocal rank (1 / rank of first relevant match)
        rr = 0.0
        for rank, cmd in enumerate(retrieved_commands, start=1):
            if cmd in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        top_match = results[0].command if results else "None"
        top_score = results[0].score if results else 0.0
        status = "PASS" if r1 else ("P-R3" if r3 else ("P-R5" if r5 else "FAIL"))
        print(f"[{qid}] {instruction[:45]:<45} -> Top: {top_match:<12} (score: {top_score:.3f}) [{status}]")

    num_q = len(queries)
    recall_1 = hits_at_1 / num_q if num_q else 0.0
    recall_3 = hits_at_3 / num_q if num_q else 0.0
    recall_5 = hits_at_5 / num_q if num_q else 0.0
    mrr = sum(reciprocal_ranks) / num_q if num_q else 0.0
    avg_latency = sum(latencies) / num_q if num_q else 0.0

    print("=" * 60)
    print("NL2SH RETRIEVAL EVALUATION RESULTS")
    print(f"Total Queries Evaluated: {num_q}")
    print(f"Recall@1:                 {recall_1 * 100:.1f}% ({hits_at_1}/{num_q})")
    print(f"Recall@3:                 {recall_3 * 100:.1f}% ({hits_at_3}/{num_q})")
    print(f"Recall@5:                 {recall_5 * 100:.1f}% ({hits_at_5}/{num_q})")
    print(f"Mean Reciprocal Rank:    {mrr:.4f}")
    print(f"Average Query Latency:   {avg_latency:.2f} ms")
    print("=" * 60)

    return {
        "total_queries": num_q,
        "recall_at_1": recall_1,
        "recall_at_3": recall_3,
        "recall_at_5": recall_5,
        "mrr": mrr,
        "avg_latency_ms": avg_latency,
    }


if __name__ == "__main__":
    evaluate_retrieval()

