"""
Centralized configuration for NL2SH RAG pipeline.
Avoids hardcoded absolute paths and defaults to CPU execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RAGConfig:
    """Configuration settings for embedding, corpus, and FAISS vector index."""

    # Embedding settings
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    device: str = "cpu"
    normalize_embeddings: bool = True

    # Storage paths (relative to project root)
    corpus_file: Path = PROJECT_ROOT / "data" / "corpus" / "bash_reference.jsonl"
    vectorstore_dir: Path = PROJECT_ROOT / "data" / "vectorstore" / "bash_index"
    index_file_name: str = "index.faiss"
    metadata_file_name: str = "metadata.json"

    # Retrieval parameters
    default_top_k: int = 3
    max_top_k: int = 10
    min_score_threshold: float = 0.0  # Optional relevance filtering threshold

