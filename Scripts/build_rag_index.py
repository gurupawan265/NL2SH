"""
Build script for NL2SH FAISS vector index.
Ingests and validates data/corpus/bash_reference.jsonl,
generates dense embeddings on CPU using sentence-transformers/all-MiniLM-L6-v2,
and serializes the FAISS CPU index and metadata to data/vectorstore/bash_index/.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RAG.config import RAGConfig
from RAG.corpus import load_corpus
from RAG.embedder import SentenceTransformerEmbedder
from RAG.vector_store import FAISSVectorStore


def build_index(
    corpus_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    device: str = "cpu",
) -> FAISSVectorStore:
    cfg = RAGConfig()
    c_path = Path(corpus_path) if corpus_path else cfg.corpus_file
    out_dir = Path(output_dir) if output_dir else cfg.vectorstore_dir

    print("=" * 60)
    print("NL2SH RAG VECTOR INDEX BUILD")
    print(f"Corpus file:       {c_path}")
    print(f"Output directory:  {out_dir}")
    print(f"Embedding model:   {cfg.model_name}")
    print(f"Device:            {device}")
    print("=" * 60)

    start_time = time.perf_counter()

    # 1. Load and validate corpus
    print("Loading and validating corpus documents...")
    documents = load_corpus(c_path)
    print(f"Successfully loaded and validated {len(documents)} documentation chunks.")

    # 2. Initialize embedding engine
    print(f"Initializing embedder ({cfg.model_name}) on {device}...")
    embedder = SentenceTransformerEmbedder(
        model_name=cfg.model_name,
        device=device,
        normalize=cfg.normalize_embeddings,
    )

    # 3. Build FAISS index
    print("Generating dense document embeddings and building FAISS index...")
    t_emb_start = time.perf_counter()
    vector_store = FAISSVectorStore(dimension=embedder.dimension)
    vector_store.build(documents, embedder)
    t_emb_duration = time.perf_counter() - t_emb_start

    # Record model metadata in build_info
    vector_store.build_info["model_name"] = cfg.model_name
    vector_store.build_info["embedding_dim"] = embedder.dimension
    vector_store.build_info["corpus_file"] = str(c_path)
    vector_store.build_info["build_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 4. Save index and metadata
    print(f"Saving FAISS index and metadata to {out_dir}...")
    vector_store.save(out_dir)

    total_duration = time.perf_counter() - start_time

    # Calculate index sizes
    idx_size = (out_dir / "index.faiss").stat().st_size if (out_dir / "index.faiss").is_file() else 0
    meta_size = (out_dir / "metadata.json").stat().st_size if (out_dir / "metadata.json").is_file() else 0

    print("=" * 60)
    print("INDEX BUILD SUCCESSFUL")
    print(f"Indexed chunks:    {vector_store.count}")
    print(f"Embedding dim:     {vector_store.dimension}")
    print(f"Embedding time:    {t_emb_duration:.2f}s")
    print(f"Total build time:  {total_duration:.2f}s")
    print(f"Index file size:   {idx_size / 1024:.1f} KB (index.faiss)")
    print(f"Metadata size:     {meta_size / 1024:.1f} KB (metadata.json)")
    print("=" * 60)

    return vector_store


if __name__ == "__main__":
    build_index()
