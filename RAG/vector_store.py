"""
Persistent FAISS vector store for NL2SH RAG.
Stores document embeddings using FAISS CPU and metadata via standard JSON (no pickle).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np

from .corpus import CorpusDocument
from .embedder import BaseEmbedder


class FAISSVectorStore:
    """
    In-memory and disk-persisted FAISS vector store.
    Uses Inner Product (Cosine Similarity on L2-normalized vectors).
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.documents: List[CorpusDocument] = []
        self.build_info: Dict[str, Any] = {}

    @property
    def is_built(self) -> bool:
        """Return True if index exists and contains vectors."""
        return self.index is not None and self.index.ntotal > 0

    @property
    def count(self) -> int:
        """Return number of indexed documents."""
        return self.index.ntotal if self.index is not None else 0

    def build(self, documents: List[CorpusDocument], embedder: BaseEmbedder) -> None:
        """
        Embed corpus documents and build the FAISS index.
        """
        if not documents:
            raise ValueError("Cannot build index from empty document list")

        if self.dimension and embedder.dimension != self.dimension:
            raise ValueError(f"Expected embedding dimension {self.dimension}, got {embedder.dimension}")

        self.dimension = embedder.dimension
        texts = [doc.to_search_text() for doc in documents]
        embeddings = embedder.embed_texts(texts)

        if embeddings.shape[0] != len(documents):
            raise ValueError(f"Mismatch: {len(documents)} documents produced {embeddings.shape[0]} embeddings")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected embedding dimension {self.dimension}, got {embeddings.shape[1]}")

        # Ensure float32 format
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # IndexFlatIP computes inner products. On L2-normalized vectors, this equals cosine similarity.
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.documents = list(documents)
        self.build_info = {
            "dimension": self.dimension,
            "total_documents": len(self.documents),
            "metric": "inner_product_cosine",
        }

    def save(self, directory_path: str | Path) -> None:
        """
        Persist the FAISS index file and document metadata JSON to directory.
        Strictly avoids unsafe pickle serialization.
        """
        if not self.is_built:
            raise RuntimeError("Cannot save an unbuilt vector store")

        out_dir = Path(directory_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        index_file = out_dir / "index.faiss"
        metadata_file = out_dir / "metadata.json"

        # 1. Write FAISS binary index
        faiss.write_index(self.index, str(index_file))

        # 2. Write document metadata as human-readable, safe JSON
        meta_payload = {
            "build_info": self.build_info,
            "documents": [doc.to_dict() for doc in self.documents],
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, directory_path: str | Path) -> FAISSVectorStore:
        """
        Load an existing FAISS index and metadata from directory.
        Validates index consistency and schema integrity.
        """
        in_dir = Path(directory_path)
        index_file = in_dir / "index.faiss"
        metadata_file = in_dir / "metadata.json"

        if not index_file.is_file():
            raise FileNotFoundError(f"FAISS index file missing: {index_file}")
        if not metadata_file.is_file():
            raise FileNotFoundError(f"Metadata JSON file missing: {metadata_file}")

        # Read metadata safely
        with open(metadata_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        build_info = meta.get("build_info", {})
        dimension = build_info.get("dimension", 384)
        doc_dicts = meta.get("documents", [])

        # Read FAISS index
        index = faiss.read_index(str(index_file))

        # Validate dimensional consistency
        if index.d != dimension:
            raise ValueError(f"Index dimension {index.d} does not match metadata dimension {dimension}")
        if index.ntotal != len(doc_dicts):
            raise ValueError(f"Vector count {index.ntotal} does not match document count {len(doc_dicts)}")

        # Reconstruct CorpusDocument objects
        documents = [
            CorpusDocument(
                doc_id=d["doc_id"],
                command=d["command"],
                title=d["title"],
                section=d["section"],
                content=d["content"],
                examples=d["examples"],
                source_url=d["source_url"],
                source_name=d["source_name"],
                license=d["license"],
                command_category=d["command_category"],
                safety_level=d["safety_level"],
            )
            for d in doc_dicts
        ]

        store = cls(dimension=dimension)
        store.index = index
        store.documents = documents
        store.build_info = build_info
        return store

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[CorpusDocument, float]]:
        """
        Search for top_k most similar documents to query_vector.
        Returns list of (CorpusDocument, score) tuples ranked in descending similarity order.
        """
        if not self.is_built:
            raise RuntimeError("Cannot search an uninitialized vector store")

        if top_k <= 0:
            return []

        # Format query_vector
        vec = query_vector.astype(np.float32)
        if vec.ndim == 1:
            vec = np.expand_dims(vec, axis=0)
        elif vec.ndim != 2 or vec.shape[0] != 1:
            raise ValueError(f"Query vector must be 1D or 2D with shape (1, dim), got {vec.shape}")

        if vec.shape[1] != self.dimension:
            raise ValueError(f"Query vector dim {vec.shape[1]} does not match index dim {self.dimension}")

        effective_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(vec, effective_k)

        results: List[Tuple[CorpusDocument, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            results.append((doc, float(dist)))

        return results
