"""
Embedding model wrappers for NL2SH RAG.
Supports SentenceTransformer on CPU and a MockEmbedder for fast, offline unit tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
from typing import List
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for text embedding engines."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of text strings into a 2D float32 numpy array of shape (N, dim)."""

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string into a 1D float32 numpy array of shape (dim,)."""
        embs = self.embed_texts([query])
        return embs[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding vector dimensionality."""


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    SentenceTransformer wrapper for dense embeddings.
    Defaults to all-MiniLM-L6-v2 running on CPU with L2-normalized outputs.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._dim = 384

        # Lazy model initialization
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            # Retrieve exact embedding dimension from model
            if hasattr(self._model, "get_embedding_dimension"):
                self._dim = self._model.get_embedding_dimension()
            else:
                self._dim = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        if self._model is not None:
            return self._dim
        return 384

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        model = self._get_model()
        embs = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            device=self.device,
        )
        return embs.astype(np.float32)


class MockEmbedder(BaseEmbedder):
    """
    Deterministic mock embedder for offline unit tests.
    Generates reproducible unit-length float32 vectors without downloading model weights.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        results = []
        for text in texts:
            # Hash text to seed random state deterministically
            h = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.RandomState(h)
            vec = rng.randn(self._dimension).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            results.append(vec)

        return np.vstack(results).astype(np.float32)
