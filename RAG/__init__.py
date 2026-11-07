"""
NL2SH RAG (Retrieval-Augmented Generation) package.
Provides modular, lightweight Bash documentation corpus loading, embedding,
FAISS vector indexing, and semantic retrieval.
"""

from .config import RAGConfig
from .corpus import CorpusDocument, load_corpus, validate_corpus_record
from .embedder import BaseEmbedder, MockEmbedder, SentenceTransformerEmbedder
from .retriever import BashRetriever, RetrievalResult
from .vector_store import FAISSVectorStore

import sys
# Alias 'rag' in sys.modules for case-insensitive compatibility
sys.modules.setdefault("rag", sys.modules[__name__])

__all__ = [
    "RAGConfig",
    "CorpusDocument",
    "load_corpus",
    "validate_corpus_record",
    "BaseEmbedder",
    "MockEmbedder",
    "SentenceTransformerEmbedder",
    "FAISSVectorStore",
    "BashRetriever",
    "RetrievalResult",
]
