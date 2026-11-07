"""
Retrieval module for NL2SH RAG pipeline.
Takes natural language instructions, embeds them, queries the vector store,
applies relevance filters, and formats clean reference context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from .corpus import CorpusDocument
from .embedder import BaseEmbedder
from .vector_store import FAISSVectorStore


@dataclass
class RetrievalResult:
    """Represents a retrieved document match with score and ranking."""

    doc_id: str
    command: str
    title: str
    section: str
    content: str
    examples: List[str]
    score: float
    command_category: str
    safety_level: str
    metadata: Dict[str, Any]

    @classmethod
    def from_doc_and_score(cls, doc: CorpusDocument, score: float) -> RetrievalResult:
        return cls(
            doc_id=doc.doc_id,
            command=doc.command,
            title=doc.title,
            section=doc.section,
            content=doc.content,
            examples=doc.examples,
            score=score,
            command_category=doc.command_category,
            safety_level=doc.safety_level,
            metadata={
                "source_url": doc.source_url,
                "source_name": doc.source_name,
                "license": doc.license,
            },
        )


class BashRetriever:
    """
    Coordinates query embedding, vector similarity search, and context assembly.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: BaseEmbedder,
        default_top_k: int = 3,
        min_score_threshold: Optional[float] = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.default_top_k = default_top_k
        self.min_score_threshold = min_score_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documentation chunks for a natural-language query.
        Returns ranked list of RetrievalResult objects in descending score order.
        """
        clean_query = query.strip() if query else ""
        if not clean_query:
            return []

        k = top_k if top_k is not None else self.default_top_k
        if k <= 0:
            return []

        threshold = score_threshold if score_threshold is not None else self.min_score_threshold

        # Embed query on CPU
        query_vec = self.embedder.embed_query(clean_query)

        # Search FAISS index
        raw_matches = self.vector_store.search(query_vec, top_k=k)

        # Filter by threshold if specified
        results: List[RetrievalResult] = []
        for doc, score in raw_matches:
            if threshold is not None and score < threshold:
                continue
            results.append(RetrievalResult.from_doc_and_score(doc, score))

        return results

    def format_context(self, results: List[RetrievalResult]) -> str:
        """
        Assemble retrieved documentation chunks into a clean, concise,
        structured text block suitable for injection into an LLM prompt.
        """
        if not results:
            return "No relevant documentation found."

        chunks: List[str] = []
        for i, res in enumerate(results, start=1):
            ex_str = "\n  - ".join(res.examples) if res.examples else "None"
            chunk = (
                f"[{i}] Utility: {res.command} ({res.title})\n"
                f"Description: {res.content}\n"
                f"Examples:\n  - {ex_str}"
            )
            chunks.append(chunk)

        return "\n\n".join(chunks)
