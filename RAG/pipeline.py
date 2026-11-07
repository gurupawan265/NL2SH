"""
Complete end-to-end RAG Generation Pipeline for NL2SH.
Coordinates retrieval, prompt construction, LLM provider invocation,
and command validation into a single modular, injectable workflow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path

from .config import RAGConfig
from .embedder import SentenceTransformerEmbedder
from .generator import BaseCommandGenerator, GenerationResult, MockCommandGenerator
from .prompt_builder import PromptBuilder
from .retriever import BashRetriever
from .validator import CommandValidator
from .vector_store import FAISSVectorStore


class RAGPipeline:
    """
    End-to-end pipeline:
    Query -> Retrieve -> Prompt -> Generate -> Validate -> GenerationResult
    """

    def __init__(
        self,
        retriever: BashRetriever,
        generator: BaseCommandGenerator,
        prompt_builder: Optional[PromptBuilder] = None,
        validator: Optional[CommandValidator] = None,
        min_retrieval_threshold: Optional[float] = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.validator = validator or CommandValidator()
        self.min_retrieval_threshold = min_retrieval_threshold

    @classmethod
    def create_mock_pipeline(
        cls,
        vector_store: FAISSVectorStore,
        embedder: Any,
        canned_responses: Optional[Dict[str, str]] = None,
        default_response: str = "ls -la",
    ) -> RAGPipeline:
        """
        Convenience factory to create an entirely offline, mock-backed pipeline for testing.
        """
        retriever = BashRetriever(vector_store=vector_store, embedder=embedder, default_top_k=3)
        generator = MockCommandGenerator(
            canned_responses=canned_responses,
            default_response=default_response,
        )
        return cls(retriever=retriever, generator=generator)

    def generate(
        self,
        instruction: str,
        top_k: int = 3,
        execution_context: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> GenerationResult:
        """
        Execute full RAG generation pipeline for an input natural-language instruction.
        """
        clean_inst = instruction.strip() if instruction else ""
        if not clean_inst:
            return GenerationResult(
                instruction="",
                command="",
                raw_response="",
                retrieved_doc_ids=[],
                provider_name=self.generator.provider_name,
                model_name=self.generator.model_name,
                is_valid=False,
                rejection_reason="Empty instruction provided to RAG pipeline",
            )

        threshold = score_threshold if score_threshold is not None else self.min_retrieval_threshold

        # 1. Retrieve relevant Bash documentation
        retrieved_docs = self.retriever.retrieve(
            clean_inst,
            top_k=top_k,
            score_threshold=threshold,
        )

        doc_ids = [d.doc_id for d in retrieved_docs]

        # 2. Check for retrieval absence
        if not retrieved_docs:
            return GenerationResult(
                instruction=clean_inst,
                command="",
                raw_response="",
                retrieved_doc_ids=[],
                provider_name=self.generator.provider_name,
                model_name=self.generator.model_name,
                is_valid=False,
                rejection_reason="No relevant Bash documentation retrieved for instruction",
                is_clarification=True,
            )

        # 3. Call generator with retrieved context
        result = self.generator.generate(
            instruction=clean_inst,
            context_documents=retrieved_docs,
            execution_context=execution_context,
        )

        return result

