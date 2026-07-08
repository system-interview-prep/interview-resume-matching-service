"""RAG (Retrieval-Augmented Generation) package for Interview Documents.

Exposes main functions for embedding generation, vector database operations, and document retrieval.
"""

from modules.rag.embedding import (
    EmbeddingAdapter,
    MockEmbeddingAdapter,
    OpenAIEmbeddingAdapter,
    BedrockEmbeddingAdapter,
    build_embedding_adapter_from_env,
    generate_embedding_payloads,
    read_interview_document,
)
from modules.rag.vector_insert import (
    VectorStoreAdapter,
    insert_vectors,
    upsert_vectors,
    upsert_interview_document,
)
from modules.rag.retriever import Retriever
from modules.rag.quality_layer import QualityLayer
from modules.rag.prompt_builder import PromptBuilder
from modules.rag.llm import LLMService

__all__ = [
    "EmbeddingAdapter",
    "MockEmbeddingAdapter",
    "OpenAIEmbeddingAdapter",
    "BedrockEmbeddingAdapter",
    "build_embedding_adapter_from_env",
    "generate_embedding_payloads",
    "read_interview_document",
    "VectorStoreAdapter",
    "insert_vectors",
    "upsert_vectors",
    "upsert_interview_document",
    "Retriever",
    "QualityLayer",
    "PromptBuilder",
    "LLMService",
]
