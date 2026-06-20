"""
core/vector_store.py
─────────────────────
Builds (or connects to) the PGVector index that backs RAG retrieval.
Uses LlamaIndex's SemanticSplitterNodeParser for document chunking.
"""
from __future__ import annotations

from typing import List, Optional

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config.settings import settings


def build_vector_store() -> PGVectorStore:
    """Connect to the Postgres vector store."""
    return PGVectorStore.from_params(
        host=settings.pg_host,
        port=str(settings.pg_port),
        user=settings.pg_user,
        password=settings.pg_password,
        database=settings.pg_database,
        table_name=settings.pg_table,
        embed_dim=1024,          # bge-large-en-v1.5 output dimension
        hnsw_kwargs={            # HNSW index for fast ANN search
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
        },
    )


def build_index(
    vector_store: PGVectorStore,
    embed_model: HuggingFaceEmbedding,
) -> VectorStoreIndex:
    """Attach a VectorStoreIndex to the existing PGVector store."""
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )


def ingest_documents(
    documents: List[Document],
    embed_model: HuggingFaceEmbedding,
    vector_store: Optional[PGVectorStore] = None,
) -> VectorStoreIndex:
    """
    Chunk documents with SemanticSplitterNodeParser, embed with bge-large-en-v1.5,
    and upsert into PGVector.

    Parameters
    ----------
    documents   : List of LlamaIndex Document objects to ingest.
    embed_model : bge-large-en-v1.5 embedding model.
    vector_store: Optional pre-built PGVectorStore (created if None).
    """
    if vector_store is None:
        vector_store = build_vector_store()

    # ── Semantic chunking ────────────────────────────────────────────────────
    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        breakpoint_percentile_threshold=settings.semantic_breakpoint_threshold,
        # buffer_size controls how many surrounding sentences are used
        # when computing semantic similarity for split decisions.
        buffer_size=1,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"[Ingest] {len(documents)} documents → {len(nodes)} semantic nodes")

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    return index
