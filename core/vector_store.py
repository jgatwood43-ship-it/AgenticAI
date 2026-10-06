"""
core/vector_store.py
────────────────────
Builds or connects to the PGVector indexes used by the application.

Two separate vector stores are maintained:

1. General RAG vector store
   - Contains NIST, CIS, and other reference documents.
   - Uses settings.pg_table.

2. Database schema vector store
   - Contains generated Markdown descriptions of database tables.
   - Uses settings.pg_schema_table when configured.
   - Defaults to "document_embeddings_v3".

Uses LlamaIndex's SemanticSplitterNodeParser for document chunking.
"""

from __future__ import annotations

from typing import List, Optional

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from config.settings import settings

# BAAI/bge-large-en-v1.5 produces 1,024-dimensional embeddings.
EMBEDDING_DIMENSION = 1024

# Used when pg_schema_table has not yet been added to config/settings.py.
DEFAULT_SCHEMA_TABLE = "document_embeddings_v3"


def _get_schema_table_name() -> str:
    """
    Return the configured PGVector table name for schema documents.

    This allows the application to work before pg_schema_table is added
    to config/settings.py.
    """

    configured_table = getattr(
        settings,
        "pg_schema_table",
        DEFAULT_SCHEMA_TABLE,
    )

    table_name = str(configured_table).strip()

    if not table_name:
        return DEFAULT_SCHEMA_TABLE

    return table_name


def _build_pgvector_store(
    table_name: str,
) -> PGVectorStore:
    """
    Create a PGVectorStore connection for the specified logical table.

    LlamaIndex may apply its own internal table-name prefix when creating
    the physical PostgreSQL table.
    """

    cleaned_table_name = table_name.strip()

    if not cleaned_table_name:
        raise ValueError("PGVector table name cannot be empty.")

    return PGVectorStore.from_params(
        host=settings.pg_host,
        port=str(settings.pg_port),
        user=settings.pg_user,
        password=settings.pg_password,
        database=settings.pg_database,
        table_name=cleaned_table_name,
        embed_dim=EMBEDDING_DIMENSION,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
        },
    )


def build_vector_store() -> PGVectorStore:
    """
    Connect to the general RAG Postgres vector store.

    This store contains documents such as NIST and CIS guidance.
    """

    return _build_pgvector_store(
        table_name=settings.pg_table,
    )


def build_schema_vector_store() -> PGVectorStore:
    """
    Connect to the separate database-schema PGVector store.

    This store contains generated schema documents such as:

        docs/schema/employees.md
        docs/schema/departments.md
        docs/schema/badge_access_log.md
    """

    return _build_pgvector_store(
        table_name=_get_schema_table_name(),
    )


def build_index(
    vector_store: PGVectorStore,
    embed_model: HuggingFaceEmbedding,
) -> VectorStoreIndex:
    """
    Attach a VectorStoreIndex to an existing PGVector store.

    This function can be used for either the general RAG store or the
    schema-document store.
    """

    if vector_store is None:
        raise ValueError("vector_store cannot be None.")

    if embed_model is None:
        raise ValueError("embed_model cannot be None.")

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
    )

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )


def build_schema_index(
    embed_model,
) -> VectorStoreIndex:
    """
    Build an in-memory vector index from the authoritative database schema
    Markdown documents under docs/schema.

    This index is intentionally separate from the main pgvector-backed policy
    index. It contains only database schema documentation.
    """

    from pathlib import Path

    from llama_index.core import (
        Document,
        SimpleDirectoryReader,
        VectorStoreIndex,
    )

    project_root = Path(__file__).resolve().parent.parent
    schema_directory = project_root / "docs" / "schema"

    schema_files = sorted(schema_directory.rglob("*.md"))

    print("\n[SchemaIndex] Inspecting schema documentation...")
    print(f"[SchemaIndex] Directory : {schema_directory}")
    print(f"[SchemaIndex] Files     : {len(schema_files)}")

    for schema_file in schema_files:
        print("[SchemaIndex]   - " f"{schema_file.relative_to(project_root)}")

    if not schema_files:
        raise RuntimeError(
            "No Markdown schema files were found under " f"{schema_directory}."
        )

    # Load only Markdown files from docs/schema.
    loaded_documents = SimpleDirectoryReader(
        input_files=[str(schema_file) for schema_file in schema_files],
    ).load_data()

    print(f"[SchemaIndex] Documents loaded: " f"{len(loaded_documents)}")

    if not loaded_documents:
        raise RuntimeError(
            "Schema Markdown files were found, but no documents " "were loaded."
        )

    schema_documents: list[Document] = []

    for document, schema_file in zip(
        loaded_documents,
        schema_files,
    ):
        document_text = str(document.text or "").strip()

        print(
            f"[SchemaIndex] Loaded "
            f"{schema_file.name}: "
            f"{len(document_text)} chars"
        )

        if not document_text:
            print(f"[SchemaIndex] WARNING: " f"{schema_file.name} is empty.")
            continue

        table_name = schema_file.stem

        schema_documents.append(
            Document(
                text=(
                    f"TABLE NAME: {table_name}\n"
                    f"SCHEMA FILE: {schema_file.name}\n\n"
                    f"{document_text}"
                ),
                metadata={
                    "source_type": "database_schema",
                    "table_name": table_name,
                    "file_name": schema_file.name,
                    "file_path": str(schema_file),
                },
            )
        )

    if not schema_documents:
        raise RuntimeError("All schema Markdown documents were empty.")

    print(f"[SchemaIndex] Non-empty schema documents: " f"{len(schema_documents)}")

    schema_index = VectorStoreIndex.from_documents(
        schema_documents,
        embed_model=embed_model,
        show_progress=True,
    )

    print("[SchemaIndex] Schema index successfully built.")

    return schema_index


def ingest_documents(
    documents: List[Document],
    embed_model: HuggingFaceEmbedding,
    vector_store: Optional[PGVectorStore] = None,
) -> VectorStoreIndex:
    """
    Chunk general reference documents with SemanticSplitterNodeParser,
    embed them with bge-large-en-v1.5, and insert them into PGVector.

    Parameters
    ----------
    documents:
        LlamaIndex Document objects to ingest.

    embed_model:
        The bge-large-en-v1.5 embedding model.

    vector_store:
        Optional pre-built PGVectorStore. When omitted, the general RAG
        vector store is used.
    """

    if not documents:
        raise ValueError("At least one document is required for ingestion.")

    if embed_model is None:
        raise ValueError("embed_model cannot be None.")

    if vector_store is None:
        vector_store = build_vector_store()

    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        breakpoint_percentile_threshold=(settings.semantic_breakpoint_threshold),
        # Controls how many surrounding sentences are included when
        # computing semantic similarity for split decisions.
        buffer_size=1,
    )

    nodes = splitter.get_nodes_from_documents(documents)

    print(f"[Ingest] {len(documents)} documents " f"→ {len(nodes)} semantic nodes")

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
    )

    return VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )


def ingest_schema_documents(
    documents: List[Document],
    embed_model: HuggingFaceEmbedding,
    vector_store: Optional[PGVectorStore] = None,
) -> VectorStoreIndex:
    """
    Embed and insert database-schema documents into the schema vector store.

    Schema documents are normally small, structured Markdown files containing
    one table definition per document. Each schema document is kept as one
    vector node so its columns, purpose, and relationships remain together.

    Parameters
    ----------
    documents:
        Schema documents loaded from docs/schema.

    embed_model:
        The same embedding model later passed to build_schema_index().

    vector_store:
        Optional pre-built schema PGVectorStore. When omitted, the function
        uses build_schema_vector_store().
    """

    if not documents:
        raise ValueError("At least one schema document is required for ingestion.")

    if embed_model is None:
        raise ValueError("embed_model cannot be None.")

    if vector_store is None:
        vector_store = build_schema_vector_store()

    print(f"[Schema Ingest] Inserting " f"{len(documents)} schema documents")

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
    )

    # Each Markdown file represents one table. Keeping each document intact
    # prevents table columns and relationships from being split apart.
    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
