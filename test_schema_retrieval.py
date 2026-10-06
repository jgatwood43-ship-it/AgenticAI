"""
test_schema_retrieval.py
────────────────────────
Standalone validation for the database schema retrieval index.

This script verifies that:

1. Markdown files exist under docs/schema.
2. build_schema_index() loads those files.
3. SchemaRetriever returns useful schema details.
4. Retrieved text contains readable table and column information.

Run from the project root:

    python test_schema_retrieval.py
"""

from __future__ import annotations

from pathlib import Path

from core.llm_factory import (
    build_embed_model,
    configure_llama_globals,
)
from core.schema_retriever import SchemaRetriever
from core.vector_store import build_schema_index

PROJECT_ROOT = Path(__file__).resolve().parent
SCHEMA_DIRECTORY = PROJECT_ROOT / "docs" / "schema"

TEST_QUERIES = (
    "employees first and last names",
    "employee badge access records",
    "secure rooms and room access levels",
    "employee clock in and clock out activity",
    (
        "relationships between employees, badge access logs, "
        "rooms, and employee time records"
    ),
)


def find_schema_files() -> list[Path]:
    """Return every Markdown schema file under docs/schema."""

    return sorted(SCHEMA_DIRECTORY.rglob("*.md"))


def print_schema_files(schema_files: list[Path]) -> None:
    """Display schema files found by the validation script."""

    print("\n" + "=" * 72)
    print("SCHEMA FILE DISCOVERY")
    print("=" * 72)

    print(f"Project root     : {PROJECT_ROOT}")
    print(f"Schema directory : {SCHEMA_DIRECTORY}")
    print(f"Files found      : {len(schema_files)}")

    for schema_file in schema_files:
        relative_path = schema_file.relative_to(PROJECT_ROOT)
        print(f"  - {relative_path}")


def validate_result(query: str, context: str) -> None:
    """Print and perform basic quality checks on retrieved schema text."""

    cleaned_context = str(context or "").strip()

    print("\n" + "-" * 72)
    print(f"QUERY: {query}")
    print("-" * 72)

    if not cleaned_context:
        print("RESULT: NO SCHEMA CONTEXT RETURNED")
        return

    print(f"Context length: {len(cleaned_context)} characters")
    print()
    print(cleaned_context)

    printable_count = sum(character.isprintable() for character in cleaned_context)

    printable_ratio = printable_count / len(cleaned_context)

    print()
    print(f"Printable ratio: {printable_ratio:.2%}")

    if printable_ratio < 0.90:
        print(
            "WARNING: Retrieved schema text may be corrupted " "or incorrectly decoded."
        )


def main() -> None:
    """Build and test the schema retrieval index."""

    schema_files = find_schema_files()
    print_schema_files(schema_files)

    if not schema_files:
        raise RuntimeError(
            "No Markdown schema files were found under " f"{SCHEMA_DIRECTORY}."
        )

    print("\n" + "=" * 72)
    print("INITIALIZING EMBEDDINGS")
    print("=" * 72)

    configure_llama_globals()
    embed_model = build_embed_model()

    print("\n" + "=" * 72)
    print("BUILDING SCHEMA INDEX")
    print("=" * 72)

    schema_index = build_schema_index(
        embed_model=embed_model,
    )

    print("\n" + "=" * 72)
    print("RAW SCHEMA INDEX TEST")
    print("=" * 72)

    raw_retriever = schema_index.as_retriever(
        similarity_top_k=5,
    )

    raw_nodes = raw_retriever.retrieve("employees first and last names")

    print(f"Raw nodes returned: {len(raw_nodes)}")

    for index, node_with_score in enumerate(raw_nodes, start=1):
        score = getattr(node_with_score, "score", None)
        text = node_with_score.node.get_content()

        print("\n" + "-" * 72)
        print(f"NODE {index}")
        print(f"Score: {score}")
        print(f"Text length: {len(text or '')}")
        print(text[:1000])

    schema_retriever = SchemaRetriever(
        schema_index=schema_index,
        similarity_top_k=5,
    )

    print("\n" + "=" * 72)
    print("TESTING SCHEMA RETRIEVAL")
    print("=" * 72)

    successful_results = 0

    for query in TEST_QUERIES:
        context = schema_retriever.retrieve_context(query)

        validate_result(
            query=query,
            context=context,
        )

        if str(context or "").strip():
            successful_results += 1

    print("\n" + "=" * 72)
    print("TEST SUMMARY")
    print("=" * 72)

    print(f"Queries tested     : {len(TEST_QUERIES)}")
    print(f"Non-empty results  : {successful_results}")
    print(f"Empty results      : " f"{len(TEST_QUERIES) - successful_results}")

    if successful_results == 0:
        raise RuntimeError(
            "Schema files exist, but SchemaRetriever returned no "
            "context for any test query."
        )

    print("\nSchema retrieval validation completed.")


if __name__ == "__main__":
    main()
