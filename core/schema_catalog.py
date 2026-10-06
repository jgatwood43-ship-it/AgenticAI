"""
core/schema_catalog.py
──────────────────────
Loads the complete authoritative database schema catalog from docs/schema.

The schema corpus is small, so SQL planning receives every schema document
rather than relying exclusively on semantic top-k retrieval. This prevents
required tables, columns, relationships, or business rules from being omitted.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIRECTORY = PROJECT_ROOT / "docs" / "schema"


def load_full_schema_catalog() -> str:
    """
    Load all Markdown schema documents into one authoritative context.

    Returns
    -------
    str
        A clearly separated schema catalog containing every Markdown file.

    Raises
    ------
    RuntimeError
        If the schema directory is missing, contains no Markdown files,
        or all files are empty.
    """

    if not SCHEMA_DIRECTORY.exists():
        raise RuntimeError("Schema directory does not exist: " f"{SCHEMA_DIRECTORY}")

    schema_files = sorted(SCHEMA_DIRECTORY.rglob("*.md"))

    if not schema_files:
        raise RuntimeError(
            "No Markdown schema files were found under " f"{SCHEMA_DIRECTORY}."
        )

    sections: list[str] = []

    print("\n[SchemaCatalog] Loading complete schema catalog...")
    print(f"[SchemaCatalog] Directory : {SCHEMA_DIRECTORY}")
    print(f"[SchemaCatalog] Files     : {len(schema_files)}")

    for schema_file in schema_files:
        document_text = schema_file.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()

        if not document_text:
            print(f"[SchemaCatalog] WARNING: " f"{schema_file.name} is empty.")
            continue

        table_name = schema_file.stem

        sections.append(
            "\n".join(
                [
                    "=" * 72,
                    f"SCHEMA DOCUMENT: {schema_file.name}",
                    f"TABLE OR SUBJECT: {table_name}",
                    "=" * 72,
                    "",
                    document_text,
                ]
            )
        )

        print(f"[SchemaCatalog]   - {schema_file.name}: " f"{len(document_text)} chars")

    if not sections:
        raise RuntimeError("All Markdown schema files were empty.")

    catalog = "\n\n".join(sections)

    print(f"[SchemaCatalog] Complete catalog length: " f"{len(catalog)} chars")

    return catalog


def load_schema_documents(
    document_names: list[str],
) -> str:
    """
    Load selected authoritative Markdown schema documents.

    Parameters
    ----------
    document_names:
        File names or stems, such as:
        ["employees", "badge_access_log", "rooms"]

    Returns
    -------
    str
        Combined schema context containing only the requested documents.
    """
    if not document_names:
        return ""

    normalized_names = {
        str(name).strip().lower().removesuffix(".md")
        for name in document_names
        if str(name).strip()
    }

    sections: list[str] = []

    for schema_file in sorted(SCHEMA_DIRECTORY.rglob("*.md")):
        if schema_file.stem.lower() not in normalized_names:
            continue

        document_text = schema_file.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()

        if not document_text:
            continue

        sections.append(
            "\n".join(
                [
                    "=" * 72,
                    f"SCHEMA DOCUMENT: {schema_file.name}",
                    f"TABLE OR SUBJECT: {schema_file.stem}",
                    "=" * 72,
                    "",
                    document_text,
                ]
            )
        )

    return "\n\n".join(sections).strip()
