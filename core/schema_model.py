"""
core/schema_model.py
────────────────────
Shared, deterministic schema model and tolerant Markdown parser.

The parser recognizes schema information by STRUCTURE rather than by requiring
specific section names such as "Column dictionary" or "Relationships".

Physical tables are accepted when a document contains a Markdown table whose
headers identify a column-name field and a data-type field. Relationship tables
are recognized from their header semantics. Direct/logical SQL-capable links are
kept separate from domain/temporal reasoning links.

Raw Markdown remains available to the LLM separately; this module provides the
single authoritative deterministic interpretation used by SQLPlanner,
SchemaGraph, and SQL validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SCHEMA_DOCUMENT_RE = re.compile(
    r"SCHEMA DOCUMENT:\s*([A-Za-z_][A-Za-z0-9_]*)\.md",
    flags=re.IGNORECASE,
)
_H1_RE = re.compile(r"^\s*#\s+(.+?)\s*$", flags=re.MULTILINE)

_COLUMN_NAME_HEADERS = {
    "column",
    "columns",
    "column name",
    "column_name",
    "field",
    "fields",
    "field name",
    "field_name",
    "name",
}
_TYPE_HEADERS = {"data type", "datatype", "data_type", "type", "sql type", "sql_type"}
_DESCRIPTION_HEADERS = {"description", "meaning", "notes", "definition"}
_NULLABLE_HEADERS = {"nullable", "null", "allows null", "allow null"}
_KEY_HEADERS = {"key", "index", "constraint"}
_EXTRA_HEADERS = {"extra", "attributes", "attribute"}
_RELATIONSHIP_TYPE_HEADERS = {
    "relationship type",
    "relationship",
    "relation type",
    "relation",
}
_JOIN_HEADERS = {
    "join possibility",
    "join",
    "join condition",
    "join expression",
    "relationship expression",
    "path",
}
_OTHER_TABLE_HEADERS = {
    "other table",
    "related table",
    "table",
    "target table",
}
_CONFIDENCE_HEADERS = {"confidence", "certainty"}

# Relationship types that are safe to expose as deterministic graph edges.
# Domain/temporal/indirect links remain semantic metadata for LLM reasoning.
_SQL_CAPABLE_RELATIONSHIP_MARKERS = (
    "direct pk/fk",
    "direct fk",
    "direct relationship",
    "logical derivation",
    "logical authorization",
    "logical join",
)

_EXPLICIT_RELATION_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)"
    r"\s*(?:=|->|→|=>)\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b",
    flags=re.IGNORECASE,
)

_LEGACY_COLUMN_RE = re.compile(
    r"^\s*[-*]\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*:\s*(.+?)\s*$",
    flags=re.MULTILINE,
)


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().strip("`")).lower()


def _identifier(value: str) -> str:
    cleaned = _norm(value).replace(" ", "_")
    return cleaned if _IDENTIFIER_RE.fullmatch(cleaned) else ""


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return []
    return [cell.strip().strip("`") for cell in stripped[1:-1].split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r"\s*:?-{3,}:?\s*", c or "") for c in cells)


def _markdown_tables(text: str) -> Iterable[tuple[list[str], list[list[str]]]]:
    """Yield (headers, rows) for structurally valid Markdown tables."""
    lines = text.splitlines()
    i = 0
    while i + 1 < len(lines):
        header = _split_markdown_row(lines[i])
        separator = _split_markdown_row(lines[i + 1])
        if (
            header
            and separator
            and len(header) == len(separator)
            and _is_separator_row(separator)
        ):
            rows: list[list[str]] = []
            j = i + 2
            while j < len(lines):
                row = _split_markdown_row(lines[j])
                if not row:
                    break
                if len(row) < len(header):
                    row += [""] * (len(header) - len(row))
                rows.append(row[: len(header)])
                j += 1
            yield header, rows
            i = j
        else:
            i += 1


def _header_index(headers: list[str], aliases: set[str]) -> int | None:
    normalized = [_norm(h) for h in headers]
    for idx, header in enumerate(normalized):
        if header in aliases:
            return idx
    return None


def _candidate_table_name(text: str, fallback_name: str = "") -> str:
    document_match = _SCHEMA_DOCUMENT_RE.search(text)
    if document_match:
        return _identifier(document_match.group(1))

    h1 = _H1_RE.search(text)
    if h1:
        title = h1.group(1).strip()
        title = re.sub(r"^table\s*:\s*", "", title, flags=re.IGNORECASE)
        candidate = _identifier(title)
        if candidate:
            return candidate

    return _identifier(fallback_name)


@dataclass(frozen=True)
class SchemaColumnRecord:
    table: str
    name: str
    data_type: str = ""
    nullable: str = ""
    key: str = ""
    extra: str = ""
    description: str = ""
    source_document: str = ""


@dataclass(frozen=True)
class SchemaRelationshipRecord:
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    relationship_type: str = ""
    source_document: str = ""
    source_text: str = ""
    sql_capable: bool = True

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            self.left_table,
            self.left_column,
            self.right_table,
            self.right_column,
        )


@dataclass(frozen=True)
class SchemaPlanningRelationshipRecord:
    """Documented relationship usable for evidence planning, not necessarily SQL."""

    left_table: str
    right_table: str
    relationship_type: str = ""
    source_document: str = ""
    source_text: str = ""
    confidence: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.left_table, self.right_table)


@dataclass
class SchemaCatalog:
    """Single authoritative deterministic schema representation."""

    tables: dict[str, set[str]] = field(default_factory=dict)
    column_records: dict[str, dict[str, SchemaColumnRecord]] = field(
        default_factory=dict
    )
    relationships: set[tuple[str, str, str, str]] = field(default_factory=set)
    relationship_records: tuple[SchemaRelationshipRecord, ...] = ()
    semantic_relationship_records: tuple[SchemaRelationshipRecord, ...] = ()
    planning_relationship_records: tuple[SchemaPlanningRelationshipRecord, ...] = ()
    warnings: tuple[str, ...] = ()

    def has_table(self, table: str) -> bool:
        return _norm(table) in self.tables

    def has_column(self, table: str, column: str) -> bool:
        return _norm(column) in self.tables.get(_norm(table), set())

    def render_physical_schema(
        self,
        *,
        include_descriptions: bool = True,
        include_relationships: bool = True,
    ) -> str:
        """Render a compact physical-schema view for SQL generation.

        This intentionally omits domain/planning prose.  It gives the SQL LLM
        only deterministic physical facts parsed from the markdown: tables,
        columns, types/descriptions, and strict SQL-capable relationships.
        """
        lines: list[str] = []

        for table_name in sorted(self.tables):
            lines.append(f"TABLE {table_name}")
            records = self.column_records.get(table_name, {})

            for column_name in sorted(self.tables[table_name]):
                record = records.get(column_name)
                if record is None:
                    lines.append(f"  - {column_name}")
                    continue

                details: list[str] = []
                if record.data_type:
                    details.append(record.data_type)
                if record.nullable:
                    details.append(f"nullable={record.nullable}")
                if record.key:
                    details.append(f"key={record.key}")
                if include_descriptions and record.description:
                    details.append(record.description)

                suffix = f" — {'; '.join(details)}" if details else ""
                lines.append(f"  - {column_name}{suffix}")

            lines.append("")

        if include_relationships and self.relationship_records:
            lines.append("STRICT SQL-CAPABLE RELATIONSHIPS")
            for relationship in sorted(
                self.relationship_records,
                key=lambda item: (
                    item.left_table,
                    item.left_column,
                    item.right_table,
                    item.right_column,
                ),
            ):
                lines.append(
                    f"  - {relationship.left_table}.{relationship.left_column} = "
                    f"{relationship.right_table}.{relationship.right_column}"
                )

        return "\n".join(lines).strip()


def _relationship_is_sql_capable(relationship_type: str) -> bool:
    normalized = _norm(relationship_type)
    return any(marker in normalized for marker in _SQL_CAPABLE_RELATIONSHIP_MARKERS)


def _parse_one_document(
    markdown_text: str,
    *,
    fallback_name: str = "",
    source_document: str = "",
) -> tuple[
    str,
    set[str],
    dict[str, SchemaColumnRecord],
    list[SchemaRelationshipRecord],
    list[SchemaPlanningRelationshipRecord],
    list[str],
]:
    table_name = _candidate_table_name(markdown_text, fallback_name)
    columns: set[str] = set()
    column_records: dict[str, SchemaColumnRecord] = {}
    relationships: list[SchemaRelationshipRecord] = []
    planning_relationships: list[SchemaPlanningRelationshipRecord] = []
    warnings: list[str] = []

    tables = list(_markdown_tables(markdown_text))

    # Detect physical-column tables by header semantics, not section title.
    for headers, rows in tables:
        column_idx = _header_index(headers, _COLUMN_NAME_HEADERS)
        type_idx = _header_index(headers, _TYPE_HEADERS)
        if column_idx is None or type_idx is None:
            continue

        description_idx = _header_index(headers, _DESCRIPTION_HEADERS)
        nullable_idx = _header_index(headers, _NULLABLE_HEADERS)
        key_idx = _header_index(headers, _KEY_HEADERS)
        extra_idx = _header_index(headers, _EXTRA_HEADERS)

        def value_at(row: list[str], idx: int | None) -> str:
            return row[idx].strip() if idx is not None and idx < len(row) else ""

        for row in rows:
            if column_idx >= len(row):
                continue
            name = _identifier(row[column_idx])
            if not name:
                continue
            columns.add(name)
            column_records.setdefault(
                name,
                SchemaColumnRecord(
                    table=table_name,
                    name=name,
                    data_type=value_at(row, type_idx),
                    nullable=value_at(row, nullable_idx),
                    key=value_at(row, key_idx),
                    extra=value_at(row, extra_idx),
                    description=value_at(row, description_idx),
                    source_document=source_document,
                ),
            )

    # Backward compatibility for legacy bullet schemas. Only used when no
    # structured physical-column table was found.
    if not columns:
        for match in _LEGACY_COLUMN_RE.finditer(markdown_text):
            name = _identifier(match.group(1))
            if name:
                columns.add(name)
                column_records.setdefault(
                    name,
                    SchemaColumnRecord(
                        table=table_name,
                        name=name,
                        description=match.group(2).strip(),
                        source_document=source_document,
                    ),
                )

    # Relationship tables are recognized structurally.  Every documented
    # relationship row can become a PLANNING edge when it names another
    # documented table and is not explicitly a "no relationship" row.
    # Only explicitly qualified column-to-column expressions whose relationship
    # type is SQL-capable become hard SQL relationships.
    for headers, rows in tables:
        rel_type_idx = _header_index(headers, _RELATIONSHIP_TYPE_HEADERS)
        join_idx = _header_index(headers, _JOIN_HEADERS)
        other_idx = _header_index(headers, _OTHER_TABLE_HEADERS)
        confidence_idx = _header_index(headers, _CONFIDENCE_HEADERS)
        if rel_type_idx is None or join_idx is None or other_idx is None:
            continue

        for row in rows:
            rel_type = row[rel_type_idx] if rel_type_idx < len(row) else ""
            join_text = row[join_idx] if join_idx < len(row) else ""
            other_table = _identifier(row[other_idx]) if other_idx < len(row) else ""
            confidence = (
                row[confidence_idx]
                if confidence_idx is not None and confidence_idx < len(row)
                else ""
            )

            normalized_type = _norm(rel_type)
            normalized_join = _norm(join_text)
            explicitly_disconnected = (
                "no natural direct join" in normalized_type
                or "no direct join recommended" in normalized_join
                or normalized_type.startswith("no natural")
            )

            if (
                table_name
                and other_table
                and other_table != table_name
                and not explicitly_disconnected
            ):
                planning_relationships.append(
                    SchemaPlanningRelationshipRecord(
                        left_table=table_name,
                        right_table=other_table,
                        relationship_type=normalized_type,
                        source_document=source_document,
                        source_text=join_text.strip(),
                        confidence=confidence.strip(),
                    )
                )

            sql_capable = _relationship_is_sql_capable(rel_type)

            for match in _EXPLICIT_RELATION_RE.finditer(join_text):
                record = SchemaRelationshipRecord(
                    left_table=_identifier(match.group(1)),
                    left_column=_identifier(match.group(2)),
                    right_table=_identifier(match.group(3)),
                    right_column=_identifier(match.group(4)),
                    relationship_type=normalized_type,
                    source_document=source_document,
                    source_text=match.group(0),
                    sql_capable=sql_capable,
                )
                if all(record.key):
                    relationships.append(record)

    if not table_name and columns:
        warnings.append(
            f"{source_document or '<schema document>'}: physical columns were found "
            "but no valid table identifier could be determined."
        )

    return (
        table_name,
        columns,
        column_records,
        relationships,
        planning_relationships,
        warnings,
    )


def _split_catalog_documents(schema_context: str) -> list[tuple[str, str]]:
    """Split concatenated SchemaCatalog text without requiring a heading style."""
    text = str(schema_context or "")
    matches = list(_SCHEMA_DOCUMENT_RE.finditer(text))
    if not matches:
        return [("", text)]

    docs: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        docs.append((f"{match.group(1)}.md", text[start:end]))
    return docs


def _build_catalog(parsed_documents: Iterable[tuple[str, str, str]]) -> SchemaCatalog:
    tables: dict[str, set[str]] = {}
    column_records: dict[str, dict[str, SchemaColumnRecord]] = {}
    records: list[SchemaRelationshipRecord] = []
    planning_records: list[SchemaPlanningRelationshipRecord] = []
    warnings: list[str] = []

    for source_document, fallback_name, text in parsed_documents:
        (
            table_name,
            columns,
            parsed_column_records,
            rels,
            parsed_planning_records,
            local_warnings,
        ) = _parse_one_document(
            text,
            fallback_name=fallback_name,
            source_document=source_document,
        )
        warnings.extend(local_warnings)

        # A document becomes a physical table only if structurally recognizable
        # physical columns exist.
        if table_name and columns:
            if table_name in tables:
                warnings.append(
                    f"{source_document}: duplicate table definition for {table_name}; merging columns."
                )
            tables.setdefault(table_name, set()).update(columns)
            column_records.setdefault(table_name, {}).update(parsed_column_records)
        records.extend(rels)
        planning_records.extend(parsed_planning_records)

    valid_sql_records: list[SchemaRelationshipRecord] = []
    semantic_records: list[SchemaRelationshipRecord] = []
    seen: set[tuple[str, str, str, str]] = set()

    for record in records:
        if not record.sql_capable:
            semantic_records.append(record)
            continue

        if record.left_table not in tables or record.right_table not in tables:
            warnings.append(
                f"{record.source_document}: relationship references undocumented table: "
                f"{record.source_text}."
            )
            continue
        if (
            record.left_column not in tables[record.left_table]
            or record.right_column not in tables[record.right_table]
        ):
            warnings.append(
                f"{record.source_document}: relationship references undocumented column: "
                f"{record.source_text}."
            )
            continue

        key = record.key
        reverse = (key[2], key[3], key[0], key[1])
        if key in seen or reverse in seen:
            continue
        seen.add(key)
        valid_sql_records.append(record)

    valid_planning_records: list[SchemaPlanningRelationshipRecord] = []
    seen_planning: set[tuple[str, str]] = set()

    for record in planning_records:
        if record.left_table not in tables or record.right_table not in tables:
            warnings.append(
                f"{record.source_document}: planning relationship references "
                f"undocumented table: {record.left_table} <-> {record.right_table}."
            )
            continue

        key = record.key
        reverse = (key[1], key[0])
        if key in seen_planning or reverse in seen_planning:
            continue

        seen_planning.add(key)
        valid_planning_records.append(record)

    return SchemaCatalog(
        tables=tables,
        column_records=column_records,
        relationships={r.key for r in valid_sql_records},
        relationship_records=tuple(valid_sql_records),
        semantic_relationship_records=tuple(semantic_records),
        planning_relationship_records=tuple(valid_planning_records),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def parse_schema_context(schema_context: str) -> SchemaCatalog:
    documents = _split_catalog_documents(schema_context)
    parsed = []
    for source_document, text in documents:
        fallback = Path(source_document).stem if source_document else ""
        parsed.append((source_document, fallback, text))
    return _build_catalog(parsed)


def parse_schema_directory(directory: str | Path) -> SchemaCatalog:
    schema_dir = Path(directory).expanduser().resolve()
    if not schema_dir.exists():
        raise FileNotFoundError(f"Schema directory does not exist: {schema_dir}")
    if not schema_dir.is_dir():
        raise NotADirectoryError(f"Schema path is not a directory: {schema_dir}")

    files = sorted(schema_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No Markdown schema files found in: {schema_dir}")

    parsed = [
        (path.name, path.stem, path.read_text(encoding="utf-8")) for path in files
    ]
    return _build_catalog(parsed)
