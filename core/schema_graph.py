"""
core/schema_graph.py
────────────────────
Deterministic schema relationship graph for the AgenticAI workflow.

The LLM should reason about WHAT evidence would be useful, but it should not
invent HOW database tables relate.

SchemaGraph turns the authoritative Markdown schema catalog into a deterministic
graph containing documented tables, documented columns, explicitly documented
relationships, valid direct edges, and valid multi-hop relationship paths.

No relationships are inferred from similar column names. The Markdown schema
remains the source of truth.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from core.schema_model import parse_schema_directory


@dataclass(frozen=True)
class SchemaColumn:
    table: str
    name: str
    description: str = ""
    source_document: str = ""

    @property
    def qualified_name(self) -> str:
        return f"{self.table}.{self.name}"


@dataclass(frozen=True)
class SchemaRelationship:
    left_table: str
    left_column: str = ""
    right_table: str = ""
    right_column: str = ""
    relationship_type: str = ""
    source_document: str = ""
    source_text: str = ""
    sql_capable: bool = True

    @property
    def left_field(self) -> str:
        return (
            f"{self.left_table}.{self.left_column}"
            if self.left_column
            else self.left_table
        )

    @property
    def right_field(self) -> str:
        return (
            f"{self.right_table}.{self.right_column}"
            if self.right_column
            else self.right_table
        )

    def connects(self, table_a: str, table_b: str) -> bool:
        a = _normalize_identifier(table_a)
        b = _normalize_identifier(table_b)
        return {self.left_table, self.right_table} == {a, b}


@dataclass
class SchemaTable:
    name: str
    source_document: str = ""
    columns: dict[str, SchemaColumn] = field(default_factory=dict)

    def has_column(self, column: str) -> bool:
        return _normalize_identifier(column) in self.columns


@dataclass(frozen=True)
class RelationshipPath:
    start_table: str
    end_table: str
    relationships: tuple[SchemaRelationship, ...]

    @property
    def hop_count(self) -> int:
        return len(self.relationships)

    @property
    def tables(self) -> tuple[str, ...]:
        if not self.relationships:
            return (self.start_table,)

        result = [self.start_table]
        current = self.start_table

        for relationship in self.relationships:
            if relationship.left_table == current:
                current = relationship.right_table
            elif relationship.right_table == current:
                current = relationship.left_table
            else:
                raise ValueError("Relationship path is internally disconnected.")

            result.append(current)

        return tuple(result)

    def render(self) -> str:
        if not self.relationships:
            return self.start_table

        pieces: list[str] = []
        current = self.start_table

        for relationship in self.relationships:
            if relationship.left_table == current:
                left_field = relationship.left_field
                right_field = relationship.right_field
                current = relationship.right_table
            else:
                left_field = relationship.right_field
                right_field = relationship.left_field
                current = relationship.left_table

            pieces.append(f"{left_field} -> {right_field}")

        return " | ".join(pieces)


def _normalize_identifier(value: str) -> str:
    return str(value or "").strip().strip("`").lower()


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").strip().split())


class SchemaGraph:
    """
    Deterministic graph of documented database relationships.

    Important:
        SchemaGraph never infers an edge from matching column names.
    """

    def __init__(
        self,
        *,
        tables: dict[str, SchemaTable],
        relationships: Iterable[SchemaRelationship],
        planning_relationships: Iterable[SchemaRelationship] = (),
        warnings: Iterable[str] = (),
    ) -> None:
        self.tables = {
            _normalize_identifier(table_name): table
            for table_name, table in tables.items()
        }
        # Strict SQL-capable / physical relationships.
        self.relationships = tuple(relationships)
        # Broader documented relationships used only to validate evidence-planning
        # connectivity. These are not certified SQL joins.
        self.planning_relationships = tuple(planning_relationships)
        self.warnings = list(warnings)

        self._adjacency: dict[str, list[SchemaRelationship]] = {
            table_name: [] for table_name in self.tables
        }

        for relationship in self.relationships:
            self._adjacency.setdefault(relationship.left_table, []).append(relationship)

            self._adjacency.setdefault(relationship.right_table, []).append(
                relationship
            )

        self._planning_adjacency: dict[str, list[SchemaRelationship]] = {
            table_name: [] for table_name in self.tables
        }
        for relationship in self.planning_relationships:
            self._planning_adjacency.setdefault(relationship.left_table, []).append(
                relationship
            )
            self._planning_adjacency.setdefault(relationship.right_table, []).append(
                relationship
            )

    @classmethod
    def from_directory(cls, directory: str | Path) -> "SchemaGraph":
        """Build the graph from the shared structural schema parser."""
        catalog = parse_schema_directory(directory)

        tables: dict[str, SchemaTable] = {}
        for table_name, columns in catalog.tables.items():
            first_record = next(
                iter(catalog.column_records.get(table_name, {}).values()),
                None,
            )
            tables[table_name] = SchemaTable(
                name=table_name,
                source_document=(first_record.source_document if first_record else ""),
                columns={
                    column_name: SchemaColumn(
                        table=table_name,
                        name=column_name,
                        description=(
                            catalog.column_records.get(table_name, {})
                            .get(column_name)
                            .description
                            if catalog.column_records.get(table_name, {}).get(
                                column_name
                            )
                            else ""
                        ),
                        source_document=(
                            catalog.column_records.get(table_name, {})
                            .get(column_name)
                            .source_document
                            if catalog.column_records.get(table_name, {}).get(
                                column_name
                            )
                            else ""
                        ),
                    )
                    for column_name in sorted(columns)
                },
            )

        relationships = [
            SchemaRelationship(
                left_table=record.left_table,
                left_column=record.left_column,
                right_table=record.right_table,
                right_column=record.right_column,
                relationship_type=record.relationship_type,
                source_document=record.source_document,
                source_text=record.source_text,
                sql_capable=True,
            )
            for record in catalog.relationship_records
        ]

        planning_relationships = [
            SchemaRelationship(
                left_table=record.left_table,
                right_table=record.right_table,
                relationship_type=record.relationship_type,
                source_document=record.source_document,
                source_text=record.source_text,
                sql_capable=False,
            )
            for record in catalog.planning_relationship_records
        ]

        return cls(
            tables=tables,
            relationships=relationships,
            planning_relationships=planning_relationships,
            warnings=catalog.warnings,
        )

    @staticmethod
    def _relationship_validation_error(
        *,
        relationship: SchemaRelationship,
        tables: dict[str, SchemaTable],
    ) -> str | None:
        if relationship.left_table not in tables:
            return (
                f"{relationship.source_document}: relationship references "
                f"undocumented table {relationship.left_table}."
            )

        if relationship.right_table not in tables:
            return (
                f"{relationship.source_document}: relationship references "
                f"undocumented table {relationship.right_table}."
            )

        if not tables[relationship.left_table].has_column(relationship.left_column):
            return (
                f"{relationship.source_document}: relationship references "
                f"undocumented column {relationship.left_field}."
            )

        if not tables[relationship.right_table].has_column(relationship.right_column):
            return (
                f"{relationship.source_document}: relationship references "
                f"undocumented column {relationship.right_field}."
            )

        return None

    def has_table(self, table: str) -> bool:
        return _normalize_identifier(table) in self.tables

    def has_column(self, table: str, column: str) -> bool:
        table_name = _normalize_identifier(table)

        if table_name not in self.tables:
            return False

        return self.tables[table_name].has_column(column)

    def table_names(self) -> list[str]:
        return sorted(self.tables)

    def column_names(self, table: str) -> list[str]:
        table_name = _normalize_identifier(table)

        if table_name not in self.tables:
            return []

        return sorted(self.tables[table_name].columns)

    def direct_relationships(
        self,
        table: str,
    ) -> list[SchemaRelationship]:
        return list(
            self._adjacency.get(
                _normalize_identifier(table),
                [],
            )
        )

    def relationships_between(
        self,
        table_a: str,
        table_b: str,
    ) -> list[SchemaRelationship]:
        a = _normalize_identifier(table_a)
        b = _normalize_identifier(table_b)

        return [
            relationship
            for relationship in self.relationships
            if relationship.connects(a, b)
        ]

    def has_direct_relationship(
        self,
        table_a: str,
        table_b: str,
    ) -> bool:
        return bool(
            self.relationships_between(
                table_a,
                table_b,
            )
        )

    def _shortest_path_from_adjacency(
        self,
        start_table: str,
        end_table: str,
        adjacency: dict[str, list[SchemaRelationship]],
        *,
        max_hops: int = 4,
    ) -> RelationshipPath | None:
        start = _normalize_identifier(start_table)
        end = _normalize_identifier(end_table)

        if start not in self.tables or end not in self.tables:
            return None
        if start == end:
            return RelationshipPath(start_table=start, end_table=end, relationships=())

        queue = deque([(start, tuple(), frozenset({start}))])
        while queue:
            current_table, current_path, visited_tables = queue.popleft()
            if len(current_path) >= max_hops:
                continue
            for relationship in adjacency.get(current_table, []):
                next_table = (
                    relationship.right_table
                    if relationship.left_table == current_table
                    else relationship.left_table
                )
                if next_table in visited_tables:
                    continue
                next_path = (*current_path, relationship)
                if next_table == end:
                    return RelationshipPath(
                        start_table=start,
                        end_table=end,
                        relationships=next_path,
                    )
                queue.append((next_table, next_path, visited_tables | {next_table}))
        return None

    def shortest_planning_path(
        self,
        start_table: str,
        end_table: str,
        *,
        max_hops: int = 4,
    ) -> RelationshipPath | None:
        """Return a path through documented planning relationships.

        A planning path proves only that the schema documentation says the two
        evidence domains are related. It does NOT certify a SQL join.
        """
        return self._shortest_path_from_adjacency(
            start_table,
            end_table,
            self._planning_adjacency,
            max_hops=max_hops,
        )

    def shortest_path(
        self,
        start_table: str,
        end_table: str,
        *,
        max_hops: int = 4,
    ) -> RelationshipPath | None:
        """Return a path using only strict SQL-capable relationships."""
        return self._shortest_path_from_adjacency(
            start_table,
            end_table,
            self._adjacency,
            max_hops=max_hops,
        )

    def reachable_tables(
        self,
        start_table: str,
        *,
        max_hops: int = 3,
    ) -> dict[str, RelationshipPath]:
        start = _normalize_identifier(start_table)

        if start not in self.tables:
            return {}

        results: dict[str, RelationshipPath] = {}

        queue = deque(
            [
                (
                    start,
                    tuple(),
                    frozenset({start}),
                )
            ]
        )

        while queue:
            current_table, current_path, visited_tables = queue.popleft()

            if len(current_path) >= max_hops:
                continue

            for relationship in self._adjacency.get(current_table, []):
                if relationship.left_table == current_table:
                    next_table = relationship.right_table
                else:
                    next_table = relationship.left_table

                if next_table in visited_tables:
                    continue

                next_path = (*current_path, relationship)

                if next_table not in results:
                    results[next_table] = RelationshipPath(
                        start_table=start,
                        end_table=next_table,
                        relationships=next_path,
                    )

                queue.append(
                    (
                        next_table,
                        next_path,
                        visited_tables | {next_table},
                    )
                )

        return results

    def render_relationship_catalog(self) -> str:
        if not self.relationships:
            return "No explicitly documented schema relationships were parsed."

        lines = ["DOCUMENTED DIRECT RELATIONSHIPS"]

        for relationship in sorted(
            self.relationships,
            key=lambda item: (
                item.left_table,
                item.left_column,
                item.right_table,
                item.right_column,
            ),
        ):
            lines.append(f"- {relationship.left_field} -> {relationship.right_field}")

        return "\n".join(lines)

    def render_planning_relationship_catalog(self) -> str:
        if not self.planning_relationships:
            return "No documented evidence-planning relationships were parsed."

        lines = [
            "DOCUMENTED EVIDENCE-PLANNING RELATIONSHIPS",
            "These relationships establish planning connectivity only; they do not certify SQL joins.",
        ]
        for relationship in sorted(
            self.planning_relationships,
            key=lambda item: (
                item.left_table,
                item.right_table,
                item.relationship_type,
            ),
        ):
            type_text = relationship.relationship_type or "documented relationship"
            detail = f": {relationship.source_text}" if relationship.source_text else ""
            lines.append(
                f"- {relationship.left_table} <-> {relationship.right_table} "
                f"[{type_text}]{detail}"
            )
        return "\n".join(lines)

    def render_table_catalog(self) -> str:
        lines: list[str] = []

        for table_name in self.table_names():
            table = self.tables[table_name]
            lines.append(f"TABLE: {table_name}")

            for column_name in sorted(table.columns):
                column = table.columns[column_name]

                if column.description:
                    lines.append(f"  - {column.name}: {column.description}")
                else:
                    lines.append(f"  - {column.name}")

            lines.append("")

        return "\n".join(lines).strip()

    def render_paths_from(
        self,
        start_table: str,
        *,
        max_hops: int = 3,
    ) -> str:
        start = _normalize_identifier(start_table)
        paths = self.reachable_tables(start, max_hops=max_hops)

        if not paths:
            return f"No documented relationship paths from {start}."

        lines = [f"DOCUMENTED PATHS FROM {start}"]

        for end_table in sorted(paths):
            path = paths[end_table]

            lines.append(
                f"- {start} -> {end_table} "
                f"({path.hop_count} hop"
                f"{'' if path.hop_count == 1 else 's'}): "
                f"{path.render()}"
            )

        return "\n".join(lines)

    def summary(self) -> str:
        return "\n".join(
            [
                "SchemaGraph",
                f"Tables        : {len(self.tables)}",
                f"SQL relationships      : {len(self.relationships)}",
                f"Planning relationships : {len(self.planning_relationships)}",
                f"Warnings               : {len(self.warnings)}",
            ]
        )


def main() -> None:
    """
    Optional local diagnostic:

        python core/schema_graph.py
    """
    project_root = Path(__file__).resolve().parents[1]
    schema_dir = project_root / "docs" / "schema"

    graph = SchemaGraph.from_directory(schema_dir)

    print(graph.summary())
    print("\n" + graph.render_relationship_catalog())

    if graph.warnings:
        print("\nWARNINGS")
        for warning in graph.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
