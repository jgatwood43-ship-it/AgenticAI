"""
Generate one Markdown schema document per MySQL table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.mcp_tools import mysql_query_tool

OUTPUT_DIRECTORY = Path("docs/schema")


TABLE_METADATA_SQL = """
SELECT
    c.TABLE_NAME,
    c.COLUMN_NAME,
    c.COLUMN_TYPE,
    c.IS_NULLABLE,
    c.COLUMN_KEY,
    c.COLUMN_COMMENT
FROM information_schema.COLUMNS AS c
WHERE c.TABLE_SCHEMA = DATABASE()
ORDER BY c.TABLE_NAME, c.ORDINAL_POSITION;
"""


RELATIONSHIP_SQL = """
SELECT
    kcu.TABLE_NAME AS child_table,
    kcu.COLUMN_NAME AS child_column,
    kcu.REFERENCED_TABLE_NAME AS parent_table,
    kcu.REFERENCED_COLUMN_NAME AS parent_column
FROM information_schema.KEY_COLUMN_USAGE AS kcu
WHERE kcu.CONSTRAINT_SCHEMA = DATABASE()
  AND kcu.REFERENCED_TABLE_NAME IS NOT NULL
ORDER BY
    kcu.TABLE_NAME,
    kcu.COLUMN_NAME;
"""


def extract_rows(raw_result: str) -> list[dict[str, Any]]:
    """
    Extract database rows from the JSON content returned by mysql_query_tool.

    Expected MCP response shape:

        [
            {
                "type": "text",
                "text": "{\"rows\": [{...}, {...}]}"
            }
        ]
    """

    try:
        outer_result = json.loads(raw_result)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not decode the outer MCP response: {exc}") from exc

    if not isinstance(outer_result, list):
        raise RuntimeError(
            "Expected the outer MCP response to be a list, "
            f"but received {type(outer_result).__name__}."
        )

    extracted_rows: list[dict[str, Any]] = []

    for content_block in outer_result:
        if not isinstance(content_block, dict):
            continue

        text_value = content_block.get("text")

        if not isinstance(text_value, str):
            continue

        try:
            inner_result = json.loads(text_value)
        except json.JSONDecodeError:
            continue

        if isinstance(inner_result, dict):
            rows = inner_result.get("rows", [])

            if isinstance(rows, list):
                extracted_rows.extend(row for row in rows if isinstance(row, dict))

        elif isinstance(inner_result, list):
            extracted_rows.extend(row for row in inner_result if isinstance(row, dict))

    if not extracted_rows:
        raise RuntimeError(
            "The MCP response was decoded, but no database rows were found."
        )

    return extracted_rows


def build_documents() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    columns = extract_rows(mysql_query_tool(TABLE_METADATA_SQL))
    print("COLUMN ROW COUNT:", len(columns))

    if columns:
        print("FIRST COLUMN ROW:", columns[0])
        print("AVAILABLE KEYS:", list(columns[0].keys()))
    relationships = extract_rows(mysql_query_tool(RELATIONSHIP_SQL))

    tables: dict[str, dict[str, Any]] = {}

    for column in columns:
        table_name = column["TABLE_NAME"]

        tables.setdefault(
            table_name,
            {
                "columns": [],
                "relationships": [],
            },
        )

        tables[table_name]["columns"].append(column)

    for relationship in relationships:
        child_table = relationship["child_table"]

        tables.setdefault(
            child_table,
            {
                "columns": [],
                "relationships": [],
            },
        )

        tables[child_table]["relationships"].append(relationship)

    for table_name, table_data in tables.items():
        lines = [
            f"# Table: {table_name}",
            "",
            "## Columns",
            "",
        ]

        for column in table_data["columns"]:
            key_text = ""

            if column["COLUMN_KEY"] == "PRI":
                key_text = ", primary key"
            elif column["COLUMN_KEY"] == "UNI":
                key_text = ", unique key"

            lines.append(
                f"- {column['COLUMN_NAME']}: "
                f"{column['COLUMN_TYPE']}"
                f"{key_text}, nullable={column['IS_NULLABLE']}"
            )

            if column.get("COLUMN_COMMENT"):
                lines.append(f"  - Description: {column['COLUMN_COMMENT']}")

        lines.extend(
            [
                "",
                "## Relationships",
                "",
            ]
        )

        if table_data["relationships"]:
            for relationship in table_data["relationships"]:
                lines.append(
                    f"- {relationship['child_table']}."
                    f"{relationship['child_column']} joins to "
                    f"{relationship['parent_table']}."
                    f"{relationship['parent_column']}"
                )
        else:
            lines.append("- No declared foreign-key relationships.")

        output_file = OUTPUT_DIRECTORY / f"{table_name}.md"
        output_file.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )


if __name__ == "__main__":
    build_documents()
