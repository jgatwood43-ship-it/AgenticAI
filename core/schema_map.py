"""
Build and cache a compact MySQL schema map for agent use.

The schema map contains:
- tables
- columns
- primary keys
- foreign-key relationships
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from tools.mcp_tools import mysql_query_tool

TABLES_SQL = """
SELECT
    TABLE_NAME
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = DATABASE()
ORDER BY TABLE_NAME;
"""


COLUMNS_SQL = """
SELECT
    TABLE_NAME,
    COLUMN_NAME,
    DATA_TYPE,
    IS_NULLABLE,
    COLUMN_KEY,
    EXTRA
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
ORDER BY TABLE_NAME, ORDINAL_POSITION;
"""


RELATIONSHIPS_SQL = """
SELECT
    kcu.TABLE_NAME AS child_table,
    kcu.COLUMN_NAME AS child_column,
    kcu.REFERENCED_TABLE_NAME AS parent_table,
    kcu.REFERENCED_COLUMN_NAME AS parent_column,
    kcu.CONSTRAINT_NAME AS constraint_name
FROM information_schema.KEY_COLUMN_USAGE AS kcu
WHERE kcu.CONSTRAINT_SCHEMA = DATABASE()
  AND kcu.REFERENCED_TABLE_NAME IS NOT NULL
ORDER BY
    kcu.TABLE_NAME,
    kcu.COLUMN_NAME;
"""


def _decode_mcp_result(raw_result: str) -> Any:
    """
    Convert the JSON string returned by mysql_query_tool into Python data.

    The exact shape may need adjustment depending on how dbmcp formats
    content blocks.
    """
    try:
        return json.loads(raw_result)
    except json.JSONDecodeError:
        raise RuntimeError(
            "Could not decode the MySQL MCP response while building the schema map."
        )


def build_schema_map() -> dict[str, Any]:
    """
    Query MySQL metadata and build a reusable schema map.
    """

    raw_tables = mysql_query_tool(TABLES_SQL)
    raw_columns = mysql_query_tool(COLUMNS_SQL)
    raw_relationships = mysql_query_tool(RELATIONSHIPS_SQL)

    tables_result = _decode_mcp_result(raw_tables)
    columns_result = _decode_mcp_result(raw_columns)
    relationships_result = _decode_mcp_result(raw_relationships)

    schema_map: dict[str, Any] = {
        "tables": {},
        "relationships": [],
    }

    # These extraction helpers may need adjustment based on your MCP response.
    table_rows = _extract_rows(tables_result)
    column_rows = _extract_rows(columns_result)
    relationship_rows = _extract_rows(relationships_result)

    for row in table_rows:
        table_name = row["TABLE_NAME"]

        schema_map["tables"][table_name] = {
            "columns": [],
            "primary_keys": [],
            "relationships": [],
        }

    for row in column_rows:
        table_name = row["TABLE_NAME"]

        if table_name not in schema_map["tables"]:
            schema_map["tables"][table_name] = {
                "columns": [],
                "primary_keys": [],
                "relationships": [],
            }

        column = {
            "name": row["COLUMN_NAME"],
            "data_type": row["DATA_TYPE"],
            "nullable": row["IS_NULLABLE"] == "YES",
            "key": row["COLUMN_KEY"],
            "extra": row["EXTRA"],
        }

        schema_map["tables"][table_name]["columns"].append(column)

        if row["COLUMN_KEY"] == "PRI":
            schema_map["tables"][table_name]["primary_keys"].append(row["COLUMN_NAME"])

    for row in relationship_rows:
        relationship = {
            "child_table": row["child_table"],
            "child_column": row["child_column"],
            "parent_table": row["parent_table"],
            "parent_column": row["parent_column"],
            "constraint_name": row["constraint_name"],
        }

        schema_map["relationships"].append(relationship)

        child_table = row["child_table"]

        if child_table in schema_map["tables"]:
            schema_map["tables"][child_table]["relationships"].append(relationship)

    return schema_map


def _extract_rows(result: Any) -> list[dict[str, Any]]:
    """
    Extract database rows from a dbmcp response.

    Update this function after inspecting one real mysql_query_tool response.
    """

    if isinstance(result, list):
        # Some MCP servers return content blocks.
        if result and isinstance(result[0], dict):
            if "text" in result[0]:
                try:
                    decoded_text = json.loads(result[0]["text"])

                    if isinstance(decoded_text, list):
                        return decoded_text
                except json.JSONDecodeError:
                    pass

            # Some servers return rows directly.
            if all(isinstance(item, dict) for item in result):
                return result

    if isinstance(result, dict):
        if "rows" in result and isinstance(result["rows"], list):
            return result["rows"]

        if "data" in result and isinstance(result["data"], list):
            return result["data"]

    raise RuntimeError("Unrecognized MCP response format while extracting schema rows.")


@lru_cache(maxsize=1)
def get_schema_map() -> dict[str, Any]:
    """
    Build the schema map once and reuse it for later agent requests.
    """

    return build_schema_map()


def refresh_schema_map() -> dict[str, Any]:
    """
    Clear the cached schema and rebuild it.

    Call this when the database structure changes.
    """

    get_schema_map.cache_clear()
    return get_schema_map()
