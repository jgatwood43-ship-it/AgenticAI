"""
core/sql_planner.py
Schema-grounded logical planning with deterministic SQL compilation.

The LLM produces only a LogicalQueryPlan. Python validates and compiles it.
"""

from __future__ import annotations

import re
from typing import Any
from llama_index.core.llms import LLM

from core.logical_plan import LogicalQueryPlan
from core.sql_compiler import SchemaCatalog, compile_logical_plan
from core.state import SQLPlan

_SCHEMA_DOCUMENT_PATTERN = re.compile(
    r"#\s*Table:\s*([A-Za-z_][A-Za-z0-9_]*)"
    r"(.*?)(?=(?:\n=+\nSCHEMA DOCUMENT:)|(?:\n#\s*Table:)|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)
_COLUMN_PATTERN = re.compile(
    r"^\s*-\s*([A-Za-z_][A-Za-z0-9_]*)\s*:",
    flags=re.MULTILINE,
)
_RELATIONSHIP_PATTERN = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)"
    r"\s+(?:joins\s+to|=|->|→)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)",
    flags=re.IGNORECASE,
)

_LOGICAL_PLAN_PROMPT = """
You are a logical database-query planner for a cybersecurity user-access
analysis system. You do not write SQL. Return only a LogicalQueryPlan.

USER QUESTION
-------------
{query}

AUTHORITATIVE SCHEMA
--------------------
{schema_context}

PRIOR VALIDATION FEEDBACK
-------------------------
{feedback}

RULES
-----
1. Use only documented tables, columns, relationships, and business values.
2. Never invent dates, identifiers, values, aliases, SQL, CTEs, or functions.
3. Model the request with selected fields, joins, filters, aggregations,
   grouping, ordering, temporal operations, and derived predicates.
4. Use latest_before when state at an event timestamp depends on the most
   recent earlier event.
5. For latest_before, correlate the historical source table to the reference
   event using documented identity keys and source_timestamp <= reference_time.
6. Values may be used only when explicitly present in the question or schema.
7. If a required business value is missing, set is_supported=false and explain
   it in missing_information.
8. selected_fields and group_by must use table.column notation.
9. required_tables must include every table needed by the plan.
10. Prefer one precise logical plan over multiple guesses.
"""


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _extract_structured_response(response: Any) -> LogicalQueryPlan:
    raw = getattr(response, "raw", None)
    if isinstance(raw, LogicalQueryPlan):
        return raw
    if isinstance(raw, dict):
        return LogicalQueryPlan.model_validate(raw)
    if isinstance(response, LogicalQueryPlan):
        return response
    parsed = (getattr(response, "additional_kwargs", {}) or {}).get("parsed")
    if isinstance(parsed, LogicalQueryPlan):
        return parsed
    if isinstance(parsed, dict):
        return LogicalQueryPlan.model_validate(parsed)
    text = _clean_text(getattr(response, "text", response))
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    if not text:
        raise ValueError("The logical planner returned an empty response.")
    return LogicalQueryPlan.model_validate_json(text)


def _parse_schema_catalog(schema_context: str) -> SchemaCatalog:
    tables: dict[str, set[str]] = {}
    relationships: set[tuple[str, str, str, str]] = set()
    for match in _SCHEMA_DOCUMENT_PATTERN.finditer(schema_context):
        table = match.group(1).lower()
        body = match.group(2) or ""
        tables.setdefault(table, set()).update(
            column.lower() for column in _COLUMN_PATTERN.findall(body)
        )
        for relationship in _RELATIONSHIP_PATTERN.findall(body):
            relationships.add(tuple(item.lower() for item in relationship))
    return SchemaCatalog(tables=tables, relationships=relationships)


def select_schema_documents(query: str) -> list[str]:
    query_lower = _clean_text(query).lower()
    selected: list[str] = []

    def include(*names: str) -> None:
        for name in names:
            if name not in selected:
                selected.append(name)

    if any(term in query_lower for term in ("employee", "employees", "staff", "user")):
        include("employees")
    if any(
        term in query_lower
        for term in ("badge", "access", "entered", "entry", "room", "door")
    ):
        include("employees", "badge_access_log", "badge_access_rules", "rooms")
    if any(
        term in query_lower
        for term in ("clock", "clocked", "punch", "attendance", "shift")
    ):
        include(
            "employees", "time_clock", "time_clock_summary", "employee_status_history"
        )
    if "department" in query_lower:
        include("employees", "departments")
    if any(term in query_lower for term in ("job title", "position", "role")):
        include("employees", "job_titles")
    if any(term in query_lower for term in ("key", "physical key")):
        include("employees", "key_access_log", "key_inventory", "rooms")
    if any(term in query_lower for term in ("camera", "video", "footage")):
        include("video_cameras", "video_footage", "rooms")
    return selected


def _build_deterministic_simple_plan(
    query: str, schema: SchemaCatalog
) -> SQLPlan | None:
    query_lower = query.lower()
    if (
        "employee" in query_lower
        and "first" in query_lower
        and "last" in query_lower
        and "name" in query_lower
        and schema.has_column("employees", "first_name")
        and schema.has_column("employees", "last_name")
    ):
        return SQLPlan(
            purpose="Return all employee first and last names.",
            database_type="mysql",
            tables=["employees"],
            columns=["employees.first_name", "employees.last_name"],
            sql=(
                "SELECT\n    e.first_name,\n    e.last_name\n"
                "FROM employees AS e\n"
                "ORDER BY\n    e.last_name,\n    e.first_name;"
            ),
            is_valid=True,
        )
    return None


class SQLPlanner:
    """Produce a logical plan with the LLM and compile it deterministically."""

    def __init__(self, llm: LLM) -> None:
        self._logical_llm = llm.as_structured_llm(LogicalQueryPlan)

    def plan(
        self, query: str, schema_context: str, feedback: list[str] | None = None
    ) -> SQLPlan:
        cleaned_query = _clean_text(query)
        cleaned_schema = _clean_text(schema_context)
        feedback_items = [
            str(item).strip() for item in (feedback or []) if str(item).strip()
        ]
        if not cleaned_query:
            return SQLPlan(
                is_valid=False, validation_errors=["The user query is empty."]
            )
        if not cleaned_schema:
            return SQLPlan(
                is_valid=False,
                validation_errors=["No authoritative schema context was supplied."],
            )
        schema = _parse_schema_catalog(cleaned_schema)
        if not schema.tables:
            return SQLPlan(
                is_valid=False,
                validation_errors=["No table definitions could be parsed."],
            )
        deterministic = _build_deterministic_simple_plan(cleaned_query, schema)
        if deterministic is not None and not feedback_items:
            print("\n" + "═" * 70)
            print("[SQLPlanner] ▶ DETERMINISTIC SIMPLE PLAN")
            print(deterministic.sql)
            print("═" * 70 + "\n")
            return deterministic
        feedback_text = (
            "\n".join(f"- {item}" for item in feedback_items)
            if feedback_items
            else "(none)"
        )
        print("\n" + "═" * 70)
        print("[SQLPlanner] ▶ LOGICAL PLAN + DETERMINISTIC COMPILATION")
        print(f"[SQLPlanner]    Query        : {cleaned_query}")
        print(f"[SQLPlanner]    Schema chars : {len(cleaned_schema)}")
        print(f"[SQLPlanner]    Feedback     : {len(feedback_items)} item(s)")
        print("─" * 70)
        try:
            response = self._logical_llm.complete(
                _LOGICAL_PLAN_PROMPT.format(
                    query=cleaned_query,
                    schema_context=cleaned_schema,
                    feedback=feedback_text,
                )
            )
            logical_plan = _extract_structured_response(response)
        except Exception as exc:
            error = f"Logical planning failed: {type(exc).__name__}: {exc}"
            print(f"[SQLPlanner] ✘ {error}")
            print("═" * 70 + "\n")
            return SQLPlan(is_valid=False, validation_errors=[error])
        compiled = compile_logical_plan(logical_plan, schema)
        print(f"[SQLPlanner]    Supported    : {logical_plan.is_supported}")
        print(f"[SQLPlanner]    Tables       : {compiled.tables}")
        print(f"[SQLPlanner]    Valid        : {compiled.is_valid}")
        for error in compiled.validation_errors:
            print(f"[SQLPlanner]      - {error}")
        if compiled.sql:
            print("[SQLPlanner]    SQL:")
            print(compiled.sql)
        print("═" * 70 + "\n")
        return compiled

    def repair(
        self, query: str, schema_context: str, prior_plan: SQLPlan, errors: list[str]
    ) -> SQLPlan:
        feedback = [*errors, *(prior_plan.validation_errors if prior_plan else [])]
        return self.plan(query=query, schema_context=schema_context, feedback=feedback)
