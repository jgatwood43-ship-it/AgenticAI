"""
core/direct_query_planner.py
────────────────────────────
Generate, self-review, and hard-validate read-only SQL for ordinary questions.

Architecture
------------

    LLM generates SQL from user question + authoritative schema
            ↓
    SAME LLM reviews the candidate SQL for semantic completeness
            ↓
    Python performs hard deterministic validation
            ↓
    if hard validation fails, up to three bounded schema/safety corrections are allowed
            ↓
    SAME LLM self-reviews the corrected SQL
            ↓
    Python validates again
            ↓
    approved SQL may be executed by RetrieverAgent

Python does not decide how to implement recency, aggregation, grouping,
cardinality, joins, or other SQL reasoning choices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import re
from typing import Any

from llama_index.core.llms import LLM

from core.direct_sql_validator import (
    DirectSQLValidationResult,
    DirectSQLValidator,
)

MAX_HARD_VALIDATION_REPAIRS = 3


class DirectQueryStatus(str, Enum):
    APPROVED = "approved"
    EMPTY_QUERY = "empty_query"
    EMPTY_SCHEMA = "empty_schema"
    VALIDATION_FAILED = "validation_failed"
    GENERATION_ERROR = "generation_error"


@dataclass
class DirectQueryPlan:
    approved: bool = False
    status: DirectQueryStatus = DirectQueryStatus.GENERATION_ERROR
    question: str = ""
    sql: str = ""
    validation: DirectSQLValidationResult | None = None
    validation_repairs: int = 0
    attempts: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def tables(self) -> list[str]:
        return list(self.validation.tables) if self.validation else []

    @property
    def columns(self) -> list[str]:
        return list(self.validation.columns) if self.validation else []

    @property
    def relationships(self) -> list[str]:
        return list(self.validation.relationships) if self.validation else []

    @property
    def validation_errors(self) -> list[str]:
        return list(self.validation.errors) if self.validation else []


_SQL_GENERATION_PROMPT = """
You are a MySQL query-generation assistant.

Generate exactly one read-only SQL query that answers the USER QUESTION using
the AUTHORITATIVE DATABASE SCHEMA.

AUTHORITATIVE DATABASE SCHEMA
-----------------------------
{schema_context}

USER QUESTION
-------------
{question}

RULES
-----
1. Answer the USER QUESTION completely.
2. Preserve the requested result meaning and result shape.
3. Use only documented tables, columns, and relationships.
   A column belongs only to the table under which it is documented.
   Never combine a table name with a column documented under another table.
4. Never invent schema elements, relationships, business values, or facts.
5. Use only read-only SELECT/query SQL.
6. Choose the SQL construction yourself.
7. Prefer the simplest correct query.
8. Return exactly one query and nothing else.
"""


_SQL_SELF_REVIEW_PROMPT = """
You are reviewing SQL that you just generated before it is allowed to execute.

USER QUESTION
-------------
{question}

AUTHORITATIVE DATABASE SCHEMA
-----------------------------
{schema_context}

CANDIDATE SQL
-------------
{candidate_sql}

KNOWN HARD-VALIDATION FEEDBACK
------------------------------
{validation_feedback}

REVIEW THE CANDIDATE
--------------------
Determine whether the SQL actually answers the USER QUESTION completely and
faithfully.

Review the reasoning represented by the SQL, including:
- whether the selected records and fields answer what the user actually asked;
- whether each documented table and relationship is being used appropriately;
- whether the result identifies the requested entities clearly enough;
- whether requested timing, ordering, comparison, aggregation, cardinality,
  or correlation semantics are actually satisfied;
- whether unnecessary joins or restrictions change the requested population;
- whether any table, column, relationship, or business value was invented;
- whether the query substitutes a different operational concept for the one
  requested by the user;
- whether the query introduces any date range, threshold, status value,
  category, or other business assumption that was not supplied by the user or
  documented in the authoritative schema/business rules;
- whether the candidate corrected every hard-validation problem already
  identified without abandoning the original user question.

Do not treat a merely schema-valid query as semantically correct.
Do not follow a fixed SQL recipe. Decide for yourself whether the construction
is correct for this question and schema.

If the candidate is correct, return it unchanged.
If it is not correct, return one corrected read-only MySQL query.

Return SQL only. Do not explain the review and do not use Markdown fences.
"""


_SQL_HARD_REPAIR_PROMPT = """
A read-only MySQL query failed deterministic schema/safety validation.

USER QUESTION
-------------
{question}

AUTHORITATIVE DATABASE SCHEMA
-----------------------------
{schema_context}

HARD VALIDATION ERRORS
----------------------
{validation_errors}

Generate a new query from the USER QUESTION and authoritative schema.

Correct every hard validation error. Do not invent schema elements,
relationships, business values, or facts.
A column belongs only to its documented table; do not move a documented
column name onto a different table.

Choose the SQL construction yourself.

Return exactly one read-only MySQL query and nothing else.
"""


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


_TABLE_BLOCK_PATTERN = re.compile(
    r"(?ms)^#\s*Table:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\n"
    r"(.*?)(?=^#\s*Table:\s*[A-Za-z_][A-Za-z0-9_]*\s*$|\Z)"
)

_COLUMN_LINE_PATTERN = re.compile(
    r"^\s*-\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)\s*$",
    flags=re.MULTILINE,
)

_PURPOSE_PATTERN = re.compile(r"(?ms)^##\s*Purpose\s*\n(.*?)(?=^##\s+|\Z)")

_ROW_GRAIN_PATTERN = re.compile(r"(?ms)^##\s*Row Grain\s*\n(.*?)(?=^##\s+|\Z)")

_RELATIONSHIP_SECTION_PATTERN = re.compile(
    r"(?ms)^##\s*(Outbound Relationships|Inbound Relationships|Relationships)\s*\n"
    r"(.*?)(?=^##\s+|\Z)"
)


def _compact_multiline(value: str) -> str:
    """Collapse Markdown prose into one readable line without changing meaning."""
    return " ".join(
        line.strip() for line in _clean_text(value).splitlines() if line.strip()
    )


def _build_fully_qualified_schema_context(
    schema_context: str,
) -> str:
    """
    Convert authoritative Markdown schema into a table-column-binding view.

    Python performs formatting only. It does not select relevant tables,
    infer business meaning, or decide how SQL should be written.

    Every documented column is repeated as table.column so the LLM cannot
    accidentally combine a table name with a column owned by another table.
    """
    source = _clean_text(schema_context)

    if not source:
        return ""

    blocks: list[str] = []

    for match in _TABLE_BLOCK_PATTERN.finditer(source):
        table_name = match.group(1).strip()
        body = match.group(2)

        purpose_match = _PURPOSE_PATTERN.search(body)
        grain_match = _ROW_GRAIN_PATTERN.search(body)

        purpose = (
            _compact_multiline(purpose_match.group(1))
            if purpose_match is not None
            else ""
        )

        row_grain = (
            _compact_multiline(grain_match.group(1)) if grain_match is not None else ""
        )

        columns = [
            (column_name.strip(), definition.strip())
            for column_name, definition in _COLUMN_LINE_PATTERN.findall(body)
        ]

        relationships: list[str] = []

        for section_match in _RELATIONSHIP_SECTION_PATTERN.finditer(body):
            section_body = section_match.group(2)

            for line in section_body.splitlines():
                cleaned = line.strip()

                if not cleaned.startswith("-"):
                    continue

                relationship = cleaned[1:].strip()

                if (
                    relationship
                    and relationship.lower() not in {"none", "none documented."}
                    and relationship not in relationships
                ):
                    relationships.append(relationship)

        lines = [f"TABLE: {table_name}"]

        if purpose:
            lines.append(f"PURPOSE: {purpose}")

        if row_grain:
            lines.append(f"ROW GRAIN: {row_grain}")

        lines.append("ONLY DOCUMENTED COLUMNS:")

        if columns:
            for column_name, definition in columns:
                lines.append(f"  - {table_name}.{column_name}: {definition}")
        else:
            lines.append("  - (none documented)")

        lines.append("DOCUMENTED RELATIONSHIPS:")

        if relationships:
            for relationship in relationships:
                lines.append(f"  - {relationship}")
        else:
            lines.append("  - (none documented)")

        blocks.append("\n".join(lines))

    if not blocks:
        return source

    header = (
        "AUTHORITATIVE SCHEMA — FULLY QUALIFIED COLUMN OWNERSHIP\n"
        "A column belongs only to the table under which it is documented.\n"
        "Do not combine a table name with a column documented under another table."
    )

    return header + "\n\n" + "\n\n".join(blocks)


def _extract_sql_candidate(value: Any) -> str:
    text = _clean_text(value)

    if not text:
        return ""

    fenced = re.search(
        r"```(?:sql|mysql)?\s*(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if fenced:
        text = fenced.group(1).strip()

    text = re.sub(
        r"^\s*SQL\s*:\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    ).strip()

    start = re.search(
        r"^[ \t]*(?:SELECT|WITH)\b",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    if start:
        text = text[start.start() :].strip()

    semicolon = text.find(";")

    if semicolon >= 0:
        text = text[: semicolon + 1]

    return text.strip()


def build_direct_query_generation_prompt(
    *,
    question: str,
    schema_context: str,
) -> str:
    qualified_schema = _build_fully_qualified_schema_context(schema_context)

    return _SQL_GENERATION_PROMPT.format(
        question=_clean_text(question),
        schema_context=qualified_schema,
    )


def direct_query_prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(_clean_text(prompt).encode("utf-8")).hexdigest()


def direct_query_schema_sha256(schema_context: str) -> str:
    return hashlib.sha256(_clean_text(schema_context).encode("utf-8")).hexdigest()


def _validation_feedback_text(
    validation: DirectSQLValidationResult,
) -> str:
    """Render only factual hard-validation findings for LLM review."""
    if validation is None or not validation.errors:
        return "(none)"

    return "\n".join(
        f"- {_clean_text(error)}" for error in validation.errors if _clean_text(error)
    )


def _validation_signature(
    validation: DirectSQLValidationResult,
) -> tuple[str, ...]:
    """Return a stable signature for repeated hard-validation states."""
    if validation is None:
        return ()

    return tuple(
        sorted(
            {
                _clean_text(error).lower()
                for error in validation.errors
                if _clean_text(error)
            }
        )
    )


class DirectQueryPlanner:
    """LLM SQL reasoning + LLM self-review + deterministic hard validation."""

    def __init__(
        self,
        *,
        llm: LLM,
        validator: DirectSQLValidator,
    ) -> None:
        self._llm = llm
        self._validator = validator

        print("\n[DirectQueryPlanner] ⚙  Initialising direct SQL planner…")
        print("[DirectQueryPlanner]    Generation : LLM SQL reasoning")
        print("[DirectQueryPlanner]    Self-review: same LLM reviews generated SQL")
        print("[DirectQueryPlanner]    Validation : hard schema + safety only")
        print("[DirectQueryPlanner]    Hard repair: maximum three bounded corrections")
        print("[DirectQueryPlanner]    Execution  : none")

    def _complete(self, prompt: str) -> str:
        response = self._llm.complete(prompt)

        raw_text = _clean_text(getattr(response, "text", response))

        sql = _extract_sql_candidate(raw_text)

        if sql != raw_text:
            print(
                "[DirectQueryPlanner]    Formatting cleanup: "
                "extracted SQL from LLM response."
            )

        return sql

    def _generate(
        self,
        *,
        question: str,
        schema_context: str,
    ) -> str:
        prompt = build_direct_query_generation_prompt(
            question=question,
            schema_context=schema_context,
        )

        print(f"[DirectQueryPlanner]    Prompt chars : {len(prompt)}")
        print(
            "[DirectQueryPlanner]    Prompt SHA256: "
            f"{direct_query_prompt_sha256(prompt)}"
        )
        qualified_schema = _build_fully_qualified_schema_context(schema_context)

        print(
            "[DirectQueryPlanner]    Raw schema SHA256: "
            f"{direct_query_schema_sha256(schema_context)}"
        )
        print(
            "[DirectQueryPlanner]    Qualified schema chars: "
            f"{len(qualified_schema)}"
        )
        print(
            "[DirectQueryPlanner]    Qualified schema SHA256: "
            f"{direct_query_schema_sha256(qualified_schema)}"
        )

        return self._complete(prompt)

    def _review(
        self,
        *,
        question: str,
        schema_context: str,
        candidate_sql: str,
        validation_feedback: str = "",
    ) -> str:
        print("[DirectQueryPlanner]    Requesting LLM self-review…")

        feedback = _clean_text(validation_feedback)

        if not feedback:
            feedback = "(none — this is the initial semantic review)"

        qualified_schema = _build_fully_qualified_schema_context(schema_context)

        return self._complete(
            _SQL_SELF_REVIEW_PROMPT.format(
                question=question,
                schema_context=qualified_schema,
                candidate_sql=candidate_sql,
                validation_feedback=feedback,
            )
        )

    def _hard_repair(
        self,
        *,
        question: str,
        schema_context: str,
        validation: DirectSQLValidationResult,
    ) -> str:
        feedback = (
            "\n".join(f"- {error}" for error in validation.errors)
            or "- Query failed hard validation."
        )

        qualified_schema = _build_fully_qualified_schema_context(schema_context)

        return self._complete(
            _SQL_HARD_REPAIR_PROMPT.format(
                question=question,
                schema_context=qualified_schema,
                validation_errors=feedback,
            )
        )

    @staticmethod
    def _print_validation(
        validation: DirectSQLValidationResult,
    ) -> None:
        print(
            "[DirectQueryPlanner]    Hard validation: "
            f"{'PASS' if validation.is_valid else 'FAIL'}"
        )
        print(f"[DirectQueryPlanner]    Tables     : {validation.tables}")
        print(f"[DirectQueryPlanner]    Columns    : {validation.columns}")
        print(f"[DirectQueryPlanner]    Relations  : {validation.relationships}")

        for error in validation.errors:
            print(f"[DirectQueryPlanner]      ERROR : {error}")

        for warning in validation.warnings:
            print(f"[DirectQueryPlanner]      WARN  : {warning}")

    def plan(
        self,
        *,
        question: str,
        schema_context: str,
    ) -> DirectQueryPlan:
        question = _clean_text(question)
        schema_context = _clean_text(schema_context)

        result = DirectQueryPlan(question=question)

        if not question:
            result.status = DirectQueryStatus.EMPTY_QUERY
            result.error = "DirectQuery question is empty."
            return result

        if not schema_context:
            result.status = DirectQueryStatus.EMPTY_SCHEMA
            result.error = "DirectQuery requires authoritative schema context."
            return result

        print("\n[DirectQueryPlanner] Planning direct query…")
        print(f"[DirectQueryPlanner]    Question : {question}")
        print(f"[DirectQueryPlanner]    Schema   : {len(schema_context)} chars")

        try:
            generated_sql = self._generate(
                question=question,
                schema_context=schema_context,
            )

            result.attempts.append(generated_sql)

            print("[DirectQueryPlanner]    Generated SQL:")
            print(generated_sql)

            sql = self._review(
                question=question,
                schema_context=schema_context,
                candidate_sql=generated_sql,
                validation_feedback="",
            )

            result.attempts.append(sql)

            print("[DirectQueryPlanner]    Self-reviewed SQL:")
            print(sql)

        except Exception as exc:
            result.status = DirectQueryStatus.GENERATION_ERROR
            result.error = (
                "DirectQuery generation/self-review failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return result

        validation = self._validator.validate(sql)
        result.sql = sql
        result.validation = validation
        self._print_validation(validation)

        seen_validation_states: set[tuple[str, ...]] = set()

        initial_signature = _validation_signature(validation)

        if initial_signature:
            seen_validation_states.add(initial_signature)

        while (
            not validation.is_valid
            and validation.is_repairable
            and result.validation_repairs < MAX_HARD_VALIDATION_REPAIRS
        ):
            result.validation_repairs += 1

            print(
                "[DirectQueryPlanner]    Hard validation failed; "
                "requesting bounded LLM correction "
                f"{result.validation_repairs}/{MAX_HARD_VALIDATION_REPAIRS}…"
            )

            feedback = _validation_feedback_text(validation)

            try:
                repaired = self._hard_repair(
                    question=question,
                    schema_context=schema_context,
                    validation=validation,
                )

                result.attempts.append(repaired)

                repaired = self._review(
                    question=question,
                    schema_context=schema_context,
                    candidate_sql=repaired,
                    validation_feedback=feedback,
                )

                result.attempts.append(repaired)

                print(
                    "[DirectQueryPlanner]    Reviewed repaired SQL "
                    f"{result.validation_repairs}:"
                )
                print(repaired)

                validation = self._validator.validate(repaired)
                result.sql = repaired
                result.validation = validation
                self._print_validation(validation)

            except Exception as exc:
                result.status = DirectQueryStatus.GENERATION_ERROR
                result.error = (
                    "DirectQuery hard-repair/self-review failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                return result

            if validation.is_valid:
                break

            signature = _validation_signature(validation)

            if signature and signature in seen_validation_states:
                print(
                    "[DirectQueryPlanner]    Repair stopped: "
                    "hard-validation state repeated; cycle detected."
                )
                break

            if signature:
                seen_validation_states.add(signature)

        if not validation.is_valid:
            result.status = DirectQueryStatus.VALIDATION_FAILED
            result.error = (
                "DirectQuery SQL failed hard deterministic validation: "
                + "; ".join(validation.errors or ["No validation reason was supplied."])
            )
            return result

        result.approved = True
        result.status = DirectQueryStatus.APPROVED
        result.error = ""

        print("[DirectQueryPlanner]    APPROVED for read-only execution.")
        print("[DirectQueryPlanner]    Hard repairs: " f"{result.validation_repairs}")

        return result


def _self_test_prompts() -> None:
    generation = build_direct_query_generation_prompt(
        question="Return the requested records.",
        schema_context="# Table: records",
    )

    assert "Choose the SQL construction yourself" in generation

    review = _SQL_SELF_REVIEW_PROMPT.format(
        question="Return the requested records.",
        schema_context="# Table: records",
        candidate_sql="SELECT value FROM records",
        validation_feedback="(none)",
    )

    assert "reviewing SQL that you just generated" in review
    assert "Do not follow a fixed SQL recipe" in review
    assert "different operational concept" in review
    assert "date range, threshold, status value" in review

    print("DirectQueryPlanner prompt self-test: PASS")


def _self_test_fully_qualified_schema() -> None:
    """Verify Markdown schema becomes explicit table.column ownership."""
    sample = """
# Table: event_log

## Purpose

Stores individual events.

## Row Grain

One row per event.

## Columns

- event_id: int, primary key
- event_time: datetime

## Outbound Relationships

- event_log.event_id joins to event_summary.event_id

## Inbound Relationships

- None documented.

# Table: event_summary

## Columns

- event_id: int
- summary_time: datetime

## Outbound Relationships

- None documented.

## Inbound Relationships

- event_log.event_id joins to event_summary.event_id
""".strip()

    qualified = _build_fully_qualified_schema_context(sample)

    assert "event_log.event_id" in qualified
    assert "event_log.event_time" in qualified
    assert "event_summary.summary_time" in qualified
    assert "A column belongs only to the table" in qualified

    print("DirectQueryPlanner qualified-schema self-test: PASS")


def _self_test_cycle_signature() -> None:
    """Verify repeated hard-validation errors normalize to one state."""
    first = DirectSQLValidationResult(
        is_valid=False,
        sql="SELECT ...",
        errors=["Undocumented column: sample.value."],
    )

    second = DirectSQLValidationResult(
        is_valid=False,
        sql="SELECT ...",
        errors=["Undocumented column: sample.value."],
    )

    assert _validation_signature(first) == _validation_signature(second)

    print("DirectQueryPlanner cycle-signature self-test: PASS")


if __name__ == "__main__":
    _self_test_prompts()
    _self_test_fully_qualified_schema()
    _self_test_cycle_signature()
