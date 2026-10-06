"""
core/direct_query_grader.py
───────────────────────────
Lightweight semantic grader for the DirectQuery path.

Purpose
-------
DirectSQLValidator answers:

    "Is this SQL safe and grounded in the documented schema?"

DirectQueryGrader answers:

    "Assuming the SQL passed deterministic validation, does the candidate SQL
     actually answer the user's question, using schema elements that represent
     the business concept the user asked about?"

The grader does NOT:
    * execute SQL;
    * validate table/column existence;
    * validate SchemaGraph relationships;
    * repair SQL itself;
    * perform broad security investigations;
    * replace DirectSQLValidator.

A failing grade may be used by the caller for ONE bounded semantic repair.
"""

from __future__ import annotations

import re
from typing import Any

from llama_index.core.llms import LLM
from pydantic import BaseModel, Field

MAX_GRADER_REASON_CHARS = 1200
MAX_REPAIR_INSTRUCTION_CHARS = 800


class DirectQueryGrade(BaseModel):
    """Structured semantic assessment of one candidate SQL statement."""

    answers_question: bool = False

    reason: str = ""

    repair_instruction: str = ""

    # Short factual checklist useful for tracing/debugging.
    matched_requirements: list[str] = Field(default_factory=list)

    missing_requirements: list[str] = Field(default_factory=list)

    business_concept_match: bool = Field(
        default=False,
        description=(
            "True only when the tables/columns used by the candidate SQL "
            "actually represent the business concept named in the question."
        ),
    )


_GRADER_PROMPT = """
You are a semantic SQL grader for a read-only DirectQuery system.

A deterministic validator has already checked SQL safety, documented tables,
documented columns, and explicit SchemaGraph join relationships.

Your ONLY task is to determine whether the candidate SQL actually answers the
USER QUESTION.

USER QUESTION
-------------
{question}

CANDIDATE SQL
-------------
{sql}

AUTHORITATIVE SCHEMA
--------------------
{schema_context}

RULES
-----
1. Judge semantic completeness, not SQL safety. Do not repeat schema-safety
   validation unless it directly affects whether the SQL answers the question.

2. Do not execute the query and do not assume what rows it will return.

3. Preserve the exact meaning of the user's request, including:
   - all;
   - each / every / per;
   - one specific person/entity;
   - most recent / last / latest;
   - first / earliest;
   - counts;
   - grouping;
   - filters;
   - requested names or descriptive fields;
   - null/missing conditions;
   - duplicate conditions.

4. A query is incomplete if it retrieves related data but fails to implement a
   requested condition.

5. For "each", "every", or "per <entity>", a global LIMIT 1 generally does NOT
   satisfy the request because it returns one row for the entire result set
   rather than one result per entity.

6. For "last", "latest", or "most recent":
   - the SQL must implement recency using an appropriate aggregate, ordering,
     window operation, correlated subquery, or equivalent;
   - if the request is per entity, recency must be computed independently for
     each entity.

7. If the user asks for a specific person such as "James Anderson", the SQL
   must constrain the query to that person using appropriate documented fields.

8. If the user asks for a title, department name, room name, or another
   descriptive value, the SQL must actually return that requested descriptive
   value rather than only its identifier unless the question explicitly asks
   for the identifier.

9. Do not fail a query merely because a different valid SQL formulation could
   be written. Grade the candidate SQL on whether it satisfies the request.

10. If the candidate does not fully answer the question:
    - answers_question=false;
    - explain the concrete semantic problem;
    - provide ONE concise repair_instruction telling the SQL generator what
      must change;
    - do not write replacement SQL.

11. If the candidate fully answers the question:
    - answers_question=true;
    - repair_instruction must be empty.

12. Never invent business rules, literal values, schema relationships, or
    database facts.

13. matched_requirements and missing_requirements must contain short factual
    statements about the user request and candidate SQL.

14. Verify BUSINESS-CONCEPT MATCH, not merely relational validity.
    The SQL must use tables/columns whose documented meaning represents the
    business concept named by the user.

15. Related timestamps are NOT interchangeable merely because they concern the
    same employee. Examples:
    - "clocked in", "clocked out", "punch", "work date", or time-attendance
      questions must be answered from documented time-clock/time-attendance
      schema concepts.
    - badge access / room entry questions must be answered from documented
      badge-access schema concepts.
    - key use questions must be answered from documented key-access concepts.
    - camera/video questions must be answered from documented video/camera
      concepts.
    A badge_access_log.access_time is not evidence of a time-clock clock-in
    merely because both records contain an employee identifier and timestamp.

16. Do not infer business meaning from a generic timestamp alone. A field such
    as access_time, punch_time, start_time, or event_time must be interpreted
    according to the table/column documentation supplied in the schema.

17. If the candidate SQL uses a different business domain from the one named in
    the question:
    - answers_question=false;
    - business_concept_match=false;
    - explain the mismatch;
    - instruct the SQL generator to use schema elements documenting the
      requested business concept;
    - do NOT name a replacement table/column unless that name is explicitly
      supported by the supplied schema.

18. If the candidate uses the correct business domain and satisfies all other
    requested outputs/filters/cardinality/recency requirements, set
    business_concept_match=true.

19. A structurally valid join does not prove semantic correctness. SchemaGraph
    validation means the relationship exists; you must still decide whether the
    joined data answers the user's actual question.

20. Example semantic failures:
    USER: "When was James Anderson last clocked in?"
    BAD CANDIDATE: selects badge_access_log.access_time for James Anderson.
    RESULT: FAIL. Badge-access time represents physical-access activity, not a
    documented clock-in event.

    USER: "When did James Anderson last enter a room?"
    BAD CANDIDATE: selects a time-clock punch timestamp.
    RESULT: FAIL. Time-clock attendance is not room-entry evidence.

21. Example semantic success pattern:
    USER asks about a clock-in.
    Candidate uses schema elements documented as time-clock/attendance data,
    constrains the requested employee, and implements "last/most recent".
    RESULT: PASS if all other requirements are satisfied.

Return only the structured DirectQueryGrade object.
"""


def _clean_text(
    value: Any,
) -> str:
    return str(value or "").strip()


def _truncate(
    value: str,
    maximum: int,
) -> str:
    cleaned = _clean_text(value)

    if len(cleaned) <= maximum:
        return cleaned

    return cleaned[: maximum - 3].rstrip() + "..."


def _strip_sql_fence(
    sql: str,
) -> str:
    """
    Remove a Markdown SQL fence if a caller accidentally supplies one.

    DirectSQLValidator should normally receive clean SQL already, but this keeps
    grader traces readable and does not alter SQL semantics.
    """
    cleaned = _clean_text(sql)

    cleaned = re.sub(
        r"^```(?:sql|mysql)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    return cleaned.strip()


def _extract_structured_grade(
    response: Any,
) -> DirectQueryGrade:
    """
    Support common LlamaIndex structured-response shapes defensively.
    """
    raw = getattr(
        response,
        "raw",
        None,
    )

    if isinstance(
        raw,
        DirectQueryGrade,
    ):
        return raw

    if isinstance(
        raw,
        dict,
    ):
        return DirectQueryGrade.model_validate(raw)

    if isinstance(
        response,
        DirectQueryGrade,
    ):
        return response

    parsed = (
        getattr(
            response,
            "additional_kwargs",
            {},
        )
        or {}
    ).get("parsed")

    if isinstance(
        parsed,
        DirectQueryGrade,
    ):
        return parsed

    if isinstance(
        parsed,
        dict,
    ):
        return DirectQueryGrade.model_validate(parsed)

    text = _clean_text(
        getattr(
            response,
            "text",
            response,
        )
    )

    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    if not text:
        raise ValueError("DirectQueryGrader returned an empty response.")

    return DirectQueryGrade.model_validate_json(text)


class DirectQueryGrader:
    """
    Judge whether validated DirectQuery SQL answers the user's question.

    This is intentionally much smaller than GraderWriterAgent. It is a semantic
    gate for ordinary database queries, not a security-investigation grader.
    """

    def __init__(
        self,
        llm: LLM,
    ) -> None:
        self._structured_llm = llm.as_structured_llm(DirectQueryGrade)

        print("\n[DirectQueryGrader] ⚙  Initialising semantic SQL grader…")
        print("[DirectQueryGrader]    Input  : question + validated SQL + schema")
        print("[DirectQueryGrader]    Output : semantic PASS/FAIL + repair instruction")
        print("[DirectQueryGrader]    SQL    : never executed here")

    def grade(
        self,
        *,
        question: str,
        sql: str,
        schema_context: str,
    ) -> DirectQueryGrade:
        """
        Grade one deterministic-validator-approved SQL candidate.
        """
        cleaned_question = _clean_text(question)

        cleaned_sql = _strip_sql_fence(sql)

        cleaned_schema = _clean_text(schema_context)

        if not cleaned_question:
            return DirectQueryGrade(
                answers_question=False,
                reason="The user question is empty.",
                repair_instruction="A non-empty user question is required.",
                missing_requirements=["Non-empty user question"],
                business_concept_match=False,
            )

        if not cleaned_sql:
            return DirectQueryGrade(
                answers_question=False,
                reason="The candidate SQL is empty.",
                repair_instruction=(
                    "Generate one SQL SELECT statement that directly answers "
                    "the user question."
                ),
                missing_requirements=["Candidate SQL"],
                business_concept_match=False,
            )

        if not cleaned_schema:
            return DirectQueryGrade(
                answers_question=False,
                reason=(
                    "No authoritative schema context was supplied to the "
                    "semantic grader."
                ),
                repair_instruction="Supply authoritative schema context.",
                missing_requirements=["Authoritative schema context"],
                business_concept_match=False,
            )

        print("\n[DirectQueryGrader] Evaluating candidate SQL…")
        print("[DirectQueryGrader]    Question chars : " f"{len(cleaned_question)}")
        print("[DirectQueryGrader]    SQL chars      : " f"{len(cleaned_sql)}")
        print("[DirectQueryGrader]    Schema chars   : " f"{len(cleaned_schema)}")

        response = self._structured_llm.complete(
            _GRADER_PROMPT.format(
                question=cleaned_question,
                sql=cleaned_sql,
                schema_context=cleaned_schema,
            )
        )

        result = _extract_structured_grade(response)

        # Defensive normalization.
        result.reason = _truncate(
            result.reason,
            MAX_GRADER_REASON_CHARS,
        )

        result.repair_instruction = _truncate(
            result.repair_instruction,
            MAX_REPAIR_INSTRUCTION_CHARS,
        )

        result.matched_requirements = list(
            dict.fromkeys(
                _clean_text(item)
                for item in result.matched_requirements
                if _clean_text(item)
            )
        )

        result.missing_requirements = list(
            dict.fromkeys(
                _clean_text(item)
                for item in result.missing_requirements
                if _clean_text(item)
            )
        )

        # Semantic PASS requires both general completeness and a matching
        # business domain. This prevents a structurally valid but conceptually
        # wrong query (for example badge-access time for a clock-in question)
        # from reaching MySQL.
        if result.answers_question and not result.business_concept_match:
            result.answers_question = False

            if not result.reason:
                result.reason = (
                    "The candidate SQL does not use schema elements that "
                    "represent the business concept requested by the user."
                )

            if not result.repair_instruction:
                result.repair_instruction = (
                    "Revise the SQL to use documented schema elements that "
                    "represent the business concept named in the user question."
                )

            if (
                "Business concept/domain represented by the SQL"
                not in result.missing_requirements
            ):
                result.missing_requirements.append(
                    "Business concept/domain represented by the SQL"
                )

        if result.answers_question:
            # A PASS must never trigger repair.
            result.repair_instruction = ""
        else:
            if not result.reason:
                result.reason = (
                    "The candidate SQL does not fully answer the user question."
                )

            if not result.repair_instruction:
                result.repair_instruction = (
                    "Revise the SQL so it uses schema elements representing "
                    "the requested business concept and directly satisfies every "
                    "requested output, filter, grouping, cardinality, and recency "
                    "requirement."
                )

        print(
            "[DirectQueryGrader]    Business concept: "
            f"{result.business_concept_match}"
        )
        print("[DirectQueryGrader]    Answers question: " f"{result.answers_question}")

        if result.reason:
            print("[DirectQueryGrader]    Reason          : " f"{result.reason}")

        if result.missing_requirements:
            print(
                "[DirectQueryGrader]    Missing         : "
                f"{result.missing_requirements}"
            )

        if result.repair_instruction:
            print(
                "[DirectQueryGrader]    Repair          : "
                f"{result.repair_instruction}"
            )

        return result

    @staticmethod
    def repair_feedback(
        grade: DirectQueryGrade,
    ) -> str:
        """
        Render one bounded semantic-repair instruction for DirectQueryPlanner.

        DirectQueryGrader does not call the SQL generator itself.
        """
        if grade.answers_question:
            return ""

        lines = [
            "The candidate SQL passed deterministic schema/safety validation,",
            "but it did not fully answer the user question.",
            "",
            "SEMANTIC GRADING:",
            f"- Reason: {grade.reason}",
            ("- Business concept match: " f"{grade.business_concept_match}"),
        ]

        if grade.missing_requirements:
            lines.append(
                "- Missing requirements: " + "; ".join(grade.missing_requirements)
            )

        lines.extend(
            [
                "",
                "REPAIR INSTRUCTION:",
                grade.repair_instruction,
                "",
                (
                    "Use the authoritative schema to identify which documented "
                    "tables/columns actually represent the business concept in "
                    "the user question. Do not substitute a merely related "
                    "timestamp or activity domain."
                ),
                "",
                "Return exactly one corrected read-only MySQL SELECT statement.",
                "Do not explain the SQL.",
            ]
        )

        return "\n".join(lines)
