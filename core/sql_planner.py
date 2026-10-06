"""
core/sql_planner.py
───────────────────
Schema-grounded SQL generation for investigation evidence tasks.

Architecture
------------

    Investigation task
            ↓
    LLM generates evidence-gathering SQL
            ↓
    Python hard-validates safety, schema, columns, tables, and relationships
    against the COMPLETE authoritative schema
            ↓
    Invalid? Give the SAME LLM the prior SQL + exact factual defects
    in bounded repair mode; preserve valid work rather than regenerating
            ↓
    Valid? SAME LLM performs LIGHTWEIGHT SEMANTIC RELEVANCE REVIEW
            ↓
    Reviewer concerns are advisory and do NOT veto execution of otherwise
    schema-valid, safe investigative SQL
            ↓
    SQLPlan returned to RetrieverAgent for execution
            ↓
    Post-execution evidence review decides whether more investigation is needed
            ↓
    If the needed fact may be obtained by improving successful SQL, SQLPlanner
    refines that query before InvestigationPlanner opens a new evidence domain

SQLPlanner selects the physical tables, columns, and documented relationships
it needs from the COMPLETE authoritative schema. InvestigationPlanner table/path
selections are not used as SQL-generation hints.

Python does not require a specific aggregation, grouping, duplicate-detection,
temporal, table-selection, or other semantic SQL recipe.
"""

from __future__ import annotations

import re
from typing import Any

from llama_index.core.llms import ChatMessage, LLM

from core.sql_compiler import validate_read_only_sql
from core.schema_model import SchemaCatalog, parse_schema_context
from core.state import SQLPlan
from core.system_prompts import (
    SQL_PLANNER_SYSTEM_PROMPT,
    SQL_REPAIR_SYSTEM_PROMPT,
    SQL_REVIEW_SYSTEM_PROMPT,
)


def _complete_with_system(
    llm: Any,
    *,
    system_prompt: str,
    user_prompt: str,
) -> Any:
    """Run an LLM turn with an explicit system-role reasoning contract."""
    return llm.chat(
        [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]
    )


MAX_SQL_CORRECTIONS = 2

MYSQL_EXPLAIN_REQUIRED_MARKER = "MYSQL_EXPLAIN_REQUIRED"

_SAFETY_VALIDATION_PREFIXES = (
    "sql is empty",
    "exactly one sql statement is allowed",
    "only read-only select/query sql is permitted",
    "modifying sql operation is not permitted",
    "mysql parse error",
)


def _database_can_adjudicate_validation_errors(errors: list[str] | None) -> bool:
    """
    Return True when Python's failure is structural/schema-related rather than a
    safety failure.

    SQLPlanner does not claim such SQL is executable. It marks the candidate as
    requiring MySQL EXPLAIN so the database can make the authoritative decision.
    """
    cleaned = _clean_string_list(errors)

    if not cleaned:
        return False

    for error in cleaned:
        lowered = error.lower()
        if lowered.startswith(_SAFETY_VALIDATION_PREFIXES):
            return False

    return True


def _mark_for_mysql_explain(plan: SQLPlan) -> SQLPlan:
    errors = _clean_string_list(plan.validation_errors)
    return plan.updated(
        is_valid=True,
        validation_errors=[
            MYSQL_EXPLAIN_REQUIRED_MARKER,
            *errors,
        ],
    )


def requires_mysql_explain(plan: SQLPlan) -> bool:
    return MYSQL_EXPLAIN_REQUIRED_MARKER in _clean_string_list(
        getattr(plan, "validation_errors", [])
    )


# Schema parsing is centralized in core.schema_model.

_SQL_GENERATION_PROMPT = """
Generate one read-only MySQL query that gathers the factual evidence needed for
the investigation question.

EVIDENCE QUESTION
-----------------
{query}

PERSISTENT INVESTIGATION LEDGER
-------------------------------
{ledger_context}

PHYSICAL DATABASE SCHEMA
------------------------
{physical_schema}

REFINEMENT CONTEXT
------------------
{refinement_context}

PREVIOUS ATTEMPT NOTE
---------------------
{feedback}

RULES
-----
1. Choose the physical tables, columns, joins, predicates, grouping, and temporal
   logic yourself from the schema above.
2. Use only table and column names that literally appear in the physical schema.
3. The ledger contains semantic evidence concepts, not database identifiers.
4. Gather evidence; do not make the final cybersecurity judgment in SQL.
5. Do not invent business values, dates, thresholds, schedules, or categorical
   values that are not supplied by the question or schema.
6. If refinement context is present, use the actual prior SQL/result/gap as
   evidence and materially improve the query.
7. Return exactly one read-only MySQL SELECT/WITH query and nothing else.
"""

_SQL_REPAIR_PROMPT = """
Repair the existing read-only MySQL query.

EVIDENCE QUESTION
-----------------
{query}

PERSISTENT INVESTIGATION LEDGER
-------------------------------
{ledger_context}

PHYSICAL DATABASE SCHEMA
------------------------
{physical_schema}

PRIOR SQL — AUTHORITATIVE STARTING POINT
----------------------------------------
{prior_sql}

CURRENT DEFECTS TO REPAIR
-------------------------
{current_errors}

REPAIR HISTORY
--------------
{repair_history}

REFINEMENT CONTEXT
------------------
{refinement_context}

INSTRUCTIONS
------------
1. Preserve every useful valid part of PRIOR SQL.
2. Correct the CURRENT DEFECTS. Do not redesign the investigation simply because
   the prior SQL is imperfect.
3. Do not add unrelated tables, predicates, categorical values, dates, or business
   restrictions.
4. Treat every previously rejected physical identifier in REPAIR HISTORY as settled
   invalid for this repair episode.
5. If a named column does not exist, do not guess a similarly named replacement.
   Re-read the physical schema and restructure only the affected portion.
6. If the defect is semantic, preserve the useful evidence domains and repair the
   specific correlation/truth condition identified by the reviewer.
7. If the defect is a database execution error, repair that SQL/runtime property
   while preserving the intended evidence question.
8. Return the complete repaired query only.
"""

_SQL_REVIEW_PROMPT = """
Review this hard-valid read-only MySQL query for EVIDENCE LOGIC.

EVIDENCE QUESTION
-----------------
{query}

PERSISTENT INVESTIGATION LEDGER
-------------------------------
{ledger_context}

PHYSICAL DATABASE SCHEMA
------------------------
{schema_context}

VALIDATED SQL
-------------
{candidate_sql}

The SQL has already passed physical-schema and safety validation.
Do not re-check table/column existence and do not rewrite the query.

REASONING TASK
--------------
Reason about what the SQL actually tests:

1. Translate the important JOIN and WHERE predicates into plain language.
2. Compare those truth conditions with the EVIDENCE QUESTION.
3. Check temporal semantics, Boolean logic, NULL behavior, join effects, and
   unrelated restrictions that could materially distort the result.
4. For event/state correlation questions, explicitly verify that the SQL relates
   the SAME entity to the relevant state at the SAME event timestamp or applicable
   interval. Global "ever/never" membership, MIN/MAX envelopes across unrelated
   rows, or merely touching both evidence domains is not enough for DIRECTLY_TESTS.
5. Check whether the query accidentally requires mutually exclusive states for the
   same event, or uses aggregate time bounds that do not correspond to the specific
   employee/event being tested.
6. Decide whether execution would directly test, materially advance, merely touch
   related domains, or clearly miss the requested evidence relationship.

DIRECTLY_TESTS requires that the SQL's truth conditions themselves establish the
requested relationship for each relevant record. Do not classify a query as
DIRECTLY_TESTS merely because its CTE names, comments, or tables sound aligned with
the question.

Return exactly one form:

DIRECTLY_TESTS
reason: <brief explanation>

MATERIALLY_ADVANCES
reason: <brief explanation>

DOMAIN_RELATED_ONLY
reason: <brief explanation>

CLEAR_MISMATCH
reason: <brief explanation>

Do not generate replacement SQL.
"""


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_string_list(values: list[str] | None) -> list[str]:
    result: list[str] = []

    for value in values or []:
        cleaned = _clean_text(value)

        if cleaned and cleaned not in result:
            result.append(cleaned)

    return result


def _extract_sql_candidate(value: Any) -> str:
    text = _clean_text(
        getattr(value, "text", None)
        or getattr(getattr(value, "message", None), "content", None)
        or value
    )

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


def _parse_semantic_review(value: Any) -> tuple[str, str]:
    """Parse the four-level LLM evidence-logic classification."""
    text = _clean_text(
        getattr(value, "text", None)
        or getattr(getattr(value, "message", None), "content", None)
        or value
    )

    if not text:
        return (
            "MATERIALLY_ADVANCES",
            "Semantic review returned an empty response; allowing execution as "
            "potentially useful investigative evidence.",
        )

    normalized = text.strip()
    first_line = normalized.splitlines()[0].strip().upper()

    labels = (
        "DIRECTLY_TESTS",
        "MATERIALLY_ADVANCES",
        "DOMAIN_RELATED_ONLY",
        "CLEAR_MISMATCH",
    )
    classification = next(
        (label for label in labels if first_line.startswith(label)),
        "MATERIALLY_ADVANCES",
    )

    reason_match = re.search(
        r"reason\s*:\s*(.*)",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    reason = reason_match.group(1).strip() if reason_match else normalized
    return classification, reason


def _parse_schema_catalog(schema_context: str) -> SchemaCatalog:
    """Return the shared authoritative deterministic schema model."""
    return parse_schema_context(schema_context)


def _compact_feedback_note(feedback: list[str] | None) -> str:
    """Keep correction guidance short and avoid repeating invalid identifiers."""
    cleaned = _clean_string_list(feedback)
    if not cleaned:
        return "(none)"

    # Runtime/execution feedback can be useful, but hard-validation details are
    # intentionally summarized so rejected identifiers are not made more salient.
    runtime = [
        item
        for item in cleaned
        if not item.lower().startswith(
            (
                "undocumented column",
                "undocumented table",
                "column is not documented",
                "undocumented relationship",
            )
        )
    ]
    if runtime:
        return "Previous attempt failed. Relevant execution feedback: " + " | ".join(
            runtime[-2:]
        )

    return (
        "The previous SQL failed physical-schema validation. Generate a fresh "
        "query by re-reading the physical schema above; do not reuse an identifier "
        "unless it literally appears there."
    )


class SQLPlanner:
    """
    Let the LLM generate investigation SQL and perform a lightweight semantic
    relevance review after hard validation.

    Python blocks safety failures. Structural/schema concerns that MySQL can
    adjudicate may be returned as provisional EXPLAIN candidates. MySQL then
    becomes the final authority on executability for those safe candidates.
    """

    def __init__(
        self,
        llm: LLM,
    ) -> None:
        self._llm = llm

    def _complete(
        self,
        prompt: str,
        *,
        system_prompt: str = SQL_PLANNER_SYSTEM_PROMPT,
    ) -> str:
        response = _complete_with_system(
            self._llm,
            system_prompt=system_prompt,
            user_prompt=prompt,
        )
        return _extract_sql_candidate(response)

    def _generate(
        self,
        *,
        query: str,
        schema_context: str,
        feedback: list[str],
        allowed_tables: list[str],
        allowed_relationship_paths: list[str],
        ledger_context: str,
        refinement_context: str = "",
        settled_schema_facts: list[str] | None = None,
    ) -> str:
        del allowed_tables
        del allowed_relationship_paths
        del settled_schema_facts

        schema = _parse_schema_catalog(schema_context)
        physical_schema = schema.render_physical_schema(
            include_descriptions=True,
            include_relationships=True,
        )

        return self._complete(
            _SQL_GENERATION_PROMPT.format(
                query=query,
                ledger_context=ledger_context or "(no ledger entries yet)",
                physical_schema=physical_schema,
                refinement_context=(
                    refinement_context.strip()
                    or "(none; generate a fresh evidence-gathering query)"
                ),
                feedback=_compact_feedback_note(feedback),
            )
        )

    def _repair_candidate(
        self,
        *,
        query: str,
        schema_context: str,
        prior_sql: str,
        current_errors: list[str],
        repair_history: list[str],
        ledger_context: str,
        refinement_context: str = "",
    ) -> str:
        """Repair prior SQL instead of starting another unconstrained generation."""
        schema = _parse_schema_catalog(schema_context)
        physical_schema = schema.render_physical_schema(
            include_descriptions=True,
            include_relationships=True,
        )

        return self._complete(
            _SQL_REPAIR_PROMPT.format(
                query=query,
                ledger_context=ledger_context or "(no ledger entries yet)",
                physical_schema=physical_schema,
                prior_sql=prior_sql or "(no prior SQL was captured)",
                current_errors=(
                    "\n".join(
                        f"- {item}" for item in _clean_string_list(current_errors)
                    )
                    or "- No specific defect was supplied."
                ),
                repair_history=(
                    "\n".join(
                        f"- {item}" for item in _clean_string_list(repair_history)
                    )
                    or "(none; this is the first repair)"
                ),
                refinement_context=(refinement_context.strip() or "(none)"),
            ),
            system_prompt=SQL_REPAIR_SYSTEM_PROMPT,
        )

    def _review(
        self,
        *,
        query: str,
        schema_context: str,
        candidate_sql: str,
        allowed_tables: list[str],
        allowed_relationship_paths: list[str],
        ledger_context: str,
    ) -> tuple[str, str]:
        """
        Perform LLM evidence-logic review after hard validation.

        This reviewer never rewrites SQL and does not veto execution. Its reason
        is retained for traceability while post-execution evidence review decides
        whether more investigation is needed.
        """
        del allowed_tables
        del allowed_relationship_paths

        physical_schema = _parse_schema_catalog(schema_context).render_physical_schema(
            include_descriptions=False,
            include_relationships=True,
        )

        review_prompt = _SQL_REVIEW_PROMPT.format(
            query=query,
            ledger_context=ledger_context or "(no ledger entries yet)",
            schema_context=physical_schema,
            candidate_sql=candidate_sql,
        )

        response = _complete_with_system(
            self._llm,
            system_prompt=SQL_REVIEW_SYSTEM_PROMPT,
            user_prompt=review_prompt,
        )

        return _parse_semantic_review(response)

    @staticmethod
    def _validation_signature(plan: SQLPlan) -> tuple[str, ...]:
        """Stable signature used to notice when the LLM repeats the same defect."""
        return tuple(
            sorted(
                _clean_text(error).lower()
                for error in (plan.validation_errors or [])
                if _clean_text(error)
            )
        )

    def _generate_review_validate_once(
        self,
        *,
        query: str,
        schema_context: str,
        schema: SchemaCatalog,
        feedback_items: list[str],
        allowed_tables: list[str],
        allowed_relationship_paths: list[str],
        ledger_context: str,
        attempt_number: int,
        total_attempts: int,
        refinement_context: str = "",
        settled_schema_facts: list[str] | None = None,
        candidate_sql: str | None = None,
    ) -> tuple[SQLPlan, str | None, str]:
        """
        Run one reasoning cycle:

            LLM generation
                -> deterministic hard validation
                -> if valid, lightweight LLM semantic relevance review
                -> return valid SQL for execution

        The semantic reviewer never rewrites SQL and never converts a hard-valid
        query into an invalid plan.

        Returns:
            (plan, semantic_feedback, generated_sql)

        semantic_feedback is populated when LLM evidence-logic review asks for
        another reasoning pass.
        """
        print("\n" + "═" * 70)
        print(
            "[SQLPlanner] ▶ SQL REASONING CYCLE " f"{attempt_number}/{total_attempts}"
        )
        print(f"[SQLPlanner]    Query           : {query}")
        physical_schema_chars = len(schema.render_physical_schema())
        print(f"[SQLPlanner]    Raw schema chars: {len(schema_context)}")
        print(f"[SQLPlanner]    SQL schema chars: {physical_schema_chars}")
        print(f"[SQLPlanner]    Feedback items  : {len(feedback_items)}")
        print("[SQLPlanner]    Table selection : LLM chooses from complete schema")
        print("[SQLPlanner]    Planner hints   : not supplied to SQL generation/review")
        print("[SQLPlanner]    Hard scope      : complete authoritative schema")
        print(
            "[SQLPlanner]    LLM grounding   : compact physical schema; no Python table-role heuristics"
        )
        print(
            "[SQLPlanner]    Refinement mode : "
            f"{'yes' if refinement_context.strip() else 'no'}"
        )

        try:
            if candidate_sql is None:
                generated = self._generate(
                    query=query,
                    schema_context=schema_context,
                    feedback=feedback_items,
                    allowed_tables=allowed_tables,
                    allowed_relationship_paths=allowed_relationship_paths,
                    ledger_context=ledger_context,
                    refinement_context=refinement_context,
                    settled_schema_facts=settled_schema_facts,
                )
                print("[SQLPlanner]    Generation mode : FRESH")
            else:
                generated = candidate_sql
                print("[SQLPlanner]    Generation mode : REPAIR")

            print("[SQLPlanner]    Generated SQL:")
            print(generated)

        except Exception as exc:
            error = "SQL generation failed: " f"{type(exc).__name__}: {exc}"

            print(f"[SQLPlanner] ✘ {error}")
            print("═" * 70 + "\n")

            return (
                SQLPlan(
                    purpose=query,
                    is_valid=False,
                    validation_errors=[error],
                ),
                None,
                "",
            )

        validated = validate_read_only_sql(
            generated,
            schema,
            purpose=query,
            allowed_tables=None,
            allowed_relationship_paths=None,
        )

        print(
            "[SQLPlanner]    Hard validate: "
            f"{'PASS' if validated.is_valid else 'FAIL'}"
        )

        if validated.validation_errors:
            for error in validated.validation_errors:
                print(f"[SQLPlanner]      - {error}")

        if not validated.is_valid:
            if _database_can_adjudicate_validation_errors(validated.validation_errors):
                print(
                    "[SQLPlanner]    Python preflight found structural/schema "
                    "concerns that MySQL can adjudicate."
                )
                print(
                    "[SQLPlanner]    Database validation: MYSQL EXPLAIN REQUIRED "
                    "before execution."
                )
                print("═" * 70 + "\n")
                return _mark_for_mysql_explain(validated), None, generated

            print(
                "[SQLPlanner]    Python safety preflight blocked the candidate; "
                "it will not be sent to MySQL."
            )
            print("═" * 70 + "\n")
            return validated, None, generated

        print(
            "[SQLPlanner]    Hard validation passed; "
            "starting lightweight LLM semantic relevance review."
        )

        try:
            classification, reason = self._review(
                query=query,
                schema_context=schema_context,
                candidate_sql=validated.sql,
                allowed_tables=allowed_tables,
                allowed_relationship_paths=allowed_relationship_paths,
                ledger_context=ledger_context,
            )
        except Exception as exc:
            classification = "MATERIALLY_ADVANCES"
            reason = (
                "Semantic evidence-logic review failed; allowing execution as "
                f"potentially useful evidence: {type(exc).__name__}: {exc}"
            )

        print(f"[SQLPlanner]    Semantic review: {classification}")
        print(f"[SQLPlanner]    Semantic reason: {reason}")

        if classification in {"DIRECTLY_TESTS", "MATERIALLY_ADVANCES"}:
            print(
                "[SQLPlanner]    Execution policy: semantic review says the SQL "
                "directly tests or materially advances the evidence question."
            )
            print("═" * 70 + "\n")
            return validated, None, generated

        semantic_feedback = (
            "The previous SQL was physically valid but the SQL evidence-logic "
            f"review classified it as {classification}. Reviewer reason: {reason}. "
            "Generate a materially different query that better addresses the "
            "evidence question."
        )

        print(
            "[SQLPlanner]    Execution policy: physically valid SQL will NOT execute "
            "yet because the LLM says it does not materially advance the evidence "
            "question. Requesting a fresh LLM reasoning attempt."
        )
        print("═" * 70 + "\n")

        rejected = validated.model_copy(
            update={
                "is_valid": False,
                "validation_errors": [semantic_feedback],
            }
        )
        return rejected, semantic_feedback, generated

    def plan(
        self,
        query: str,
        schema_context: str,
        feedback: list[str] | None = None,
        allowed_tables: list[str] | None = None,
        allowed_relationship_paths: list[str] | None = None,
        ledger_context: str = "",
        refinement_context: str = "",
    ) -> SQLPlan:
        """
        Produce one schema-valid investigation SQL plan.

        Reasoning budget:
            initial generation
            + up to MAX_SQL_CORRECTIONS fresh corrections

        Each failed hard-validation result triggers a fresh LLM attempt grounded
        in the compact physical schema. Detailed rejected identifiers remain in
        logs but are not repeatedly injected into the next prompt. Semantic review
        does not consume the correction budget because it is advisory.
        """
        query = _clean_text(query)
        schema_context = _clean_text(schema_context)
        accumulated_feedback = _clean_string_list(feedback)

        # Backward-compatible parameters only. Upstream physical-table/path hints
        # are intentionally excluded from the SQL LLM context.
        allowed_tables = []
        allowed_relationship_paths = []

        if not query:
            return SQLPlan(
                is_valid=False,
                validation_errors=["The evidence question is empty."],
            )

        if not schema_context:
            return SQLPlan(
                purpose=query,
                is_valid=False,
                validation_errors=["No authoritative schema context was supplied."],
            )

        schema = _parse_schema_catalog(schema_context)

        print(
            "[SQLPlanner]    Parsed schema   : "
            f"{len(schema.tables)} tables / "
            f"{sum(len(columns) for columns in schema.tables.values())} columns / "
            f"{len(schema.relationships)} relationships"
        )
        if schema.tables:
            print(
                "[SQLPlanner]    Parsed tables   : " + ", ".join(sorted(schema.tables))
            )

        if not schema.tables:
            return SQLPlan(
                purpose=query,
                is_valid=False,
                validation_errors=["No table definitions could be parsed."],
            )

        total_attempts = 1 + MAX_SQL_CORRECTIONS
        settled_schema_facts: list[str] = []  # compatibility only; not LLM-facing
        final_plan = SQLPlan(
            purpose=query,
            is_valid=False,
            validation_errors=["SQL planning did not run."],
        )

        prior_sql = ""
        repair_history: list[str] = []
        pending_candidate: str | None = None

        print("[SQLPlanner]    Validation correction budget: " f"{MAX_SQL_CORRECTIONS}")

        for attempt_number in range(1, total_attempts + 1):
            final_plan, semantic_feedback, generated_sql = (
                self._generate_review_validate_once(
                    query=query,
                    schema_context=schema_context,
                    schema=schema,
                    feedback_items=accumulated_feedback,
                    allowed_tables=allowed_tables,
                    allowed_relationship_paths=allowed_relationship_paths,
                    ledger_context=ledger_context,
                    attempt_number=attempt_number,
                    total_attempts=total_attempts,
                    refinement_context=refinement_context,
                    settled_schema_facts=settled_schema_facts,
                    candidate_sql=pending_candidate,
                )
            )

            if generated_sql:
                prior_sql = generated_sql

            if final_plan.is_valid:
                corrections_used = attempt_number - 1
                if requires_mysql_explain(final_plan):
                    print(
                        "[SQLPlanner] ✔ Safe SQL candidate produced after "
                        f"{corrections_used} correction(s); MySQL EXPLAIN will "
                        "decide executability."
                    )
                else:
                    print(
                        "[SQLPlanner] ✔ Executable investigative SQL produced after "
                        f"{corrections_used} correction(s)."
                    )
                return final_plan

            if attempt_number >= total_attempts:
                break

            current_errors = _clean_string_list(final_plan.validation_errors)

            if semantic_feedback:
                current_errors = [semantic_feedback]

            repair_history.extend(
                f"Attempt {attempt_number}: {item}" for item in current_errors if item
            )
            repair_history = _clean_string_list(repair_history)

            print(
                "[SQLPlanner] ↻ Entering bounded REPAIR MODE "
                f"{attempt_number}/{MAX_SQL_CORRECTIONS}; preserving prior SQL."
            )
            for item in current_errors:
                print(f"[SQLPlanner]    Repair defect: {item}")

            try:
                pending_candidate = self._repair_candidate(
                    query=query,
                    schema_context=schema_context,
                    prior_sql=prior_sql,
                    current_errors=current_errors,
                    repair_history=repair_history,
                    ledger_context=ledger_context,
                    refinement_context=refinement_context,
                )
            except Exception as exc:
                pending_candidate = None
                accumulated_feedback = [
                    "SQL repair generation failed: " f"{type(exc).__name__}: {exc}"
                ]
                print(
                    "[SQLPlanner] ⚠ Repair generation failed; the next bounded "
                    "attempt may fall back to fresh generation."
                )

        print(
            "[SQLPlanner] ✘ No schema-valid SQL after initial generation + "
            f"{MAX_SQL_CORRECTIONS} correction(s)."
        )

        return final_plan

    def refine_from_evidence(
        self,
        *,
        original_query: str,
        schema_context: str,
        prior_plan: SQLPlan,
        row_count: int,
        columns: list[str],
        evidence_preview: str,
        semantic_gap: str,
        ledger_context: str = "",
        refinement_feedback: list[str] | None = None,
    ) -> SQLPlan:
        """
        Refine a successfully executed investigation query after post-execution
        LLM reasoning identifies a remaining semantic gap.

        Python supplies the successful SQL, observed result shape, and the LLM's
        own semantic shortcoming. The LLM still chooses whether and how to adjust
        the query and whether another evidence domain is genuinely necessary.
        """
        prior_sql = _clean_text(getattr(prior_plan, "sql", ""))

        refinement_context = "\n".join(
            [
                "A previous investigation query EXECUTED SUCCESSFULLY.",
                "",
                "ORIGINAL USER QUESTION:",
                _clean_text(original_query),
                "",
                "PREVIOUS SUCCESSFUL SQL:",
                prior_sql or "(not recorded)",
                "",
                f"ROWS RETURNED: {int(row_count)}",
                "RETURNED COLUMNS:",
                ", ".join(_clean_string_list(columns)) or "(none)",
                "",
                "EVIDENCE PREVIEW:",
                _clean_text(evidence_preview) or "(no preview available)",
                "",
                "POST-EXECUTION LLM SEMANTIC GAP:",
                _clean_text(semantic_gap) or "(not specified)",
                "",
                "REFINEMENT GOAL:",
                (
                    "First determine whether the existing successful query already "
                    "contains the evidence domains needed and can be improved by "
                    "changing its correlation, predicates, or returned fields. "
                    "Preserve useful existing domains when possible. Add another "
                    "domain only if it is genuinely needed to establish the "
                    "remaining fact."
                ),
                "",
                "PRIOR REFINEMENT FEEDBACK:",
                (
                    "\n".join(
                        f"- {item}" for item in _clean_string_list(refinement_feedback)
                    )
                    or "(none)"
                ),
            ]
        ).strip()

        print(
            "[SQLPlanner] ↻ REFINING SUCCESSFUL INVESTIGATION SQL from "
            "post-execution evidence reasoning."
        )

        return self.plan(
            query=original_query,
            schema_context=schema_context,
            feedback=_clean_string_list(refinement_feedback),
            ledger_context=ledger_context,
            refinement_context=refinement_context,
        )

    def repair(
        self,
        query: str,
        schema_context: str,
        prior_plan: SQLPlan,
        errors: list[str],
        allowed_tables: list[str] | None = None,
        allowed_relationship_paths: list[str] | None = None,
        ledger_context: str = "",
    ) -> SQLPlan:
        """
        Repair SQL after a database execution error.

        The successfully hard-validated prior SQL remains the authoritative starting
        point. The LLM fixes the runtime defect without reopening unconstrained table
        selection or redesigning the investigation.
        """
        del allowed_tables
        del allowed_relationship_paths

        query = _clean_text(query)
        schema_context = _clean_text(schema_context)
        prior_sql = _clean_text(getattr(prior_plan, "sql", ""))
        current_errors = _clean_string_list(errors)

        if not prior_sql:
            return self.plan(
                query=query,
                schema_context=schema_context,
                feedback=current_errors,
                ledger_context=ledger_context,
            )

        schema = _parse_schema_catalog(schema_context)
        repair_history = [f"Execution defect: {item}" for item in current_errors]
        candidate = prior_sql
        final_plan = prior_plan

        print(
            "[SQLPlanner] ↻ EXECUTION-ERROR REPAIR MODE — "
            "preserving the previously hard-valid SQL."
        )
        for item in current_errors:
            print(f"[SQLPlanner]    Repair defect: {item}")

        for repair_number in range(1, MAX_SQL_CORRECTIONS + 2):
            if repair_number > 1:
                try:
                    candidate = self._repair_candidate(
                        query=query,
                        schema_context=schema_context,
                        prior_sql=candidate,
                        current_errors=current_errors,
                        repair_history=repair_history,
                        ledger_context=ledger_context,
                    )
                except Exception as exc:
                    return SQLPlan(
                        purpose=query,
                        database_type="mysql",
                        sql=candidate,
                        is_valid=False,
                        validation_errors=[
                            "Execution-error SQL repair failed: "
                            f"{type(exc).__name__}: {exc}"
                        ],
                    )

            validated = validate_read_only_sql(
                candidate,
                schema,
                purpose=query,
                allowed_tables=None,
                allowed_relationship_paths=None,
            )

            if (
                not validated.is_valid
                and repair_number > 1
                and _database_can_adjudicate_validation_errors(
                    validated.validation_errors
                )
            ):
                print(
                    "[SQLPlanner]    Repaired SQL is safe enough for MySQL EXPLAIN; "
                    "deferring structural/schema adjudication to the database."
                )
                return _mark_for_mysql_explain(validated)

            if validated.is_valid and repair_number > 1:
                try:
                    classification, reason = self._review(
                        query=query,
                        schema_context=schema_context,
                        candidate_sql=validated.sql,
                        allowed_tables=[],
                        allowed_relationship_paths=[],
                        ledger_context=ledger_context,
                    )
                except Exception as exc:
                    classification = "MATERIALLY_ADVANCES"
                    reason = (
                        "Semantic review failed during execution repair: "
                        f"{type(exc).__name__}: {exc}"
                    )

                print(
                    "[SQLPlanner]    Execution-repair semantic review: "
                    f"{classification}"
                )
                print(f"[SQLPlanner]    Semantic reason: {reason}")

                if classification in {"DIRECTLY_TESTS", "MATERIALLY_ADVANCES"}:
                    return validated

                current_errors = [
                    "The repaired SQL is physically valid but was classified "
                    f"{classification}. Reviewer reason: {reason}"
                ]
            elif repair_number == 1:
                # The prior SQL already passed hard validation; the first iteration
                # exists only to establish the repair seed and does not re-execute it.
                pass
            else:
                current_errors = _clean_string_list(validated.validation_errors)

            if repair_number >= MAX_SQL_CORRECTIONS + 1:
                final_plan = validated
                break

            repair_history.extend(
                f"Repair {repair_number}: {item}" for item in current_errors
            )
            repair_history = _clean_string_list(repair_history)

        return final_plan
