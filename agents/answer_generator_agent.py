"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  agents/answer_generator_agent.py                                           ║
║  Agent 4 – Final Answer Writer                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE
───────
Write the final user-facing answer from evidence already graded by
GraderWriterAgent.

Agent 4 follows the presentation contract created by Agent 3:

1. VERBATIM
   Present complete verified evidence deterministically for contracts that
   explicitly require exact rendering.

2. SUMMARY
   Use the LLM to present verified factual evidence without inference.
   DirectQuery normally uses this mode so database rows become a clean
   user-facing answer rather than a raw schema/evidence dump.

3. INTERPRET
   Use the LLM to explain, compare, correlate, or analyze verified evidence.

4. FAILURE
   Present a deterministic evidence-failure response. Do not invoke the LLM.

AUTO is retained only as a compatibility fallback for states produced by older
workflow code.

The agent does not:

    * retrieve evidence;
    * inspect schema files;
    * generate SQL;
    * call MCP tools;
    * repair failed queries;
    * re-grade Agent 3's evidence-sufficiency decision;
    * override Agent 3's final presentation contract because ledger/task labels
      remain unresolved;
    * convert IDs into names;
    * invent dates, filters, thresholds, tables, columns, or findings;
    * present evidence as sufficient when Agent 3 explicitly graded it incomplete.
"""

from __future__ import annotations

import json
import re
from typing import Any

from llama_index.core.llms import ChatMessage, LLM
from pydantic import BaseModel, Field

from core.system_prompts import ANSWER_GENERATOR_SYSTEM_PROMPT
from core.state import (
    GradeResult,
    PresentationMode,
    RetrievalMode,
    WorkflowState,
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


# ============================================================================
# Structured response model
# ============================================================================


class FinalAnswerResponse(BaseModel):
    """Structured response produced by the final-answer LLM."""

    answer: str = Field(
        min_length=1,
        description=(
            "The final evidence-faithful answer. It must not contain claims "
            "that are absent from the supplied verified context."
        ),
    )

    used_evidence_types: list[str] = Field(
        default_factory=list,
        description=(
            "Evidence categories actually used: policy, schema, database, "
            "or investigation."
        ),
    )

    limitations: list[str] = Field(
        default_factory=list,
        description=("Only material limitations supported by the supplied state."),
    )


# ============================================================================
# Prompts
# ============================================================================

_SUMMARY_ANSWER_PROMPT = """
You are summarizing verified factual evidence for the user.

USER QUERY
----------
{query}

VERIFIED EVIDENCE
-----------------
{refined_context}

PERSISTENT INVESTIGATION EVIDENCE LEDGER
----------------------------------------
{ledger_context}

EVIDENCE CONTRACT
-----------------
Required evidence types: {required_evidence_types}
Required output fields: {required_output_fields}
Available evidence types: {available_evidence_types}

TASK
----
Summarize the verified evidence using only facts present in that evidence.

MANDATORY RULES
---------------
1. Condense the factual material requested by the user.

2. Preserve all material facts required to understand the evidence.

3. Organize the summary with clear headings and concise formatting.

3. Preserve exact names from the evidence, including:
   - table names;
   - column names;
   - key names;
   - relationships;
   - policy source names;
   - employee names;
   - dates and timestamps.

4. Do not add or rename tables, columns, relationships, records, people,
   policies, or facts.

5. Do not characterize a descriptive request as:
   - numerical;
   - mathematical;
   - inapplicable;
   - unsupported merely because it is not an investigation.

6. Do not discuss security findings unless the user requested them and the
   verified evidence supports them.

7. Do not add a limitations section unless the verified evidence is materially
   incomplete for the user's direct factual request.

8. Do not begin with meta-commentary such as:
   - "The final answer is not applicable";
   - "This format is designed for numerical problems";
   - "Based on the supplied evidence, I will";
   - "As an AI".

9. Do not summarize away requested detail. When the user asks for all schemas,
   tables, columns, or relationships, include all documented items supplied in
   the verified evidence.

10. IDs are not names. Do not relabel numeric IDs as people.

11. Do not invent dates, filters, thresholds, business values, or conclusions.

12. used_evidence_types must contain only evidence types actually present in
    the verified evidence.

13. limitations must be empty when the evidence directly satisfies the request.

14. A heading by itself is not an answer.

15. The response must include substantive factual content from VERIFIED
    EVIDENCE.

16. When the verified evidence contains multiple documented items, enumerate
    or summarize those items rather than returning only a title.

17. Do not finish the response until the requested factual material has
    actually been presented.

18. Treat technical identifiers as exact, immutable values. Copy table names,
    column names, control IDs, filenames, relationship endpoints, and other
    code-like identifiers exactly as they appear in VERIFIED EVIDENCE.

19. Never create a variation, synonym, prefix, suffix, or alternative form of
    a technical identifier.

20. Every technical identifier in the summary must match VERIFIED
    EVIDENCE exactly.

21. When VERIFIED EVIDENCE is a DirectQuery database result:
    - answer the user's question directly;
    - present the returned rows/values rather than schema documentation;
    - use natural labels when obvious from returned column names;
    - preserve all requested rows when the user asks for all/each/every;
    - do not mention schema tables or columns unless the user asked for them.

22. For a one-value DirectQuery, state the requested value directly. For a
    multi-row DirectQuery, organize the returned records clearly without
    dropping requested rows.
"""


_INTERPRETIVE_ANSWER_PROMPT = """
You are the final answer writer for a cybersecurity user-access analysis
system.

USER QUERY
----------
{query}

VERIFIED REFINED CONTEXT
------------------------
{refined_context}

PERSISTENT INVESTIGATION EVIDENCE LEDGER
----------------------------------------
{ledger_context}

EVIDENCE CONTRACT
-----------------
Required evidence types: {required_evidence_types}
Required output fields: {required_output_fields}
Available evidence types: {available_evidence_types}
Requirement reason: {evidence_requirement_reason}

EVIDENCE METADATA
-----------------
Database row count: {database_row_count}
Database columns: {database_columns}
Policy sources: {policy_sources}
Schema sources: {schema_sources}
Investigation task summary:
{investigation_summary}

TASK
----
Write the final interpretive answer using only the verified refined context.

AGENT 3 AUTHORITY
-----------------
Agent 3 has already graded the evidence and selected INTERPRET presentation.
Treat Agent 3's FINAL security/evidence conclusion in VERIFIED REFINED CONTEXT
as authoritative for presentation.

Important: grade=PASS may mean the grading process completed successfully; it
does NOT necessarily mean the investigation found a positive or negative result.
If Agent 3's overall conclusion is INCONCLUSIVE, preserve that qualification in
the overall assessment, but still present any supported findings and investigative
leads that Agent 3 explicitly preserved. Global uncertainty must not erase local
evidence-supported discoveries.

Do NOT independently downgrade a PASS/INTERPRET result to "incomplete" merely
because:
- the ledger still contains advisory missing labels;
- an individual investigation task was inconclusive;
- not every planned task succeeded;
- more evidence could theoretically be collected.

If Agent 3's refined context identifies a potential security risk, present that
finding clearly and include the supporting employees/events/records that are
actually present in the verified context. When available, identify who (name or
ID exactly as evidenced), what happened, when, where, and why the verified facts
were security-relevant. Do not omit those concrete details in favor of a generic
statement that concerns exist.

Distinguish an unauthorized or suspicious attempt that was blocked from a control
failure that allowed unauthorized activity. A Denied event can be security-relevant
behavior while simultaneously showing that the access control enforced the rule.

MANDATORY FIDELITY RULES
------------------------
1. Every factual claim must be directly supported by the verified context.

2. Do not use outside knowledge or assumptions as retrieved evidence.

3. Do not convert an ID into a name.

4. Do not invent:
   - employee names;
   - dates or date ranges;
   - filters;
   - thresholds;
   - room classifications;
   - access outcomes;
   - anomaly labels;
   - business rules;
   - security findings;
   - citations.

5. Do not claim records met a condition unless verified evidence demonstrates
   that condition was evaluated.

6. Do not describe raw events as anomalies, violations, or security risks
   unless the verified context supports that interpretation.

7. Preserve evidence distinctions:
   - policy guidance is not organizational fact;
   - schema documentation is not event evidence;
   - raw database activity is not automatically a finding;
   - a suspicious indicator is not a confirmed violation.

8. When requested information is absent, state exactly what is missing.

9. When an investigation has partial coverage, distinguish:
   - successful analyses;
   - failed or unsupported analyses;
   - supported findings;
   - unresolved areas.

10. Do not begin with meta-commentary.

11. Cite NIST or CIS only when those sources appear in verified context.

12. If no supported finding exists AND evidence coverage is sufficient, say so
    rather than constructing one.

13. Treat overall inconclusive status as a qualification on scope/coverage, not as
    a special presentation mode that suppresses evidence. If Agent 3 preserved
    supported findings or investigative leads, present them and separately explain
    what remains inconclusive. Only use a deterministic incomplete/failure response
    when Agent 3's FINAL presentation contract itself is FAILURE/incomplete.

    Never turn unresolved evidence into either "no issue found" or a confirmed
    finding, but do not discard verified concerns simply because other areas remain
    unresolved.

14. Neither zero rows nor many rows from one evidence domain establish the
    user's requested cross-domain condition unless the verified context
    explicitly says all required evidence domains were evaluated and correlated.

14a. If Agent 3's OVERALL conclusion is inconclusive:
    - state that the overall review remains inconclusive or partial;
    - still present every supported finding and investigative lead explicitly
      preserved by Agent 3;
    - explain the material unresolved evidence separately;
    - do not upgrade an investigative lead into a confirmed finding;
    - do not rewrite inconclusive as "no matches were found", "none occurred", or
      "no issue exists".

15. used_evidence_types must list only evidence types visibly present.

16. limitations must identify material gaps that affect the answer.

17. For an investigation, VERIFIED REFINED CONTEXT may contain a structured
    INVESTIGATION REPORT. Treat that report as the presentation source of truth.
    Do not reduce a broad review to only the most prominent finding.

18. Every material item under CONFIRMED / SUPPORTED SECURITY FINDINGS must be
    represented in the final answer unless it is genuinely duplicative of another
    finding you already presented. Preserve the concrete actor/ID, event/condition,
    time/date, location, control/rule state, evidence basis, and security significance
    whenever Agent 3 supplied those details.

18a. Every material item under INVESTIGATIVE LEADS / POTENTIAL CONCERNS must also
    be represented. Clearly label these as leads/potential concerns and preserve
    Agent 3's uncertainty rather than silently omitting or promoting them.

19. Present CONTROL EFFECTIVENESS OBSERVATIONS separately from security concerns
    when they are present. A blocked attempt may be concerning behavior while also
    demonstrating that a control worked.

20. Present material NEGATIVE FINDINGS only when Agent 3 explicitly lists them.
    A negative finding means a condition was actually evaluated and not observed;
    it is not a claim that unreviewed areas are safe.

21. Present UNRESOLVED AREAS / LIMITATIONS after supported findings. Unresolved
    evidence in one domain must not erase or globally downgrade supported findings
    in other domains.

22. Use a report shape appropriate to broad investigations, normally:
    - Overall assessment / executive summary
    - Supported security findings
    - Investigative leads / potential concerns
    - Control observations
    - Negative or benign findings when material
    - Unresolved areas / limitations
    Omit empty sections rather than inventing content.

23. Do not treat answer brevity as a goal for broad investigations. Be concise
    within each finding, but preserve the breadth of Agent 3's material discoveries.
"""


_FAILED_ANSWER_PROMPT = """
You are the final answer writer for a cybersecurity user-access analysis
system.

USER QUERY
----------
{query}

FAILURE CATEGORY
----------------
{failure_category}

FAILURE REASON
--------------
{failure_reason}

EVIDENCE CONTRACT
-----------------
Required evidence types: {required_evidence_types}
Required output fields: {required_output_fields}
Requires interpretation: {requires_interpretation}

DATABASE STATUS
---------------
Database query succeeded: {database_succeeded}
Database row count: {database_row_count}
Database columns: {database_columns}
Database error: {database_error}

INVESTIGATION STATUS
--------------------
Investigation requested: {investigation_requested}
Investigation succeeded: {investigation_succeeded}
Investigation task summary:
{investigation_summary}

TASK
----
Explain that a verified answer could not be produced.

RULES
-----
1. Do not invent an answer.
2. Do not infer findings from ungraded or incomplete evidence.
3. Explain the material reason in plain language.
4. If some tasks succeeded but the overall request was incomplete, say that
   partial evidence existed but was insufficient.
5. Do not expose internal stack traces unless no clearer reason exists.
6. Do not blame the user.
7. State what evidence or outputs were missing when known.
8. Do not claim that no issues exist when the investigation was incomplete.
9. Do not generate schemas, table names, columns, records, policies, people,
   dates, or findings that are absent from verified context.
10. Do not begin with meta-commentary.
"""


# ============================================================================
# General helpers
# ============================================================================


def _clean_text(
    value: Any,
) -> str:
    return str(value or "").strip()


def _extract_structured_response(
    response: Any,
) -> FinalAnswerResponse:
    raw = getattr(
        response,
        "raw",
        None,
    )

    if isinstance(
        raw,
        FinalAnswerResponse,
    ):
        return raw

    if isinstance(
        raw,
        dict,
    ):
        return FinalAnswerResponse.model_validate(raw)

    if isinstance(
        response,
        FinalAnswerResponse,
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
        FinalAnswerResponse,
    ):
        return parsed

    if isinstance(
        parsed,
        dict,
    ):
        return FinalAnswerResponse.model_validate(parsed)

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
        raise ValueError("The final-answer LLM returned an empty response.")

    payload = json.loads(text)

    return FinalAnswerResponse.model_validate(payload)


def _investigation_summary(
    state: WorkflowState,
) -> str:
    results = (
        getattr(
            state,
            "investigation_results",
            [],
        )
        or []
    )

    if not results:
        return "(none)"

    return "\n".join(
        (
            f"- {result.task_id}: {result.title}; "
            f"success={result.database_query_succeeded}; "
            f"rows={result.database_row_count}; "
            f"columns={result.database_columns}; "
            f"error={result.database_error or 'none'}"
        )
        for result in results
    )


def _available_evidence_types(
    state: WorkflowState,
) -> set[str]:
    method = getattr(
        state,
        "available_evidence_types",
        None,
    )

    if callable(method):
        return set(method())

    available: set[str] = set()

    if _clean_text(
        getattr(
            state,
            "policy_context",
            "",
        )
    ):
        available.add("policy")

    if _clean_text(
        getattr(
            state,
            "schema_evidence",
            "",
        )
    ):
        available.add("schema")

    if _clean_text(
        getattr(
            state,
            "database_evidence",
            "",
        )
    ):
        available.add("database")

    if getattr(
        state,
        "investigation_results",
        [],
    ):
        available.add("investigation")

    return available


def _remove_meta_opening(
    answer: str,
) -> str:
    """Remove common invalid or unhelpful opening statements."""
    cleaned = _clean_text(answer)

    patterns = (
        (r"^\s*the final answer is not applicable.*?" r"(?:\.\s+|\n+)"),
        (
            r"^\s*this format is designed for "
            r"(?:numerical|mathematical) problems.*?"
            r"(?:\.\s+|\n+)"
        ),
        (
            r"^\s*based on the provided "
            r"(?:verified )?evidence[,;:]?\s*"
            r"(?:i will|i can|here is)\s*"
        ),
        (
            r"^\s*i will provide "
            r"(?:a )?(?:clear|concise|professional|detailed)"
            r".*?answer(?: to the user)?[.!]\s*"
        ),
        (r"^\s*as an ai(?: language model)?[,;:]?\s*"),
    )

    for pattern in patterns:
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

    return cleaned


def _contains_invalid_direct_factual_language(
    answer: str,
) -> bool:
    lowered = answer.lower()

    invalid_phrases = (
        "final answer is not applicable",
        "format is designed for numerical problems",
        "format is designed for mathematical problems",
        "no supported finding",
        "does not support a specific answer",
    )

    return any(phrase in lowered for phrase in invalid_phrases)


def _direct_answer_is_complete(
    answer: str,
    refined_context: str,
    state: WorkflowState,
) -> tuple[bool, str]:
    """
    Verify that a direct factual response contains substantive evidence.

    This is a general completeness check. It is not tied to schemas or any
    other specific evidence type.
    """
    cleaned_answer = _clean_text(answer)

    cleaned_context = _clean_text(refined_context)

    if not cleaned_answer:
        return (
            False,
            "The answer is empty.",
        )

    # Reject a heading-only or very short response when the verified context
    # contains substantial factual material.
    if len(cleaned_context) >= 500 and len(cleaned_answer) < 150:
        return (
            False,
            (
                "The response is too short to present the available "
                "verified evidence."
            ),
        )

    answer_words = re.findall(
        r"\b[A-Za-z_][A-Za-z0-9_]*\b",
        cleaned_answer.lower(),
    )

    context_words = set(
        re.findall(
            r"\b[A-Za-z_][A-Za-z0-9_]*\b",
            cleaned_context.lower(),
        )
    )

    substantive_answer_words = {word for word in answer_words if len(word) >= 4}

    supported_words = substantive_answer_words & context_words

    if len(cleaned_context) >= 500 and len(supported_words) < 5:
        return (
            False,
            (
                "The response does not contain enough factual content "
                "from the verified evidence."
            ),
        )

    required_outputs = [
        _clean_text(value).lower()
        for value in getattr(
            state,
            "required_output_fields",
            [],
        )
        if _clean_text(value)
    ]

    answer_lower = cleaned_answer.lower()
    missing_outputs: list[str] = []

    for requirement in required_outputs:
        terminal_name = requirement.rsplit(
            ".",
            1,
        )[-1]

        if requirement not in answer_lower and terminal_name not in answer_lower:
            missing_outputs.append(requirement)

    if missing_outputs:
        return (
            False,
            ("The response omitted required outputs: " + ", ".join(missing_outputs)),
        )

    if bool(
        getattr(
            state,
            "requires_complete_output",
            False,
        )
    ):
        rows_complete, row_reason = _direct_query_rows_are_complete(
            answer=cleaned_answer,
            state=state,
        )

        if not rows_complete:
            return False, row_reason

    return (
        True,
        "",
    )


def _normalized_database_rows(
    state: WorkflowState,
) -> list[dict[str, Any]]:
    """Return verified database rows as dictionaries."""
    rows = getattr(state, "database_rows", []) or []

    return [row for row in rows if isinstance(row, dict)]


def _format_database_value(
    value: Any,
) -> str:
    if value is None:
        return ""

    if isinstance(value, bool):
        return "True" if value else "False"

    return str(value).strip()


def _direct_query_rows_are_complete(
    *,
    answer: str,
    state: WorkflowState,
) -> tuple[bool, str]:
    """
    Verify that a complete-output DirectQuery rewrite did not omit rows.

    Python checks preservation of verified row values; it does not decide how
    the LLM should phrase or organize them.
    """
    if not _is_direct_query_state(state):
        return True, ""

    rows = _normalized_database_rows(state)

    if not rows:
        return True, ""

    answer_lower = _clean_text(answer).lower()

    if not answer_lower:
        return False, "The DirectQuery answer is empty."

    missing_rows: list[int] = []

    for index, row in enumerate(rows, start=1):
        values = [
            _format_database_value(value)
            for value in row.values()
            if _format_database_value(value)
        ]

        if not values:
            continue

        if not all(value.lower() in answer_lower for value in values):
            missing_rows.append(index)

    if missing_rows:
        preview = ", ".join(str(i) for i in missing_rows[:10])
        suffix = "" if len(missing_rows) <= 10 else f" (+{len(missing_rows) - 10} more)"

        return (
            False,
            "The DirectQuery rewrite omitted verified database row(s): "
            f"{preview}{suffix}.",
        )

    return True, ""


def _deterministic_direct_query_answer(
    state: WorkflowState,
) -> str:
    """
    Render verified DirectQuery rows without LLM rewriting.

    Used only when a complete-output LLM presentation fails deterministic
    completeness or fidelity checks.
    """
    rows = _normalized_database_rows(state)

    if not rows:
        return _present_verified_evidence(state)

    columns = getattr(state, "database_columns", []) or list(rows[0].keys())
    columns = [str(column) for column in columns]

    if len(rows) == 1 and len(columns) == 1:
        return _format_database_value(rows[0].get(columns[0]))

    lowered_columns = {column.lower() for column in columns}

    if lowered_columns.issubset(
        {
            "first_name",
            "middle_name",
            "last_name",
            "full_name",
            "employee_name",
        }
    ):
        lines: list[str] = []

        for row in rows:
            if "full_name" in row:
                name = _format_database_value(row.get("full_name"))
            elif "employee_name" in row:
                name = _format_database_value(row.get("employee_name"))
            else:
                parts = [
                    _format_database_value(row.get(key))
                    for key in (
                        "first_name",
                        "middle_name",
                        "last_name",
                    )
                ]
                name = " ".join(part for part in parts if part)

            if name:
                lines.append(f"- {name}")

        if lines:
            return "\n".join(lines)

    lines: list[str] = []

    for row in rows:
        parts = [
            f"{column}: {_format_database_value(row.get(column))}" for column in columns
        ]
        lines.append("- " + "; ".join(parts))

    return "\n".join(lines)


_UNDERSCORE_IDENTIFIER_PATTERN = re.compile(
    r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b"
)

_DOTTED_IDENTIFIER_PATTERN = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*\." r"[A-Za-z_][A-Za-z0-9_]*\b"
)

_CONTROL_IDENTIFIER_PATTERN = re.compile(r"\b[A-Z]{1,8}-\d+(?:\(\d+\))?\b")

_FILENAME_PATTERN = re.compile(
    r"\b[A-Za-z0-9_.-]+\." r"(?:md|py|json|yaml|yml|sql|csv|txt)\b",
    flags=re.IGNORECASE,
)


def _extract_technical_identifiers(
    text: str,
) -> set[str]:
    """
    Extract code-like identifiers whose spelling should remain exact.

    The validator is evidence-type agnostic. It applies to database objects,
    policy/control IDs, filenames, qualified fields, API-style identifiers,
    and other technical names.
    """
    cleaned = _clean_text(text)

    identifiers: set[str] = set()

    for pattern in (
        _UNDERSCORE_IDENTIFIER_PATTERN,
        _DOTTED_IDENTIFIER_PATTERN,
        _CONTROL_IDENTIFIER_PATTERN,
        _FILENAME_PATTERN,
    ):
        identifiers.update(
            match.group(0).lower() for match in pattern.finditer(cleaned)
        )

    return identifiers


def _technical_identifier_fidelity(
    answer: str,
    refined_context: str,
) -> tuple[
    bool,
    list[str],
]:
    """
    Verify that every technical identifier in the answer exists verbatim in
    the verified context.

    Returns:
        (is_faithful, unsupported_identifiers)
    """
    answer_identifiers = _extract_technical_identifiers(answer)

    context_identifiers = _extract_technical_identifiers(refined_context)

    unsupported = sorted(answer_identifiers - context_identifiers)

    return (
        not unsupported,
        unsupported,
    )


def _detect_unsupported_id_as_name(
    answer: str,
    database_columns: list[str],
) -> bool:
    columns = {_clean_text(column).lower() for column in database_columns}

    has_name_columns = any(
        column in columns
        for column in (
            "first_name",
            "middle_name",
            "last_name",
            "full_name",
            "employee_name",
        )
    )

    if has_name_columns:
        return False

    return (
        re.search(
            r"\bemployee\s+\d+\b",
            answer,
            flags=re.IGNORECASE,
        )
        is not None
    )


def _detect_unsupported_date_claim(
    answer: str,
    refined_context: str,
) -> bool:
    date_patterns = (
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        (
            r"\b(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{1,2},\s+\d{4}\b"
        ),
    )

    answer_dates: set[str] = set()
    context_dates: set[str] = set()

    for pattern in date_patterns:
        answer_dates.update(
            match.group(0).lower()
            for match in re.finditer(
                pattern,
                answer,
                flags=re.IGNORECASE,
            )
        )

        context_dates.update(
            match.group(0).lower()
            for match in re.finditer(
                pattern,
                refined_context,
                flags=re.IGNORECASE,
            )
        )

    return bool(answer_dates - context_dates)


def _detect_unsupported_filter_language(
    answer: str,
    refined_context: str,
) -> bool:
    risky_phrases = (
        "entered before",
        "entered after",
        "accessed before",
        "accessed after",
        "only includes",
        "meet the criteria",
        "whose name is",
    )

    answer_lower = answer.lower()
    context_lower = refined_context.lower()

    return any(
        phrase in answer_lower and phrase not in context_lower
        for phrase in risky_phrases
    )


def _strip_outer_evidence_marker(
    refined_context: str,
) -> str:
    """Remove one outer evidence marker without changing verified content."""
    cleaned = _clean_text(refined_context)

    outer_markers = (
        "=== POLICY EVIDENCE ===",
        "=== SCHEMA EVIDENCE ===",
        "=== DATABASE EVIDENCE ===",
        "=== AUTHORITATIVE SCHEMA ===",
    )

    for marker in outer_markers:
        if cleaned.startswith(marker):
            return cleaned[len(marker) :].lstrip("\n ")

    return cleaned


def _is_direct_query_state(
    state: WorkflowState,
) -> bool:
    """Return True when the current answer is for RetrievalMode.DIRECT_QUERY."""
    raw_mode = getattr(
        state,
        "retrieval_mode",
        RetrievalMode.EVIDENCE_ONLY,
    )

    if isinstance(
        raw_mode,
        RetrievalMode,
    ):
        return raw_mode == RetrievalMode.DIRECT_QUERY

    return _clean_text(raw_mode).lower() == RetrievalMode.DIRECT_QUERY.value


def _direct_query_database_context(
    state: WorkflowState,
) -> str:
    """
    Return only verified database evidence for DirectQuery presentation.

    This is also used as the safe fallback if LLM presentation fails.
    """
    evidence = _clean_text(
        getattr(
            state,
            "database_evidence",
            "",
        )
    )

    if not evidence:
        return ""

    lines = [
        "=== VERIFIED DATABASE EVIDENCE ===",
        f"Rows: {getattr(state, 'database_row_count', 0)}",
        f"Columns: {getattr(state, 'database_columns', [])}",
        "Evidence:",
        evidence,
    ]

    return "\n".join(lines).strip()


def _present_verified_evidence(
    state: WorkflowState,
) -> str:
    """Present complete verified evidence without generative rewriting."""
    if _is_direct_query_state(state):
        direct_context = _direct_query_database_context(state)

        if direct_context:
            return _strip_outer_evidence_marker(direct_context)

    refined = _clean_text(state.refined_context)

    if not refined:
        return (
            "The evidence passed grading, but no refined evidence text "
            "was available for presentation."
        )

    return _strip_outer_evidence_marker(refined)


def _resolve_presentation_mode(
    state: WorkflowState,
    grade_text: str,
    refined_context: str,
) -> PresentationMode:
    """Resolve Agent 3's presentation contract with backward compatibility."""
    if grade_text != GradeResult.PASS.value or not refined_context:
        return PresentationMode.FAILURE

    raw_mode = getattr(
        state,
        "presentation_mode",
        PresentationMode.AUTO,
    )

    if isinstance(
        raw_mode,
        PresentationMode,
    ):
        mode = raw_mode

    else:
        try:
            mode = PresentationMode(_clean_text(raw_mode).lower())

        except ValueError:
            mode = PresentationMode.AUTO

    if mode != PresentationMode.AUTO:
        return mode

    if bool(
        getattr(
            state,
            "requires_interpretation",
            False,
        )
    ):
        return PresentationMode.INTERPRET

    if not bool(
        getattr(
            state,
            "allow_llm_rewrite",
            True,
        )
    ):
        return PresentationMode.VERBATIM

    return PresentationMode.SUMMARY


def _extract_investigation_report_items(
    refined_context: str,
    heading: str,
) -> list[str]:
    """Extract bullet items from one Agent 3 investigation-report section."""
    text = _clean_text(refined_context)
    if not text:
        return []

    pattern = rf"(?is)===\s*{re.escape(heading)}\s*===\s*(.*?)" rf"(?=\n\s*===|\Z)"
    match = re.search(pattern, text)
    if not match:
        return []

    return [
        item.strip()
        for item in re.findall(r"(?m)^\s*-\s+(.+?)\s*$", match.group(1))
        if item.strip()
    ]


def _investigation_answer_is_complete(
    *,
    answer: str,
    refined_context: str,
) -> tuple[bool, str]:
    """
    Guard against Agent 4 compressing a broad investigation to one discovery.

    This validator does not decide which findings are security risks. It only
    checks that material findings already selected by Agent 3 are represented.
    """
    findings = _extract_investigation_report_items(
        refined_context,
        "CONFIRMED / SUPPORTED SECURITY FINDINGS",
    )
    leads = _extract_investigation_report_items(
        refined_context,
        "INVESTIGATIVE LEADS / POTENTIAL CONCERNS",
    )
    report_items = findings + leads

    if not report_items:
        return True, ""

    answer_lower = _clean_text(answer).lower()
    if not answer_lower:
        return False, "The investigation answer is empty."

    stop = {
        "that",
        "this",
        "with",
        "from",
        "were",
        "was",
        "have",
        "has",
        "into",
        "when",
        "where",
        "which",
        "their",
        "there",
        "been",
        "evidence",
        "security",
        "finding",
        "records",
        "record",
        "supported",
    }

    missing: list[int] = []
    for index, finding in enumerate(report_items, start=1):
        tokens = {
            token.lower()
            for token in re.findall(r"\b[A-Za-z0-9_:-]{4,}\b", finding)
            if token.lower() not in stop
        }
        if not tokens:
            continue

        present = sum(1 for token in tokens if token in answer_lower)
        required = max(2, min(5, (len(tokens) + 2) // 3))
        if present < required:
            missing.append(index)

    if missing:
        return (
            False,
            "The final rewrite appears to omit Agent 3 finding/lead item(s): "
            + ", ".join(str(i) for i in missing[:10])
            + ("." if len(missing) <= 10 else " and additional findings."),
        )

    # A multi-finding investigation should not collapse to a tiny paragraph even
    # when a few shared keywords happen to survive the rewrite.
    if len(report_items) >= 3 and len(_clean_text(answer)) < 600:
        return (
            False,
            "The final investigation rewrite is too short to preserve the "
            f"breadth of {len(report_items)} supported findings/investigative leads.",
        )

    return True, ""


def _safe_direct_factual_fallback(
    state: WorkflowState,
) -> str:
    """
    Return verified evidence directly when the LLM cannot reliably transform
    direct factual evidence.

    This is preferable to inventing or rejecting a supported factual answer.
    """
    if _is_direct_query_state(state):
        direct_context = _direct_query_database_context(state)

        if direct_context:
            return _strip_outer_evidence_marker(direct_context)

    refined = _clean_text(state.refined_context)

    if not refined:
        return (
            "The requested factual evidence was graded as available, but no "
            "refined evidence text was provided."
        )

    return refined


def _safe_interpretive_fallback(
    state: WorkflowState,
) -> str:
    """
    Preserve Agent 3's verified refined interpretation when the final-answer
    rewrite fails fidelity checks.

    Agent 4 is a renderer, not a second grader.
    """
    refined = _clean_text(
        getattr(
            state,
            "refined_context",
            "",
        )
    )

    if refined:
        return _strip_outer_evidence_marker(refined)

    row_count = getattr(
        state,
        "database_row_count",
        0,
    )

    columns = (
        getattr(
            state,
            "database_columns",
            [],
        )
        or []
    )

    lines = [
        "The evidence passed Agent 3 grading, but the final interpretive rewrite "
        "could not be rendered safely."
    ]

    if row_count:
        lines.append(f"The verified database evidence contained {row_count} row(s).")

    if columns:
        lines.append(
            "Returned fields: " + ", ".join(str(column) for column in columns) + "."
        )

    return "\n\n".join(lines)


def _safe_failed_answer(
    state: WorkflowState,
) -> str:
    category = (
        _clean_text(
            getattr(
                state,
                "evidence_failure_category",
                "",
            )
        )
        or "incomplete_evidence"
    )

    reason = (
        _clean_text(
            getattr(
                state,
                "evidence_failure_reason",
                "",
            )
        )
        or _clean_text(
            getattr(
                state,
                "database_error",
                "",
            )
        )
        or (
            "The available evidence did not completely support the " "requested answer."
        )
    )

    investigation_requested = bool(
        getattr(
            state,
            "investigation_requested",
            False,
        )
    )

    if investigation_requested:
        opening = (
            "I could not complete a verified security review of the company " "records."
        )

        closing = (
            "This does not mean that no issues exist; it means the available "
            "evidence was insufficient to determine the requested result."
        )

    else:
        opening = (
            "I could not produce a verified answer from the available " "evidence."
        )

        closing = "No conclusion was generated from incomplete or unverified data."

    return (
        f"{opening}\n\n"
        f"**Failure category:** {category}\n\n"
        f"**Reason:** {reason}\n\n"
        f"{closing}"
    )


def _append_limitations(
    answer: str,
    limitations: list[str],
) -> str:
    cleaned_limitations = [
        _clean_text(item) for item in limitations if _clean_text(item)
    ]

    if not cleaned_limitations:
        return answer

    return (
        answer.rstrip()
        + "\n\n"
        + "**Limitations**\n"
        + "\n".join(f"- {item}" for item in cleaned_limitations)
    )


# ============================================================================
# AnswerGeneratorAgent
# ============================================================================


def _normalize_investigation_outcome(value: Any) -> str:
    """Normalize common Agent 3 investigation outcome values."""
    cleaned = _clean_text(getattr(value, "value", value)).lower()

    if not cleaned:
        return ""

    if "inconclusive" in cleaned or "insufficient" in cleaned:
        return "inconclusive"

    if cleaned in {
        "supported",
        "confirmed",
        "positive",
        "finding",
        "risk",
        "issue",
    }:
        return "supported"

    if cleaned in {
        "not_supported",
        "unsupported",
        "negative",
        "no_finding",
        "clear",
    }:
        return "not_supported"

    return cleaned


def _agent3_final_investigation_outcome(
    state: WorkflowState,
) -> str:
    """
    Resolve Agent 3's final investigation outcome.

    Prefer explicit structured state attributes when available. This keeps Agent
    4 compatible with future WorkflowState/GraderWriter versions that expose a
    dedicated outcome field. For current states, parse only FINAL/labeled
    assessment text from Agent 3's refined context rather than relying on one
    exact phrase.
    """
    if not getattr(state, "investigation_requested", False):
        return ""

    # Prefer structured outcome fields if current/future state versions expose
    # one. Unknown fields are harmless because getattr defaults to None.
    for attribute in (
        "investigation_outcome",
        "overall_security",
        "overall_security_assessment",
        "overall_assessment",
        "security_assessment",
        "investigation_assessment",
    ):
        normalized = _normalize_investigation_outcome(getattr(state, attribute, None))
        if normalized:
            return normalized

    refined = _clean_text(getattr(state, "refined_context", ""))
    if not refined:
        return ""

    # Agent 3's current investigation interpretation starts with an explicit
    # OVERALL SECURITY OPINION section whose first labeled value is Assessment.
    # Resolve that authoritative value before scanning later per-task or
    # inconclusive-observation text. This prevents a later diagnostic word such
    # as "inconclusive" from overriding the actual overall assessment.
    overall_section = re.search(
        r"(?is)===\s*OVERALL SECURITY OPINION\s*===\s*.*?^\s*Assessment\s*[:=\-]\s*([^\n]+)",
        refined,
        flags=re.MULTILINE,
    )
    if overall_section:
        normalized = _normalize_investigation_outcome(overall_section.group(1))
        if normalized:
            return normalized

    # Backward compatibility for older Agent 3 refined-context formats. Look for
    # explicitly labeled FINAL/overall/security assessment values.
    labeled_patterns = (
        r"(?im)^\s*(?:final\s+)?overall\s+security(?:\s+assessment)?\s*[:=\-]\s*([^\n]+)",
        r"(?im)^\s*(?:final\s+)?overall\s+assessment\s*[:=\-]\s*([^\n]+)",
        r"(?im)^\s*(?:final\s+)?security\s+assessment\s*[:=\-]\s*([^\n]+)",
        r"(?im)^\s*(?:final\s+)?investigation\s+(?:outcome|assessment|conclusion)\s*[:=\-]\s*([^\n]+)",
    )

    for pattern in labeled_patterns:
        match = re.search(pattern, refined)
        if match:
            normalized = _normalize_investigation_outcome(match.group(1))
            if normalized:
                return normalized

    # Do not infer the overall result from unlabeled diagnostic prose. Refined
    # context can legitimately contain per-task "inconclusive" observations even
    # when Agent 3's overall assessment is a supported finding. If no explicit
    # overall outcome can be resolved, leave it unspecified and follow the
    # presentation contract rather than guessing.
    return ""


def _agent3_final_investigation_is_inconclusive(
    state: WorkflowState,
) -> bool:
    """Return True when Agent 3's resolved final investigation outcome is inconclusive."""
    return _agent3_final_investigation_outcome(state) == "inconclusive"


def _agent3_final_investigation_is_incomplete(
    state: WorkflowState,
    *,
    grade_text: str,
    presentation_mode: PresentationMode,
) -> bool:
    """
    Return True only when Agent 3's FINAL graded state explicitly requires
    failure/incomplete presentation.

    Ledger labels and per-task inconclusive diagnostics must not override a
    final PASS + INTERPRET/SUMMARY/VERBATIM contract.
    """
    if not getattr(state, "investigation_requested", False):
        return False

    if grade_text != GradeResult.PASS.value:
        return True

    if presentation_mode == PresentationMode.FAILURE:
        return True

    category = _clean_text(
        getattr(
            state,
            "evidence_failure_category",
            "",
        )
    ).lower()

    repairable = bool(
        getattr(
            state,
            "evidence_failure_repairable",
            False,
        )
    )

    failure_reason = _clean_text(
        getattr(
            state,
            "evidence_failure_reason",
            "",
        )
    )

    return bool(category == "incomplete_evidence" and (repairable or failure_reason))


def _inconclusive_investigation_answer(
    state: WorkflowState,
) -> str:
    """Preserve Agent 3's explicit final inconclusive outcome deterministically."""
    refined = _clean_text(getattr(state, "refined_context", ""))

    # Prefer a concise supported explanation from the refined context.
    sentences = re.split(r"(?<=[.!?])\s+", refined)
    relevant = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
        and any(
            token in sentence.lower()
            for token in (
                "inconclusive",
                "insufficient",
                "missing",
                "not establish",
                "unresolved",
            )
        )
    ]

    detail = " ".join(relevant[:3]).strip()

    if not detail:
        detail = (
            "The verified evidence did not establish the requested condition "
            "well enough to support either a positive or negative conclusion."
        )

    return (
        "The investigation is inconclusive. "
        + detail
        + " This does not establish that the condition occurred or that it did not occur."
    )


def _incomplete_investigation_answer(
    state: WorkflowState,
) -> str:
    """Render Agent 3's unresolved evidence limitation without LLM inference."""
    reason = _clean_text(
        getattr(
            state,
            "evidence_failure_reason",
            "",
        )
    )

    if not reason:
        reason = (
            "The investigation did not retrieve all evidence needed to evaluate "
            "the requested condition. The available records may describe one "
            "related activity domain, but they do not establish the requested "
            "cross-domain relationship."
        )

    return (
        "The investigation is inconclusive because the verified evidence is "
        "incomplete. "
        + reason
        + " No conclusion that the condition did or did not occur can be made "
        "from the evidence retrieved."
    )


class AnswerGeneratorAgent:
    """Agent 4 – write the final answer from graded evidence only."""

    def __init__(
        self,
        llm: LLM,
        schema_retriever: Any | None = None,
    ) -> None:
        self._structured_llm = llm.as_structured_llm(FinalAnswerResponse)

        print("\n[AnswerGeneratorAgent] ⚙  Initialising final answer writer…")
        print(
            "[AnswerGeneratorAgent]    Framework : "
            "presentation contract: verbatim + summary + interpret + failure"
        )
        print("[AnswerGeneratorAgent]    MCP tools : none")
        print("[AnswerGeneratorAgent]    SQL       : none")
        print("[AnswerGeneratorAgent]    Retrieval : none")
        print(
            "[AnswerGeneratorAgent]    Rule      : "
            "Agent 3 final grade/presentation is authoritative; Agent 4 does not re-grade investigation sufficiency"
        )
        print(
            "[AnswerGeneratorAgent]    DirectQuery : "
            "LLM presentation + deterministic row-completeness fallback"
        )

    async def run(
        self,
        state: WorkflowState,
    ) -> WorkflowState:
        """Render the final answer according to Agent 3's presentation contract."""
        query = _clean_text(state.query)

        refined_context = _clean_text(state.refined_context)

        # Defense in depth: DirectQuery presentation must be based on verified
        # database evidence, never on the full schema catalog accidentally
        # carried in refined_context.
        if _is_direct_query_state(state):
            direct_context = _direct_query_database_context(state)

            if direct_context:
                refined_context = direct_context

        grade = state.grade

        grade_text = (
            grade.value
            if isinstance(
                grade,
                GradeResult,
            )
            else _clean_text(grade).lower()
        )

        available_types = sorted(_available_evidence_types(state))

        presentation_mode = _resolve_presentation_mode(
            state=state,
            grade_text=grade_text,
            refined_context=refined_context,
        )

        allow_llm_rewrite = bool(
            getattr(
                state,
                "allow_llm_rewrite",
                True,
            )
        )

        allow_summary = bool(
            getattr(
                state,
                "allow_summary",
                True,
            )
        )

        requires_complete_output = bool(
            getattr(
                state,
                "requires_complete_output",
                False,
            )
        )

        print("\n" + "═" * 70)
        print("[AnswerGeneratorAgent] ▶  STARTING — " "Agent 4: Final Answer Renderer")
        print("[AnswerGeneratorAgent]    Query          : " f"{query}")
        print(
            "[AnswerGeneratorAgent]    Route          : "
            f"{getattr(state.route, 'value', state.route) or 'N/A'}"
        )
        print("[AnswerGeneratorAgent]    Grade          : " f"{grade_text or 'none'}")
        print(
            "[AnswerGeneratorAgent]    DirectQuery    : "
            f"{_is_direct_query_state(state)}"
        )
        print(
            "[AnswerGeneratorAgent]    Refined ctx    : "
            f"{len(refined_context)} chars"
        )
        print(
            "[AnswerGeneratorAgent]    Presentation   : " f"{presentation_mode.value}"
        )
        print("[AnswerGeneratorAgent]    LLM rewrite    : " f"{allow_llm_rewrite}")
        print(
            "[AnswerGeneratorAgent]    Complete output: " f"{requires_complete_output}"
        )
        print("[AnswerGeneratorAgent]    Evidence types : " f"{available_types}")
        print(
            "[AnswerGeneratorAgent]    Agent 3 outcome  : "
            f"{_agent3_final_investigation_outcome(state) or 'unspecified'}"
        )
        print(
            "[AnswerGeneratorAgent]    Agent 3 inconcl. : "
            f"{_agent3_final_investigation_is_inconclusive(state)}"
        )
        print(
            "[AnswerGeneratorAgent]    Agent 3 final    : "
            f"grade={grade_text}; presentation={presentation_mode.value}; "
            f"failure_category={_clean_text(getattr(state, 'evidence_failure_category', '')) or 'none'}; "
            f"repairable={bool(getattr(state, 'evidence_failure_repairable', False))}"
        )
        print("─" * 70)

        if _agent3_final_investigation_is_incomplete(
            state,
            grade_text=grade_text,
            presentation_mode=presentation_mode,
        ):
            answer = _incomplete_investigation_answer(state)
            mode = "INCOMPLETE INVESTIGATION"

            state.add_warning(
                "Agent 4 used incomplete-investigation presentation because "
                "Agent 3's FINAL graded state explicitly remained incomplete."
            )

        elif presentation_mode == PresentationMode.FAILURE:
            answer = _safe_failed_answer(state)
            mode = "FAILURE"

        elif presentation_mode == PresentationMode.VERBATIM:
            answer = _present_verified_evidence(state)
            mode = "VERBATIM"

        elif presentation_mode == PresentationMode.SUMMARY:
            if not allow_llm_rewrite or not allow_summary:
                state.add_warning(
                    (
                        "SUMMARY presentation was requested, but the "
                        "presentation contract prohibited rewriting or "
                        "summarization. Verified evidence was presented "
                        "without rewriting."
                    )
                )

                answer = _present_verified_evidence(state)
                mode = "VERBATIM CONTRACT FALLBACK"

            else:
                prompt = _SUMMARY_ANSWER_PROMPT.format(
                    query=query,
                    refined_context=refined_context,
                    ledger_context=state.investigation_ledger.compact_summary(),
                    required_evidence_types=(
                        getattr(
                            state,
                            "required_evidence_types",
                            [],
                        )
                    ),
                    required_output_fields=(
                        getattr(
                            state,
                            "required_output_fields",
                            [],
                        )
                    ),
                    available_evidence_types=(available_types),
                )

                try:
                    parsed = _extract_structured_response(
                        _complete_with_system(
                            self._structured_llm,
                            system_prompt=ANSWER_GENERATOR_SYSTEM_PROMPT,
                            user_prompt=prompt,
                        )
                    )

                    answer = _remove_meta_opening(parsed.answer)

                    invalid_types = {
                        _clean_text(item).lower()
                        for item in parsed.used_evidence_types
                        if _clean_text(item)
                    } - set(available_types)

                    (
                        identifiers_are_faithful,
                        unsupported_identifiers,
                    ) = _technical_identifier_fidelity(
                        answer=answer,
                        refined_context=refined_context,
                    )

                    fidelity_failed = any(
                        (
                            not answer,
                            bool(invalid_types),
                            not identifiers_are_faithful,
                            _contains_invalid_direct_factual_language(answer),
                            _detect_unsupported_id_as_name(
                                answer,
                                state.database_columns,
                            ),
                            _detect_unsupported_date_claim(
                                answer,
                                refined_context,
                            ),
                        )
                    )

                    if not identifiers_are_faithful:
                        state.add_warning(
                            (
                                "Summary identifier fidelity check failed. "
                                "Unsupported identifiers: "
                                + ", ".join(unsupported_identifiers)
                            )
                        )

                    completeness_failed = False
                    completeness_reason = ""

                    if not fidelity_failed and requires_complete_output:
                        (
                            is_complete,
                            completeness_reason,
                        ) = _direct_answer_is_complete(
                            answer=answer,
                            refined_context=refined_context,
                            state=state,
                        )

                        completeness_failed = not is_complete

                        if completeness_failed:
                            state.add_warning(
                                "DirectQuery completeness check failed. "
                                + completeness_reason
                            )

                    if fidelity_failed or completeness_failed:
                        if _is_direct_query_state(state):
                            answer = _deterministic_direct_query_answer(state)
                            mode = "DIRECTQUERY COMPLETE-OUTPUT FALLBACK"
                        else:
                            answer = _present_verified_evidence(state)
                            mode = "VERBATIM FIDELITY FALLBACK"

                    else:
                        answer = _append_limitations(
                            answer,
                            parsed.limitations,
                        )
                        mode = "SUMMARY"

                except Exception as exc:
                    state.add_warning(
                        ("Summary generation failed: " f"{type(exc).__name__}: {exc}")
                    )

                    if _is_direct_query_state(state):
                        answer = _deterministic_direct_query_answer(state)
                        mode = "DIRECTQUERY GENERATION FALLBACK"
                    else:
                        answer = _present_verified_evidence(state)
                        mode = "VERBATIM GENERATION FALLBACK"

        elif presentation_mode == PresentationMode.INTERPRET:
            if not allow_llm_rewrite:
                state.add_warning(
                    (
                        "INTERPRET presentation was requested, but the "
                        "presentation contract prohibited LLM rewriting. "
                        "Verified evidence was presented without interpretation."
                    )
                )

                answer = _present_verified_evidence(state)
                mode = "VERBATIM CONTRACT FALLBACK"

            else:
                prompt = _INTERPRETIVE_ANSWER_PROMPT.format(
                    query=query,
                    refined_context=refined_context,
                    ledger_context=state.investigation_ledger.compact_summary(),
                    required_evidence_types=(
                        getattr(
                            state,
                            "required_evidence_types",
                            [],
                        )
                    ),
                    required_output_fields=(
                        getattr(
                            state,
                            "required_output_fields",
                            [],
                        )
                    ),
                    available_evidence_types=(available_types),
                    evidence_requirement_reason=(
                        getattr(
                            state,
                            "evidence_requirement_reason",
                            "",
                        )
                        or "(none)"
                    ),
                    database_row_count=(state.database_row_count),
                    database_columns=(state.database_columns),
                    policy_sources=(state.policy_sources),
                    schema_sources=(state.schema_sources),
                    investigation_summary=(_investigation_summary(state)),
                )

                try:
                    parsed = _extract_structured_response(
                        _complete_with_system(
                            self._structured_llm,
                            system_prompt=ANSWER_GENERATOR_SYSTEM_PROMPT,
                            user_prompt=prompt,
                        )
                    )

                    answer = _remove_meta_opening(parsed.answer)

                    invalid_types = {
                        _clean_text(value).lower()
                        for value in parsed.used_evidence_types
                        if _clean_text(value)
                    } - set(available_types)

                    (
                        identifiers_are_faithful,
                        unsupported_identifiers,
                    ) = _technical_identifier_fidelity(
                        answer=answer,
                        refined_context=refined_context,
                    )

                    fidelity_failed = any(
                        (
                            not answer,
                            bool(invalid_types),
                            not identifiers_are_faithful,
                            _detect_unsupported_id_as_name(
                                answer,
                                state.database_columns,
                            ),
                            _detect_unsupported_date_claim(
                                answer,
                                refined_context,
                            ),
                            _detect_unsupported_filter_language(
                                answer,
                                refined_context,
                            ),
                        )
                    )

                    if not identifiers_are_faithful:
                        state.add_warning(
                            (
                                "Interpretive identifier fidelity check "
                                "failed. Unsupported identifiers: "
                                + ", ".join(unsupported_identifiers)
                            )
                        )

                    investigation_complete = True
                    investigation_completeness_reason = ""

                    if not fidelity_failed and getattr(
                        state, "investigation_requested", False
                    ):
                        (
                            investigation_complete,
                            investigation_completeness_reason,
                        ) = _investigation_answer_is_complete(
                            answer=answer,
                            refined_context=refined_context,
                        )

                        if not investigation_complete:
                            state.add_warning(
                                "Investigation presentation completeness check "
                                "failed. " + investigation_completeness_reason
                            )

                    if fidelity_failed or not investigation_complete:
                        answer = _safe_interpretive_fallback(state)
                        mode = (
                            "INTERPRET FIDELITY FALLBACK"
                            if fidelity_failed
                            else "INTERPRET COMPLETENESS FALLBACK"
                        )

                    else:
                        answer = _append_limitations(
                            answer,
                            parsed.limitations,
                        )
                        mode = "INTERPRET"

                except Exception as exc:
                    state.add_warning(
                        (
                            "Interpretive answer generation failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    )

                    answer = _safe_interpretive_fallback(state)
                    mode = "INTERPRET GENERATION FALLBACK"

        else:
            state.add_warning(
                (
                    "No recognized presentation contract was available. "
                    "Verified evidence was presented without rewriting."
                )
            )

            answer = _present_verified_evidence(state)
            mode = "VERBATIM AUTO FALLBACK"

        state.answer = answer

        state.add_trace(
            "AnswerGeneratorAgent",
            (
                f"Final answer mode={mode}; "
                f"contract={presentation_mode.value}; "
                f"length={len(answer)}."
            ),
        )

        print("[AnswerGeneratorAgent] ✔  COMPLETE")
        print("[AnswerGeneratorAgent]    Mode         : " f"{mode}")
        print("[AnswerGeneratorAgent]    Answer length: " f"{len(answer)} chars")
        print("[AnswerGeneratorAgent]    Preview      : " f"{answer[:160]}")
        print("═" * 70 + "\n")

        return state
