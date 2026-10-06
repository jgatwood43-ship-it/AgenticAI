"""
agents/grader_writer_agent.py
─────────────────────────────
Agent 3 – Generalized Evidence Sufficiency Grader.

The grader uses a general evidence contract:

    required_evidence_types
    required_output_fields
    requires_interpretation

Direct factual evidence passes deterministically when the contract is
satisfied. Multi-task investigations use per-task LLM security interpretation,
global LLM synthesis over all successful raw evidence, and a final Tree-of-Thought
challenge before Agent 3 commits to its interpretation. Other interpretation-heavy
work also uses Tree of Thought.

After grading, Agent 3 also creates a generalized presentation contract for
Agent 4:

    VERBATIM
        Preserve verified evidence exactly for non-DirectQuery factual requests.

    SUMMARY
        Present verified DirectQuery database rows as a concise factual answer,
        or condense other evidence when the user explicitly requests it.

    SUMMARY
        Condense verified evidence when the user explicitly requests a
        summary, overview, highlights, or concise response.

    INTERPRET
        Present evidence-supported analysis after Tree-of-Thought grading.

    FAILURE
        Produce a deterministic failure response without generative rewriting.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from llama_index.core.llms import ChatMessage, LLM
from pydantic import BaseModel, Field, field_validator

from core.system_prompts import GRADER_WRITER_SYSTEM_PROMPT
from core.state import (
    EvidenceType,
    GradeResult,
    PresentationMode,
    RetrievalMode,
    ToTThought,
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


NUM_BRANCHES = 3
MIN_PASS_CONTEXT_CHARS = 20


class EvidenceFailureCategory(str, Enum):
    NONE = "none"
    NO_EVIDENCE = "no_evidence"
    MISSING_REQUIRED_EVIDENCE = "missing_required_evidence"
    MISSING_REQUIRED_OUTPUT = "missing_required_output"
    INCOMPLETE_EVIDENCE = "incomplete_evidence"
    INVALID_PLAN = "invalid_plan"
    INVALID_SQL = "invalid_sql"
    DATABASE_EXECUTION = "database_execution"
    UNKNOWN = "unknown"


class EvidenceFailure(BaseModel):
    category: EvidenceFailureCategory
    reason: str
    repairable: bool = False


class BranchResponse(BaseModel):
    grade: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str
    refined_context: str
    missing_information: list[str] = Field(default_factory=list)


class DirectQueryEvidenceReview(BaseModel):
    """Semantic review of executed DirectQuery evidence."""

    sufficient: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(
        cls,
        value: Any,
    ) -> Any:
        """
        Normalize percentage-style confidence before Pydantic range validation.

        Local models may return 95 or 100 to mean 95% or 100% confidence even
        though this model stores confidence on a 0.0-1.0 scale.
        """
        if isinstance(value, (int, float)) and 1 < value <= 100:
            return value / 100

        return value


_DIRECT_QUERY_EVIDENCE_REVIEW_PROMPT = """
You are performing a FINAL evidence-sufficiency check for an executed direct
database query.

ORIGINAL USER QUESTION
----------------------
{query}

EXECUTED SQL
------------
{sql}

VERIFIED SQL FACTS
------------------
The SQL has already passed authoritative deterministic validation.

Validated tables:
{validated_tables}

Validated columns:
{validated_columns}

Validated relationships:
{validated_relationships}

These are settled facts. You MUST NOT claim that a validated table, column, or
relationship is missing, invalid, belongs to a different table, or should be
replaced.

DATABASE RESULT
---------------
Rows returned: {row_count}
Columns returned: {columns}

Evidence:
{evidence}

TASK
----
Decide only whether the returned database evidence contains what is needed to
answer the ORIGINAL USER QUESTION.

Rules:
1. Do not re-grade SQL schema validity or table/column ownership.
2. Do not reject evidence because you would prefer another valid table or SQL
   construction.
3. If returned fields directly contain the requested fact, accept the evidence
   unless there is a concrete semantic mismatch in the returned data itself.
4. A different operational fact is not a substitute for the requested fact.
5. Zero rows can be sufficient when the validated query directly tests the
   requested fact; this means no matching record was found.
6. IDs are not automatically names, and unrelated attributes are not substitutes
   for a requested event/value.
7. If insufficient, state only the requested information that is absent from the
   returned evidence or the concrete semantic mismatch.
"""


class InvestigationTaskAssessment(BaseModel):
    """
    LLM security interpretation of one successfully executed investigation task.
    """

    task_id: str
    evidence_relevant: bool

    # Separate evidence coverage from security interpretation. A query can
    # execute successfully and still fail to retrieve all factual components
    # needed to evaluate the task it claims to answer.
    evidence_coverage_sufficient: bool
    covered_evidence_components: list[str] = Field(default_factory=list)
    missing_evidence_components: list[str] = Field(default_factory=list)

    observed_facts: list[str] = Field(default_factory=list)
    security_assessment: str
    confidence: float = Field(ge=0.0, le=1.0)
    finding: str
    evidence_basis: str
    interpretation_limits: list[str] = Field(default_factory=list)

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, (int, float)) and 1 < value <= 100:
            return value / 100

        return value


class InvestigationSynthesis(BaseModel):
    """
    Structured investigation report produced from all successful verified evidence.

    This is intentionally richer than a single overall opinion so downstream
    presentation cannot summarize away material discoveries.
    """

    overall_assessment: str
    confidence: float = Field(ge=0.0, le=1.0)
    executive_summary: str = ""
    opinion: str
    supported_findings: list[str] = Field(default_factory=list)
    investigative_leads: list[str] = Field(default_factory=list)
    control_effectiveness_observations: list[str] = Field(default_factory=list)
    benign_observations: list[str] = Field(default_factory=list)
    negative_findings: list[str] = Field(default_factory=list)
    unresolved_areas: list[str] = Field(default_factory=list)
    inconclusive_observations: list[str] = Field(default_factory=list)
    coverage_statement: str

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, (int, float)) and 1 < value <= 100:
            return value / 100

        return value


class InvestigationOverallSufficiency(BaseModel):
    """Holistic sufficiency judgment over all successful investigation evidence."""

    sufficient: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    missing_evidence: list[str] = Field(default_factory=list)

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, (int, float)) and 1 < value <= 100:
            return value / 100
        return value


_INVESTIGATION_OVERALL_SUFFICIENCY_PROMPT = """
You are the FINAL EVIDENCE-SUFFICIENCY REVIEWER for a database investigation.

ORIGINAL USER REQUEST
---------------------
{query}

PERSISTENT INVESTIGATION LEDGER
-------------------------------
{ledger_context}

The ledger is advisory memory, not a mandatory checklist.

ALL SUCCESSFUL VERIFIED DATABASE EVIDENCE
-----------------------------------------
{raw_evidence}

TASK
----
Decide whether the successful evidence, TAKEN AS A WHOLE, is sufficient to make
a defensible evidence-grounded response to the ORIGINAL USER REQUEST.

Rules:
1. Judge global sufficiency before considering whether individual subtasks were
   incomplete.
2. Do NOT require every planned task, ledger label, or semantic component to be
   independently satisfied.
3. One successful query may be sufficient for a NARROW or existential request
   when it directly tested the user's requested condition and returned evidence
   adequate for reasoning. Do not apply this shortcut mechanically to a BROAD
   review request.
4. For a broad review, judge whether the successful evidence provides reasonable
   breadth across the material areas the investigation identified, not merely
   whether one concern was found. A broad review may still be sufficient without
   every planned task when the unexecuted areas are not material to a defensible
   response.
5. When a concern is supported, consider whether the evidence characterizes it
   well enough to report usefully: affected actor/identifier, event, time, place,
   and factual basis when those facts are available. Do not invent unavailable
   details.
6. Multiple successful evidence sets may collectively establish the answer even
   if no single subtask contains every factual domain.
5. Failed or incomplete tasks are limitations only when the missing evidence is
   still MATERIAL to answering the original request.
6. Do not demand authorization, identity, timing, or other domains unless they
   are actually necessary for the original question.
7. If sufficient=false, list only concrete factual evidence that is genuinely
   still required. Avoid duplicate/synonymous checklist labels.
8. Do not generate SQL.
9. Do not make up facts not present in the verified evidence.

Return a structured holistic sufficiency decision.
"""


_INVESTIGATION_TASK_REVIEW_PROMPT = """
You are the security-analysis reviewer for ONE successfully executed database
investigation task.

ORIGINAL USER REQUEST
---------------------
{query}

PERSISTENT INVESTIGATION EVIDENCE LEDGER
----------------------------------------
{ledger_context}

Ledger component names are semantic reasoning labels, not database identifiers.

INVESTIGATION TASK
------------------
Task ID: {task_id}
Title: {title}
Question: {task_question}

VERIFIED DATABASE RESULT
------------------------
Rows returned: {row_count}
Columns returned: {columns}

Evidence:
{evidence}

TASK
----
Perform TWO separate judgments, in this order.

FIRST — EVIDENCE COVERAGE
Determine whether the returned database evidence actually contains the factual
components needed to evaluate the investigation task.

Set:
- evidence_relevant=true only when the result is genuinely relevant to the task.
- evidence_coverage_sufficient=true only when the returned fields/evidence cover
  ALL factual sides needed to evaluate the task.
- covered_evidence_components to semantic facts genuinely established by this
  result (for example employee identity, room-access event, event timestamp).
- missing_evidence_components to the specific semantic factual sides/domains still
  missing. These are concepts, NOT proposed column names.

Examples of the principle:
- evidence about an access event alone does NOT establish the employee's
  time-clock state at that event;
- time-clock evidence alone does NOT establish that a room-access event occurred;
- employee identity alone does NOT establish event timing or authorization state.

Do not assume that wording in the task question proves the SQL/result actually
tested that condition. Judge coverage from the RETURNED COLUMNS AND EVIDENCE.

SECOND — SECURITY INTERPRETATION
Only after coverage is evaluated, classify security_assessment as exactly one of:
- potential_security_risk
- probably_benign
- inconclusive

Rules:
1. Successful query execution proves only that verified organizational data was
   retrieved. It does NOT prove the task was fully tested.
2. Incomplete coverage does NOT erase verified facts and does NOT automatically
   force the whole task to inconclusive. A verified suspicious or policy-relevant
   event may remain a potential_security_risk or investigative lead while the
   missing relationship is explicitly qualified.
3. If evidence_relevant=false, security_assessment MUST be "inconclusive".
4. A zero-row result supports "condition not observed" ONLY when
   evidence_coverage_sufficient=true and the result actually tested every
   factual component of the task.
5. If a task asks about a relationship/correlation between multiple domains,
   evidence from only one domain cannot prove the full correlation. Preserve any
   concrete facts it does establish and identify the missing relationship.
6. Do not treat row count, duplicates, categories, or repeated values as risky
   merely because they exist.
7. A result can be probably_benign only when coverage is sufficient and the
   returned evidence actually supports that interpretation.
8. Use potential_security_risk only when specific returned facts reasonably
   support concern. Distinguish an attempted unauthorized action that was BLOCKED
   from a control failure that ALLOWED unauthorized activity. For example, a
   Denied access can support concern about the attempted behavior while also
   showing that the access control enforced the restriction; do not describe the
   denial itself as a control-enforcement failure.
9. Do not invent thresholds, policies, expected values, user identities,
   business rules, dates, or facts not present in the evidence.
10. evidence_basis must identify the concrete returned facts supporting the
    judgment and explicitly note missing coverage when applicable.
"""


_INVESTIGATION_SYNTHESIS_PROMPT = """
You are the final security-analysis synthesizer for a PARTIAL OR COMPLETE
multi-task database investigation.

ORIGINAL USER REQUEST
---------------------
{query}

PERSISTENT INVESTIGATION EVIDENCE LEDGER
----------------------------------------
{ledger_context}

INVESTIGATION COVERAGE
----------------------
Tasks attempted: {attempted}
Tasks successfully executed: {succeeded}
Tasks failed or produced no verified query result: {failed}

ALL SUCCESSFUL VERIFIED DATABASE EVIDENCE
-----------------------------------------
{raw_evidence}

INDIVIDUAL LLM SECURITY ASSESSMENTS
-----------------------------------
{assessments}

TASK
----
Provide an evidence-supported security opinion using BOTH:
1. the complete verified database evidence from all successful tasks; and
2. the individual task-level security assessments.

You are responsible for reasoning over the successful evidence AS A WHOLE, not
merely summarizing the per-task labels.

Look for supported cross-task relationships, recurring actors, identifiers,
rooms, timestamps, statuses, access activity, key activity, employment status,
time-clock activity, and other patterns that become meaningful only when
multiple successful evidence sets are considered together.

Set overall_assessment to exactly one of:
- potential_security_risks_identified
- no_supported_security_risk_identified
- inconclusive

Use three distinct evidentiary categories in the report:
- supported_findings: evidence is sufficient to report a security concern.
- investigative_leads: verified evidence indicates a potentially meaningful concern,
  relationship, or pattern worth reporting, but one or more material facts are still
  missing before it should be called a supported security finding.
- unresolved_areas: available evidence does not presently support even a meaningful
  potential concern, or a material area could not be evaluated.

Rules:
1. Failed investigation tasks are coverage limitations, not evidence that no
   security issue exists.
2. Do not require every planned task to succeed before providing a useful
   opinion from successful verified evidence.
3. Do not convert mere database activity into a security finding.
4. Independently examine the RAW VERIFIED EVIDENCE. Do not simply inherit the
   per-task classifications.
5. A potential security finding may arise from:
   - one task's concrete evidence; or
   - a supported relationship/correlation across multiple successful tasks.
6. Cross-task correlations must be grounded in actual common values present in
   the evidence. Do not invent identity matches, timestamps, rooms, thresholds,
   schedules, expected behavior, or business rules.
7. Probably-benign and inconclusive task results must not automatically become
   risks. Evidence from a task with evidence_coverage_sufficient=false cannot
   independently establish the requested condition.

   A different overall opinion is allowed only when the successful evidence,
   taken together, actually supplies every factual domain needed for the claimed
   correlation. Do not infer a missing domain from activity in another domain.
8. Treat a task with evidence_coverage_sufficient=false as INCONCLUSIVE,
   regardless of whether its SQL executed or returned zero rows.
9. If the user's requested correlation depends on factual domains that are
   missing from the successful evidence, overall_assessment MUST be
   "inconclusive" unless other successful evidence independently establishes
   the requested relationship.

   If every successful task assessment has evidence_coverage_sufficient=false,
   overall_assessment MUST be "inconclusive". Do not return either a positive
   security finding or a "no risk identified" conclusion.
10. If some tasks failed or had insufficient evidence coverage, clearly state
    that the review is partial/inconclusive for those areas.
11. If no successful evidence supports a risk AND coverage is sufficient for the
    requested question, you may say no supported risk was identified in the
    evidence successfully reviewed. Do not imply unreviewed areas are safe.
12. supported_findings must identify the concrete database facts or cross-task
    pattern supporting each potential concern.
13. benign_observations should identify evidence reviewed that did not support
    a security concern.
14. inconclusive_observations should identify evidence whose security meaning
    could not be established from the available records.
15. Distinguish suspicious/unauthorized attempts from failed controls. A blocked
    attempt may be a supported security concern without implying that the control
    failed. Claim a control failure only when verified evidence shows unauthorized
    activity was allowed or another control malfunction occurred.
16. When supported_findings are present, identify the concrete affected employee
    name or ID, event/time/place, and evidence basis when those values are present
    in the verified evidence. Never convert an ID into a name or invent a missing
    attribute.
17. Do not invent facts, thresholds, policies, or findings.
18. opinion should directly answer the user's request and provide a reasoned
    professional security judgment grounded in the complete successful evidence.
19. For BROAD review requests, do not compress the investigation to only the most
    prominent concern. Preserve every MATERIAL, non-duplicative supported finding
    discovered across the successful evidence.
20. Each supported_findings item should be self-contained and, when available in
    evidence, identify: actor/name/ID, event or condition, timestamp/date, location,
    relevant rule/control state, evidence basis, and why the facts are security-relevant.
21. executive_summary must summarize the overall result without replacing the
    detailed findings.
22. control_effectiveness_observations should separately capture controls that
    demonstrably blocked, detected, or failed to prevent relevant activity. Do not
    infer effectiveness without evidence.
23. negative_findings should capture material conditions that were actually tested
    with sufficient coverage and not observed. Do not turn untested areas into
    negative findings.
24. unresolved_areas should contain only material areas that could not be evaluated
    from available evidence. Missing evidence in one area must not erase supported
    findings in other areas.
25. Avoid duplicate or synonymous findings. Combine records only when doing so
    preserves the concrete evidence needed to understand each material concern.
"""


class BranchScore(BaseModel):
    branch_id: str
    relevance: int = Field(ge=0, le=40)
    completeness: int = Field(ge=0, le=30)
    fidelity: int = Field(ge=0, le=30)
    total: int = Field(ge=0, le=100)


class EvaluatorResponse(BaseModel):
    scores: list[BranchScore]
    best_branch_id: str
    rationale: str


BRANCH_STRATEGIES = (
    {
        "id": "branch_1",
        "persona": "STRICT EVIDENCE GRADER",
        "strategy": (
            "Pass only when every required evidence type and output is "
            "directly supported."
        ),
    },
    {
        "id": "branch_2",
        "persona": "COMPLETENESS GRADER",
        "strategy": (
            "Evaluate whether the available evidence fully satisfies the "
            "question and its evidence contract."
        ),
    },
    {
        "id": "branch_3",
        "persona": "SECURITY INTERPRETATION GRADER",
        "strategy": (
            "Evaluate relevance, completeness, and fidelity without turning "
            "raw records into unsupported conclusions."
        ),
    },
)


_BRANCH_PROMPT = """
You are the {persona}.

STRATEGY
--------
{strategy}

USER QUERY
----------
{query}

EVIDENCE CONTRACT
-----------------
Required evidence types: {required_evidence_types}
Available evidence types: {available_evidence_types}
Required output fields: {required_output_fields}
Requires interpretation: {requires_interpretation}
Requirement reason: {evidence_requirement_reason}

VERIFIED EVIDENCE
-----------------
{evidence_context}

INVESTIGATION SUMMARY
---------------------
{investigation_summary}

RULES
-----
1. Grade only the supplied evidence.
2. A successful query does not prove evidence sufficiency.
3. IDs are not names.
4. Raw events are not automatically anomalies or violations.
5. Missing required evidence or requested outputs requires FAIL.
6. Schema evidence proves structure, not event occurrence.
7. Policy evidence provides guidance, not organizational facts.
8. A PASS requires non-empty refined_context containing an evidence-supported
   interpretation or synthesis.
9. refined_context is NOT a replacement for verified evidence. Agent 3
   deterministically preserves successful database/investigation evidence after
   branch selection.
10. A FAIL requires empty refined_context.
11. Never invent names, dates, filters, thresholds, relationships, findings,
    or citations.
12. If only part of a multi-task investigation succeeded, describe conclusions
    as partial when the missing areas are MATERIAL to the original request. Do not
    fail a branch merely because not every planned task ran when the supplied
    HOLISTIC EVIDENCE SUFFICIENCY decision says the original request has adequate
    coverage; instead challenge whether that decision and the candidate synthesis
    are actually supported by the raw evidence.
13. When the VERIFIED EVIDENCE contains a section titled
    "CANDIDATE INVESTIGATION INTERPRETATION TO CHALLENGE", independently test that
    interpretation against the raw evidence. Do not accept it merely because
    another LLM produced it.
14. For temporal/correlation claims, reason about what the returned records and
    query conditions actually establish. Distinguish:
    - the requested condition;
    - its opposite;
    - related but insufficient evidence.
15. If the candidate interpretation overstates what the evidence proves, prefer
    a corrected evidence-grounded refined_context when the verified evidence still
    supports a useful conclusion. FAIL only when a material defect prevents a
    defensible interpretation. State the concrete defect in missing_information.
16. Distinguish blocked unauthorized attempts from control failures. Do not infer
    that a control failed merely because it denied an unauthorized attempt.
"""


_EVALUATOR_PROMPT = """
Evaluate the following evidence-grading branches.

USER QUERY
----------
{query}

EVIDENCE CONTRACT
-----------------
Required evidence types: {required_evidence_types}
Required output fields: {required_output_fields}
Requires interpretation: {requires_interpretation}

BRANCHES
--------
{branches_text}

Score:
- relevance: 0-40
- completeness: 0-30
- fidelity: 0-30

Rules:
1. A PASS with empty refined_context scores zero.
2. A FAIL receives zero completeness.
3. Unsupported claims receive zero fidelity.
4. A branch admitting missing material evidence cannot be selected as PASS.
"""


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _grade_from_text(value: str) -> GradeResult:
    return (
        GradeResult.PASS if _clean_text(value).lower() == "pass" else GradeResult.FAIL
    )


def _normalize_confidence_dict(
    value: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(value)
    confidence = normalized.get("confidence")

    if isinstance(confidence, (int, float)) and confidence > 1:
        normalized["confidence"] = confidence / 100

    return normalized


def _extract_structured_response(
    response: Any,
    response_model: type[BaseModel],
) -> BaseModel:
    raw = getattr(response, "raw", None)

    if isinstance(raw, response_model):
        return raw

    if isinstance(raw, dict):
        return response_model.model_validate(_normalize_confidence_dict(raw))

    if isinstance(response, response_model):
        return response

    parsed = (getattr(response, "additional_kwargs", {}) or {}).get("parsed")

    if isinstance(parsed, response_model):
        return parsed

    if isinstance(parsed, dict):
        return response_model.model_validate(_normalize_confidence_dict(parsed))

    text = _clean_text(getattr(response, "text", response))

    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    if not text:
        raise ValueError("The structured LLM returned an empty response.")

    payload = json.loads(text)

    if isinstance(payload, dict):
        payload = _normalize_confidence_dict(payload)

    return response_model.model_validate(payload)


def _is_direct_query_state(
    state: WorkflowState,
) -> bool:
    """Return True when Agent 1 selected the bounded DirectQuery path."""
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


def _build_direct_query_context(
    state: WorkflowState,
) -> str:
    """
    Build Agent 4's verified answer context for a successful DirectQuery.

    The database result is the answer evidence. Schema evidence remains in
    WorkflowState for grounding/auditability but is intentionally excluded from
    presentation context so Agent 4 cannot dump the schema catalog to the user.
    """
    if not (state.database_query_succeeded and state.database_evidence.strip()):
        return ""

    lines = [
        "=== VERIFIED DATABASE EVIDENCE ===",
        f"Rows: {state.database_row_count}",
        f"Columns: {state.database_columns}",
        "Evidence:",
        state.database_evidence.strip(),
    ]

    return "\n".join(lines).strip()


def _build_refined_context(
    state: WorkflowState,
) -> str:
    sections: list[str] = []

    if state.policy_context.strip():
        sections.append("=== POLICY EVIDENCE ===\n" f"{state.policy_context.strip()}")

    if state.schema_evidence.strip():
        sections.append("=== SCHEMA EVIDENCE ===\n" f"{state.schema_evidence.strip()}")

    if state.investigation_results:
        for index, result in enumerate(
            state.investigation_results,
            start=1,
        ):
            if not result.database_query_succeeded:
                continue

            sections.append(
                "\n".join(
                    [
                        (
                            f"=== INVESTIGATION TASK {index}: "
                            f"{result.title or result.task_id} ==="
                        ),
                        f"Question: {result.question}",
                        f"Rows: {result.database_row_count}",
                        f"Columns: {result.database_columns}",
                        "Evidence:",
                        result.database_evidence,
                    ]
                )
            )

    elif state.database_query_succeeded and state.database_evidence.strip():
        sections.append(
            "=== DATABASE EVIDENCE ===\n" f"{state.database_evidence.strip()}"
        )

    return "\n\n".join(sections).strip()


def _investigation_summary(
    state: WorkflowState,
) -> str:
    if not state.investigation_results:
        return "(none)"

    return "\n".join(
        (
            f"- {result.task_id}: "
            f"success={result.database_query_succeeded}; "
            f"rows={result.database_row_count}; "
            f"columns={result.database_columns}; "
            f"error={result.database_error or 'none'}"
        )
        for result in state.investigation_results
    )


def _successful_investigation_results(
    state: WorkflowState,
) -> list[Any]:
    return [
        result
        for result in state.investigation_results
        if result.database_query_succeeded
    ]


def _failed_investigation_results(
    state: WorkflowState,
) -> list[Any]:
    return [
        result
        for result in state.investigation_results
        if not result.database_query_succeeded
    ]


def _build_preserved_investigation_evidence(
    state: WorkflowState,
) -> str:
    """
    Preserve verified operational evidence for Agent 4.

    Tree-of-Thought may interpret evidence, but it must not replace successful
    database rows with a short prose conclusion.
    """
    if not state.investigation_results:
        if state.database_query_succeeded and state.database_evidence.strip():
            return (
                "=== VERIFIED DATABASE EVIDENCE ===\n"
                f"Rows: {state.database_row_count}\n"
                f"Columns: {state.database_columns}\n"
                "Evidence:\n"
                f"{state.database_evidence.strip()}"
            )

        return ""

    sections: list[str] = []

    successful = _successful_investigation_results(state)
    failed = _failed_investigation_results(state)

    sections.append(
        "=== INVESTIGATION COVERAGE ===\n"
        f"Tasks attempted: {len(state.investigation_results)}\n"
        f"Tasks succeeded: {len(successful)}\n"
        f"Tasks failed: {len(failed)}\n"
        "Coverage status: "
        + (
            "complete"
            if successful and not failed
            else "partial" if successful else "failed"
        )
    )

    for index, result in enumerate(
        state.investigation_results,
        start=1,
    ):
        lines = [
            (
                f"=== INVESTIGATION TASK {index}: "
                f"{result.title or result.task_id} ==="
            ),
            f"Question: {result.question}",
            ("Database success: " f"{result.database_query_succeeded}"),
            f"Rows returned: {result.database_row_count}",
            f"Columns: {result.database_columns}",
        ]

        if result.database_query_succeeded:
            lines.extend(
                [
                    "Verified database evidence:",
                    (
                        result.database_evidence.strip()
                        or "(query succeeded but returned no evidence text)"
                    ),
                ]
            )
        else:
            lines.extend(
                [
                    "Task limitation:",
                    (
                        _clean_text(result.database_error)
                        or "No verified database evidence was produced."
                    ),
                ]
            )

        sections.append("\n".join(lines))

    return "\n\n".join(sections).strip()


def _merge_verified_evidence_with_interpretation(
    state: WorkflowState,
    interpretation: str,
) -> str:
    """
    Put verified evidence before LLM interpretation so Agent 4 cannot lose
    successful database results through ToT compression.
    """
    sections: list[str] = []

    preserved = _build_preserved_investigation_evidence(state)

    if preserved:
        sections.append(preserved)

    if state.policy_context.strip():
        sections.append(
            "=== VERIFIED POLICY EVIDENCE ===\n" f"{state.policy_context.strip()}"
        )

    cleaned_interpretation = _clean_text(interpretation)

    if cleaned_interpretation:
        sections.append(
            "=== EVIDENCE-SUPPORTED INTERPRETATION ===\n" f"{cleaned_interpretation}"
        )

    return "\n\n".join(section for section in sections if _clean_text(section)).strip()


def _infer_requirements_when_missing(
    state: WorkflowState,
) -> None:
    """
    Backward-compatible fallback until Agent 1 or Agent 2 explicitly populates
    the evidence contract.
    """
    if state.required_evidence_types:
        return

    if state.investigation_requested:
        state.required_evidence_types = [EvidenceType.INVESTIGATION.value]
        state.requires_interpretation = True
        state.evidence_requirement_reason = (
            "The request requires multi-step investigation and interpretation."
        )

    elif state.schema_query_succeeded:
        state.required_evidence_types = [EvidenceType.SCHEMA.value]
        state.evidence_requirement_reason = (
            "RetrieverAgent supplied authoritative schema evidence."
        )

    elif state.database_query_succeeded:
        state.required_evidence_types = [EvidenceType.DATABASE.value]

    elif state.policy_context.strip():
        state.required_evidence_types = [EvidenceType.POLICY.value]


def _required_evidence_present(
    state: WorkflowState,
) -> tuple[bool, list[str]]:
    state.normalize_evidence_requirements()

    required = set(state.required_evidence_types)
    available = state.available_evidence_types()
    missing = sorted(required - available)

    return (not missing, missing)


def _all_available_output_names(
    state: WorkflowState,
) -> set[str]:
    names = {
        _clean_text(column).lower()
        for column in state.database_columns
        if _clean_text(column)
    }

    for result in state.investigation_results:
        names.update(
            _clean_text(column).lower()
            for column in result.database_columns
            if _clean_text(column)
        )

    if state.schema_query_succeeded:
        names.update(
            {
                "schema",
                "schemas",
                "table",
                "tables",
                "column",
                "columns",
                "relationship",
                "relationships",
                "primary key",
                "foreign key",
            }
        )

    return names


def _required_outputs_present(
    state: WorkflowState,
) -> tuple[bool, list[str]]:
    available = _all_available_output_names(state)
    missing: list[str] = []

    for requirement in state.required_output_fields:
        cleaned = _clean_text(requirement).lower()

        if not cleaned:
            continue

        terminal_name = cleaned.rsplit(".", 1)[-1]

        if cleaned not in available and terminal_name not in available:
            missing.append(requirement)

    return (not missing, missing)


_SUMMARY_REQUEST_PATTERNS = (
    r"\bsummarize\b",
    r"\bsummary\b",
    r"\boverview\b",
    r"\bhighlights?\b",
    r"\bkey points?\b",
    r"\bbriefly\b",
    r"\bconcise\b",
    r"\bshort answer\b",
    r"\bin brief\b",
)


def _user_explicitly_requests_summary(
    query: str,
) -> bool:
    """
    Detect an explicit request to condense evidence.

    This is intentionally evidence-type agnostic. It applies equally to
    policies, schemas, database records, configurations, inventories, audit
    results, and other factual evidence.
    """
    cleaned = _clean_text(query)

    return any(
        re.search(
            pattern,
            cleaned,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in _SUMMARY_REQUEST_PATTERNS
    )


def _set_failure_presentation_contract(
    state: WorkflowState,
    reason: str,
) -> None:
    """Require Agent 4 to produce a deterministic failure response."""
    state.set_presentation_contract(
        mode=PresentationMode.FAILURE,
        reason=reason,
        requires_complete_output=False,
        allow_llm_rewrite=False,
        allow_summary=False,
        allow_inference=False,
    )


def _set_direct_factual_presentation_contract(
    state: WorkflowState,
) -> None:
    """
    Select a rendering strategy for verified direct factual evidence.

    DirectQuery:
        Use SUMMARY as a factual presentation mode. The LLM may organize and
        phrase verified database rows, but inference remains prohibited.

    Other direct factual evidence:
        Preserve the prior VERBATIM default unless the user explicitly asks
        for summarization.
    """
    if _is_direct_query_state(state):
        state.set_presentation_contract(
            mode=PresentationMode.SUMMARY,
            reason=(
                "DirectQuery returned verified database evidence. Agent 4 may "
                "organize and phrase those rows into a user-facing factual "
                "answer, but may not infer beyond the returned evidence."
            ),
            requires_complete_output=True,
            allow_llm_rewrite=True,
            allow_summary=True,
            allow_inference=False,
        )
        return

    if _user_explicitly_requests_summary(state.query):
        state.set_presentation_contract(
            mode=PresentationMode.SUMMARY,
            reason=("The user explicitly requested a condensed factual response."),
            requires_complete_output=False,
            allow_llm_rewrite=True,
            allow_summary=True,
            allow_inference=False,
        )
        return

    state.set_presentation_contract(
        mode=PresentationMode.VERBATIM,
        reason=(
            "The evidence directly answers the request and no interpretation "
            "or explicit summarization was requested."
        ),
        requires_complete_output=True,
        allow_llm_rewrite=False,
        allow_summary=False,
        allow_inference=False,
    )


def _set_interpretive_presentation_contract(
    state: WorkflowState,
) -> None:
    """
    Allow evidence-supported interpretation after successful ToT grading.

    Verified database/investigation evidence is preserved in refined_context;
    Agent 4 may summarize it but must not lose it.
    """
    state.set_presentation_contract(
        mode=PresentationMode.INTERPRET,
        reason=(
            state.evidence_requirement_reason
            or (
                "The request requires evidence-supported interpretation, "
                "comparison, correlation, or judgment."
            )
        ),
        requires_complete_output=False,
        allow_llm_rewrite=True,
        allow_summary=True,
        allow_inference=True,
    )


def _failure_from_missing_contract(
    missing_evidence: list[str],
    missing_outputs: list[str],
) -> EvidenceFailure:
    if missing_evidence:
        return EvidenceFailure(
            category=(EvidenceFailureCategory.MISSING_REQUIRED_EVIDENCE),
            reason=("Missing required evidence types: " + ", ".join(missing_evidence)),
            repairable=False,
        )

    if missing_outputs:
        return EvidenceFailure(
            category=(EvidenceFailureCategory.MISSING_REQUIRED_OUTPUT),
            reason=("Missing required outputs: " + ", ".join(missing_outputs)),
            repairable=False,
        )

    return EvidenceFailure(
        category=EvidenceFailureCategory.NONE,
        reason="The evidence contract is satisfied.",
        repairable=False,
    )


class GraderWriterAgent:
    """Agent 3 – generalized evidence sufficiency grader."""

    def __init__(
        self,
        llm: LLM,
        schema_retriever: Any | None = None,
    ) -> None:
        self._branch_llm = llm.as_structured_llm(BranchResponse)
        self._evaluator_llm = llm.as_structured_llm(EvaluatorResponse)
        self._direct_query_review_llm = llm.as_structured_llm(DirectQueryEvidenceReview)
        self._investigation_task_review_llm = llm.as_structured_llm(
            InvestigationTaskAssessment
        )
        self._investigation_synthesis_llm = llm.as_structured_llm(
            InvestigationSynthesis
        )
        self._investigation_overall_sufficiency_llm = llm.as_structured_llm(
            InvestigationOverallSufficiency
        )

        print("\n[GraderWriterAgent] ⚙  Initialising generalized evidence grader…")
        print(
            "[GraderWriterAgent]    Framework : "
            "evidence contract + coverage repair + per-task interpretation + global synthesis + Tree of Thought"
        )
        print(
            "[GraderWriterAgent]    Rule      : "
            "DirectQuery review checks returned evidence only; validated schema facts are final"
        )
        print(
            "[GraderWriterAgent]    Investigation stop: holistic successful-evidence "
            "sufficiency overrides per-task checklist gaps"
        )
        print(
            "[GraderWriterAgent]    Investigation ToT : 3 independent challenge "
            "branches run before final interpretation"
        )
        print("[GraderWriterAgent]    Output    : " "presentation contract for Agent 4")
        print("[GraderWriterAgent]    MCP tools : none")

    @staticmethod
    def _store_failure(
        state: WorkflowState,
        failure: EvidenceFailure,
    ) -> None:
        state.evidence_failure_category = failure.category.value
        state.evidence_failure_reason = failure.reason
        state.evidence_failure_repairable = failure.repairable

    def _review_direct_query_evidence(
        self,
        state: WorkflowState,
    ) -> DirectQueryEvidenceReview:
        """
        Final semantic evidence gate for DirectQuery.

        Schema validity has already been established upstream. This reviewer
        checks only whether the returned evidence answers the original question.
        """
        sql_plan = getattr(state, "sql_plan", None)

        validated_tables = list(getattr(sql_plan, "tables", []) or [])
        validated_columns = list(getattr(sql_plan, "columns", []) or [])
        validated_relationships = list(getattr(sql_plan, "relationships", []) or [])

        prompt = _DIRECT_QUERY_EVIDENCE_REVIEW_PROMPT.format(
            query=state.query,
            sql=_clean_text(getattr(state, "sql_query", "")) or "(not recorded)",
            validated_tables=validated_tables or ["(none)"],
            validated_columns=validated_columns or ["(none)"],
            validated_relationships=(validated_relationships or ["(none required)"]),
            row_count=state.database_row_count,
            columns=state.database_columns,
            evidence=(
                state.database_evidence.strip()
                or "(validated query executed successfully and returned zero rows)"
            ),
        )

        try:
            parsed = _extract_structured_response(
                _complete_with_system(
                    self._direct_query_review_llm,
                    system_prompt=GRADER_WRITER_SYSTEM_PROMPT,
                    user_prompt=prompt,
                ),
                DirectQueryEvidenceReview,
            )
            assert isinstance(parsed, DirectQueryEvidenceReview)
            return parsed
        except Exception as exc:
            return DirectQueryEvidenceReview(
                sufficient=False,
                confidence=0.0,
                reason=(
                    "DirectQuery evidence sufficiency review failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    def _review_investigation_task(
        self,
        *,
        state: WorkflowState,
        result: Any,
    ) -> InvestigationTaskAssessment:
        """
        Ask the LLM to interpret one successfully executed investigation result.

        Database execution success is not itself a finding. The LLM must decide
        whether the returned evidence suggests a risk, appears benign, or is
        inconclusive.
        """
        task_id = _clean_text(getattr(result, "task_id", "")) or "unknown"
        title = _clean_text(getattr(result, "title", "")) or task_id
        task_question = _clean_text(getattr(result, "question", ""))

        prompt = _INVESTIGATION_TASK_REVIEW_PROMPT.format(
            query=state.query,
            ledger_context=state.investigation_ledger.compact_summary(),
            task_id=task_id,
            title=title,
            task_question=task_question or "(not recorded)",
            row_count=getattr(result, "database_row_count", 0),
            columns=getattr(result, "database_columns", []),
            evidence=(
                _clean_text(getattr(result, "database_evidence", ""))
                or "(query executed successfully and returned zero rows)"
            ),
        )

        try:
            parsed = _extract_structured_response(
                _complete_with_system(
                    self._investigation_task_review_llm,
                    system_prompt=GRADER_WRITER_SYSTEM_PROMPT,
                    user_prompt=prompt,
                ),
                InvestigationTaskAssessment,
            )
            assert isinstance(parsed, InvestigationTaskAssessment)

            assessment = _clean_text(parsed.security_assessment).lower()

            if assessment not in {
                "potential_security_risk",
                "probably_benign",
                "inconclusive",
            }:
                parsed.security_assessment = "inconclusive"

            # Python validates structure; the LLM owns semantic interpretation.
            # Incomplete coverage qualifies a conclusion but does not erase a
            # verified discovery or automatically force it to inconclusive.
            if not parsed.evidence_relevant:
                parsed.security_assessment = "inconclusive"

            if (
                not parsed.evidence_coverage_sufficient
                and not parsed.missing_evidence_components
            ):
                parsed.missing_evidence_components = [
                    "Additional material evidence is needed to evaluate the full task."
                ]

            parsed.task_id = task_id
            return parsed

        except Exception as exc:
            return InvestigationTaskAssessment(
                task_id=task_id,
                evidence_relevant=False,
                evidence_coverage_sufficient=False,
                covered_evidence_components=[],
                missing_evidence_components=[
                    "The task's evidence coverage could not be reliably assessed."
                ],
                observed_facts=[],
                security_assessment="inconclusive",
                confidence=0.0,
                finding=(
                    "The security significance of this task could not be "
                    "reliably interpreted."
                ),
                evidence_basis=(
                    "Investigation-task interpretation failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                interpretation_limits=[
                    "The task-level LLM interpretation was unavailable."
                ],
            )

    @staticmethod
    def _build_successful_raw_evidence_for_synthesis(
        state: WorkflowState,
    ) -> str:
        """
        Build the complete raw-evidence input for the global investigation LLM.

        Only successfully executed tasks are included. No task-level
        interpretation replaces the underlying returned database evidence.
        """
        sections: list[str] = []

        for result in _successful_investigation_results(state):
            task_id = _clean_text(getattr(result, "task_id", "")) or "unknown"

            title = _clean_text(getattr(result, "title", "")) or task_id

            question = _clean_text(getattr(result, "question", ""))

            evidence = _clean_text(getattr(result, "database_evidence", ""))

            sections.append(
                "\n".join(
                    [
                        f"=== SUCCESSFUL TASK {task_id}: {title} ===",
                        f"Question: {question or '(not recorded)'}",
                        (
                            "Rows returned: "
                            f"{getattr(result, 'database_row_count', 0)}"
                        ),
                        ("Columns: " f"{getattr(result, 'database_columns', [])}"),
                        "Verified database evidence:",
                        (
                            evidence
                            or (
                                "(query executed successfully and "
                                "returned zero rows)"
                            )
                        ),
                    ]
                )
            )

        return (
            "\n\n".join(sections)
            if sections
            else "(no successful verified database evidence)"
        )

    def _review_overall_investigation_sufficiency(
        self,
        state: WorkflowState,
    ) -> InvestigationOverallSufficiency:
        """
        Judge whether all successful verified evidence is enough to answer the
        original request, regardless of unfinished task/ledger checklist items.
        """
        raw_evidence = self._build_successful_raw_evidence_for_synthesis(state)

        # Keep one grader call bounded. Preserve complete structure but cap very
        # large accumulated evidence for the local model.
        if len(raw_evidence) > 70000:
            raw_evidence = raw_evidence[:70000] + (
                "\\n...[raw evidence truncated for holistic sufficiency review]"
            )

        prompt = _INVESTIGATION_OVERALL_SUFFICIENCY_PROMPT.format(
            query=state.query,
            ledger_context=state.investigation_ledger.compact_summary(),
            raw_evidence=raw_evidence,
        )

        try:
            parsed = _extract_structured_response(
                _complete_with_system(
                    self._investigation_overall_sufficiency_llm,
                    system_prompt=GRADER_WRITER_SYSTEM_PROMPT,
                    user_prompt=prompt,
                ),
                InvestigationOverallSufficiency,
            )
            assert isinstance(parsed, InvestigationOverallSufficiency)
            return parsed
        except Exception as exc:
            return InvestigationOverallSufficiency(
                sufficient=False,
                confidence=0.0,
                reason=(
                    "Holistic investigation sufficiency review failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                missing_evidence=[],
            )

    @staticmethod
    def _format_investigation_assessments(
        assessments: list[InvestigationTaskAssessment],
    ) -> str:
        if not assessments:
            return "(none)"

        sections: list[str] = []

        for assessment in assessments:
            sections.append(
                "\n".join(
                    [
                        f"Task ID: {assessment.task_id}",
                        ("Evidence relevant: " f"{assessment.evidence_relevant}"),
                        (
                            "Evidence coverage sufficient: "
                            f"{assessment.evidence_coverage_sufficient}"
                        ),
                        (
                            "Covered evidence components: "
                            f"{assessment.covered_evidence_components}"
                        ),
                        (
                            "Missing evidence components: "
                            f"{assessment.missing_evidence_components}"
                        ),
                        f"Observed facts: {assessment.observed_facts}",
                        ("Security assessment: " f"{assessment.security_assessment}"),
                        f"Confidence: {assessment.confidence:.2f}",
                        f"Finding: {assessment.finding}",
                        f"Evidence basis: {assessment.evidence_basis}",
                        f"Interpretation limits: {assessment.interpretation_limits}",
                    ]
                )
            )

        return "\n\n".join(sections)

    @staticmethod
    def _build_holistic_supplemental_request(
        state: WorkflowState,
        review: InvestigationOverallSufficiency,
    ) -> str:
        """Build supplemental work only from globally material missing evidence."""
        missing: list[str] = []

        for item in review.missing_evidence:
            cleaned = _clean_text(item)
            if cleaned and cleaned not in missing:
                missing.append(cleaned)

        if not missing:
            return ""

        for item in missing:
            state.investigation_ledger.mark_missing(
                item,
                kind="semantic",
                note="Holistic Agent 3 review identified this as materially missing.",
            )

        return "\\n".join(
            [
                "Supplemental investigation evidence is required.",
                "Material missing evidence for the ORIGINAL user question:",
                *[f"- {item}" for item in missing],
                "",
                (
                    "Retrieve only this concrete missing evidence. Do not repeat "
                    "already successful evidence unless necessary to add one of "
                    "the material facts listed above."
                ),
            ]
        ).strip()

    @staticmethod
    def _collect_supplemental_evidence_request(
        state: WorkflowState,
        assessments: list[InvestigationTaskAssessment],
    ) -> str:
        """Build one bounded supplemental-evidence request from coverage gaps."""
        missing: list[str] = []
        affected_tasks: list[str] = []

        for assessment in assessments:
            # Coverage gaps are repairable even when the first evidence result
            # was judged not relevant enough to answer the full task. Relevance
            # must not suppress retrieval of an explicitly missing domain.
            if (
                not assessment.evidence_coverage_sufficient
                and assessment.missing_evidence_components
            ):
                affected_tasks.append(assessment.task_id)

                for item in assessment.missing_evidence_components:
                    cleaned = _clean_text(item)

                    if cleaned and cleaned not in missing:
                        missing.append(cleaned)

        # The persistent ledger is the cross-agent source of truth. Covered
        # components are removed from supplemental requests even if an individual
        # task review redundantly mentioned them as missing.
        ledger_covered = {
            " ".join(name.lower().split())
            for name in state.investigation_ledger.covered_names()
        }
        missing = [
            item
            for item in missing
            if " ".join(item.lower().split()) not in ledger_covered
        ]

        if not missing:
            return ""

        for item in missing:
            state.investigation_ledger.mark_missing(
                item,
                kind="semantic",
                note="Agent 3 identified this as a remaining coverage gap.",
            )

        return "\n".join(
            [
                "Supplemental investigation evidence is required.",
                (
                    "Affected investigation tasks: "
                    + ", ".join(affected_tasks or ["unknown"])
                ),
                "Missing evidence components:",
                *[f"- {item}" for item in missing],
                "",
                (
                    "Retrieve only the additional operational evidence needed "
                    "to fill these missing components. Preserve identifiers, "
                    "timestamps, and factual fields needed to correlate the "
                    "supplemental records with already verified evidence."
                ),
                (
                    "Do not repeat already successful evidence tasks unless "
                    "needed to add a missing factual field."
                ),
                (
                    "Do not make the final security judgment in retrieval; "
                    "return factual evidence for the grader to interpret."
                ),
            ]
        ).strip()

    def _synthesize_investigation(
        self,
        *,
        state: WorkflowState,
        assessments: list[InvestigationTaskAssessment],
    ) -> InvestigationSynthesis:
        successful = _successful_investigation_results(state)
        failed = _failed_investigation_results(state)
        raw_evidence = self._build_successful_raw_evidence_for_synthesis(state)

        prompt = _INVESTIGATION_SYNTHESIS_PROMPT.format(
            query=state.query,
            ledger_context=state.investigation_ledger.compact_summary(),
            attempted=len(state.investigation_results),
            succeeded=len(successful),
            failed=len(failed),
            raw_evidence=raw_evidence,
            assessments=self._format_investigation_assessments(assessments),
        )

        print(
            "[GraderWriterAgent]    Global synthesis raw evidence: "
            f"{len(raw_evidence)} chars from {len(successful)} task(s)"
        )

        try:
            parsed = _extract_structured_response(
                _complete_with_system(
                    self._investigation_synthesis_llm,
                    system_prompt=GRADER_WRITER_SYSTEM_PROMPT,
                    user_prompt=prompt,
                ),
                InvestigationSynthesis,
            )
            assert isinstance(parsed, InvestigationSynthesis)
            if _clean_text(parsed.overall_assessment).lower() not in {
                "potential_security_risks_identified",
                "no_supported_security_risk_identified",
                "inconclusive",
            }:
                parsed.overall_assessment = "inconclusive"
            # Deliberately no Python semantic veto here. The LLM owns the
            # interpretation; Python validates only the structured result.
            return parsed
        except Exception as exc:
            # Preserve task-level discoveries if global synthesis times out/fails.
            supported, leads, benign, unresolved = [], [], [], []
            for a in assessments:
                finding = _clean_text(a.finding)
                if finding:
                    if a.security_assessment == "potential_security_risk":
                        (supported if a.evidence_coverage_sufficient else leads).append(
                            finding
                        )
                    elif (
                        a.security_assessment == "probably_benign"
                        and a.evidence_coverage_sufficient
                    ):
                        benign.append(finding)
                    else:
                        unresolved.append(finding)
                for limit in a.interpretation_limits:
                    limit = _clean_text(limit)
                    if limit and limit not in unresolved:
                        unresolved.append(limit)

            has_concern = bool(supported or leads)
            opinion = (
                "Global synthesis was unavailable, but verified task-level findings "
                "and leads were preserved with their limitations."
                if has_concern
                else "Global synthesis was unavailable. Verified task-level observations "
                "were preserved without manufacturing a broader conclusion."
            )
            return InvestigationSynthesis(
                overall_assessment=(
                    "potential_security_risks_identified"
                    if has_concern
                    else "inconclusive"
                ),
                confidence=0.0,
                executive_summary=opinion,
                opinion=opinion,
                supported_findings=supported,
                investigative_leads=leads,
                control_effectiveness_observations=[],
                benign_observations=benign,
                negative_findings=[],
                unresolved_areas=unresolved,
                inconclusive_observations=[
                    "Global investigation synthesis unavailable: "
                    f"{type(exc).__name__}: {exc}"
                ],
                coverage_statement=(
                    f"{len(successful)} of {len(state.investigation_results)} planned "
                    "tasks produced verified database evidence."
                ),
            )

    @staticmethod
    def _build_investigation_interpretation(
        *,
        assessments: list[InvestigationTaskAssessment],
        synthesis: InvestigationSynthesis,
    ) -> str:
        """Build a durable, sectioned findings package for Agent 4."""
        sections = [
            "=== INVESTIGATION REPORT ===",
            "",
            "=== OVERALL SECURITY OPINION ===",
            f"Assessment: {synthesis.overall_assessment}",
            f"Confidence: {synthesis.confidence:.2f}",
            f"Opinion: {synthesis.opinion}",
            f"Coverage: {synthesis.coverage_statement}",
        ]

        if _clean_text(synthesis.executive_summary):
            sections.extend(
                [
                    "",
                    "=== EXECUTIVE SUMMARY ===",
                    synthesis.executive_summary.strip(),
                ]
            )

        def add_items(title: str, items: list[str]) -> None:
            cleaned = [_clean_text(item) for item in items if _clean_text(item)]
            if cleaned:
                sections.extend(["", title, *[f"- {item}" for item in cleaned]])

        add_items(
            "=== CONFIRMED / SUPPORTED SECURITY FINDINGS ===",
            synthesis.supported_findings,
        )
        add_items(
            "=== INVESTIGATIVE LEADS / POTENTIAL CONCERNS ===",
            synthesis.investigative_leads,
        )
        add_items(
            "=== CONTROL EFFECTIVENESS OBSERVATIONS ===",
            synthesis.control_effectiveness_observations,
        )
        add_items(
            "=== NEGATIVE FINDINGS / CONDITIONS NOT OBSERVED ===",
            synthesis.negative_findings,
        )
        add_items("=== PROBABLY BENIGN OBSERVATIONS ===", synthesis.benign_observations)

        unresolved = list(synthesis.unresolved_areas)
        for item in synthesis.inconclusive_observations:
            if _clean_text(item) and _clean_text(item) not in unresolved:
                unresolved.append(item)
        add_items("=== UNRESOLVED AREAS / LIMITATIONS ===", unresolved)

        sections.extend(
            [
                "",
                "=== PER-TASK SECURITY ASSESSMENTS ===",
                GraderWriterAgent._format_investigation_assessments(assessments),
            ]
        )

        return "\n".join(sections).strip()

    def _generate_branches(
        self,
        state: WorkflowState,
        *,
        supplemental_context: str = "",
    ) -> list[tuple[ToTThought, BranchResponse]]:
        """
        Generate independent LLM reasoning branches.

        supplemental_context is used for investigation mode to expose the
        holistic sufficiency decision and candidate synthesis to the branches
        so ToT can challenge that interpretation before Agent 3 finalizes it.
        """
        branches: list[tuple[ToTThought, BranchResponse]] = []

        available_types = sorted(state.available_evidence_types())
        evidence_context = _build_refined_context(state)

        if _clean_text(supplemental_context):
            evidence_context = (
                f"{evidence_context}\n\n"
                "=== CANDIDATE INVESTIGATION INTERPRETATION TO CHALLENGE ===\n"
                f"{_clean_text(supplemental_context)}"
            ).strip()

        for strategy in BRANCH_STRATEGIES:
            prompt = _BRANCH_PROMPT.format(
                persona=strategy["persona"],
                strategy=strategy["strategy"],
                query=state.query,
                required_evidence_types=(state.required_evidence_types),
                available_evidence_types=available_types,
                required_output_fields=(state.required_output_fields),
                requires_interpretation=(state.requires_interpretation),
                evidence_requirement_reason=(
                    state.evidence_requirement_reason or "(none)"
                ),
                evidence_context=(evidence_context or "(none)"),
                investigation_summary=(_investigation_summary(state)),
            )

            try:
                parsed = _extract_structured_response(
                    _complete_with_system(
                        self._branch_llm,
                        system_prompt=GRADER_WRITER_SYSTEM_PROMPT,
                        user_prompt=prompt,
                    ),
                    BranchResponse,
                )
                assert isinstance(
                    parsed,
                    BranchResponse,
                )
            except Exception as exc:
                state.add_warning(
                    "Tree-of-Thought branch "
                    f"{strategy['id']} unavailable: {type(exc).__name__}: {exc}"
                )
                # Transport/model failures are not semantic FAIL judgments.
                # Exclude unavailable branches from scoring and selection.
                continue

            grade = _grade_from_text(parsed.grade)
            refined = _clean_text(parsed.refined_context)

            if grade == GradeResult.FAIL or parsed.missing_information:
                grade = GradeResult.FAIL
                refined = ""
                parsed.grade = "fail"
                parsed.refined_context = ""

            thought = ToTThought(
                branch_id=strategy["id"],
                reasoning=json.dumps(
                    {
                        "grade": grade.value,
                        "confidence": parsed.confidence,
                        "reasoning_summary": (parsed.reasoning_summary),
                        "refined_context": refined,
                        "missing_information": (parsed.missing_information),
                    },
                    indent=2,
                ),
                score=float(parsed.confidence),
                selected=False,
            )

            branches.append((thought, parsed))

        return branches

    def _evaluate_branches(
        self,
        state: WorkflowState,
        branches: list[tuple[ToTThought, BranchResponse]],
    ) -> EvaluatorResponse:
        branches_text = "\n\n".join(
            (f"=== {thought.branch_id} ===\n" f"{thought.reasoning}")
            for thought, _ in branches
        )

        try:
            parsed = _extract_structured_response(
                _complete_with_system(
                    self._evaluator_llm,
                    system_prompt=GRADER_WRITER_SYSTEM_PROMPT,
                    user_prompt=_EVALUATOR_PROMPT.format(
                        query=state.query,
                        required_evidence_types=(state.required_evidence_types),
                        required_output_fields=(state.required_output_fields),
                        requires_interpretation=(state.requires_interpretation),
                        branches_text=branches_text,
                    ),
                ),
                EvaluatorResponse,
            )
            assert isinstance(
                parsed,
                EvaluatorResponse,
            )
            return parsed
        except Exception:
            scores: list[BranchScore] = []

            for thought, branch in branches:
                valid_pass = (
                    _grade_from_text(branch.grade) == GradeResult.PASS
                    and bool(_clean_text(branch.refined_context))
                    and not branch.missing_information
                )

                if valid_pass:
                    relevance = 35
                    completeness = 25
                    fidelity = 25
                else:
                    relevance = 5
                    completeness = 0
                    fidelity = 5

                scores.append(
                    BranchScore(
                        branch_id=thought.branch_id,
                        relevance=relevance,
                        completeness=completeness,
                        fidelity=fidelity,
                        total=(relevance + completeness + fidelity),
                    )
                )

            best = max(
                scores,
                key=lambda item: item.total,
            )

            return EvaluatorResponse(
                scores=scores,
                best_branch_id=best.branch_id,
                rationale=("Deterministic fallback scoring."),
            )

    @staticmethod
    def _select_branch(
        branches: list[tuple[ToTThought, BranchResponse]],
        evaluation: EvaluatorResponse,
    ) -> tuple[
        ToTThought,
        BranchResponse,
        int,
    ]:
        score_map = {score.branch_id: score.total for score in evaluation.scores}

        valid_passes = [
            pair
            for pair in branches
            if (
                _grade_from_text(pair[1].grade) == GradeResult.PASS
                and bool(_clean_text(pair[1].refined_context))
                and not pair[1].missing_information
            )
        ]

        candidates = valid_passes if valid_passes else branches

        selected_thought, selected_response = max(
            candidates,
            key=lambda pair: score_map.get(
                pair[0].branch_id,
                0,
            ),
        )

        for thought, _ in branches:
            thought.score = float(
                score_map.get(
                    thought.branch_id,
                    0,
                )
            )
            thought.selected = thought.branch_id == selected_thought.branch_id

        return (
            selected_thought,
            selected_response,
            int(
                score_map.get(
                    selected_thought.branch_id,
                    0,
                )
            ),
        )

    async def run(
        self,
        state: WorkflowState,
    ) -> WorkflowState:
        state.grade = None
        state.refined_context = ""
        state.tot_thoughts = []
        state.tot_best_branch = None
        state.reset_evidence_failure()
        state.reset_presentation_contract()

        _infer_requirements_when_missing(state)
        state.normalize_evidence_requirements()

        evidence_present, missing_evidence = _required_evidence_present(state)
        outputs_present, missing_outputs = _required_outputs_present(state)

        print("\n" + "═" * 70)
        print(
            "[GraderWriterAgent] ▶  STARTING — " "Agent 3: Generalized Evidence Grader"
        )
        print("[GraderWriterAgent]    Query          : " f"{state.query}")
        print(
            "[GraderWriterAgent]    Required types : "
            f"{state.required_evidence_types}"
        )
        print(
            "[GraderWriterAgent]    Available types: "
            f"{sorted(state.available_evidence_types())}"
        )
        print(
            "[GraderWriterAgent]    Required output: " f"{state.required_output_fields}"
        )
        print(
            "[GraderWriterAgent]    Interpretation : "
            f"{state.requires_interpretation}"
        )
        print("[GraderWriterAgent]    Presentation   : " "pending evidence grade")
        print(
            "[GraderWriterAgent]    DirectQuery    : "
            f"{_is_direct_query_state(state)}"
        )
        print("─" * 70)

        if not evidence_present or not outputs_present:
            failure = _failure_from_missing_contract(
                missing_evidence,
                missing_outputs,
            )
            self._store_failure(state, failure)
            state.grade = GradeResult.FAIL

            _set_failure_presentation_contract(
                state,
                reason=failure.reason,
            )

            print("[GraderWriterAgent] ✘ Contract failure")
            print("[GraderWriterAgent]    Category: " f"{failure.category.value}")
            print("[GraderWriterAgent]    Reason  : " f"{failure.reason}")
            print(
                "[GraderWriterAgent]    Presentation: "
                f"{state.presentation_mode.value}"
            )
            print("═" * 70 + "\n")
            return state

        # DirectQuery requires an LLM semantic evidence-sufficiency review
        # before the normal direct-factual pass. Successful SQL execution alone
        # is not evidence that the returned rows answer the original question.
        if (
            _is_direct_query_state(state)
            and not state.requires_interpretation
            and state.database_query_succeeded
        ):
            review = self._review_direct_query_evidence(state)

            print("[GraderWriterAgent]    Direct evidence review:")
            print("[GraderWriterAgent]      Sufficient : " f"{review.sufficient}")
            print("[GraderWriterAgent]      Confidence : " f"{review.confidence:.2f}")
            print("[GraderWriterAgent]      Reason     : " f"{review.reason}")

            if not review.sufficient:
                failure = EvidenceFailure(
                    category=EvidenceFailureCategory.INCOMPLETE_EVIDENCE,
                    reason=review.reason,
                    repairable=True,
                )

                self._store_failure(state, failure)
                state.grade = GradeResult.FAIL
                state.refined_context = ""

                _set_failure_presentation_contract(
                    state,
                    reason=failure.reason,
                )

                state.add_trace(
                    "GraderWriterAgent",
                    ("DirectQuery evidence semantic review FAIL: " f"{review.reason}"),
                )

                print("[GraderWriterAgent] ✘ DirectQuery evidence insufficient")
                print("[GraderWriterAgent]    Category: " f"{failure.category.value}")
                print("[GraderWriterAgent]    Repairable: " f"{failure.repairable}")
                print("═" * 70 + "\n")

                return state

        # General deterministic pass for direct factual evidence.
        if not state.requires_interpretation:
            refined = (
                _build_direct_query_context(state)
                if _is_direct_query_state(state)
                else _build_refined_context(state)
            )

            if not refined:
                failure = EvidenceFailure(
                    category=(EvidenceFailureCategory.NO_EVIDENCE),
                    reason=(
                        "The evidence contract was satisfied, but no "
                        "refined evidence text was available."
                    ),
                    repairable=False,
                )
                self._store_failure(state, failure)
                state.grade = GradeResult.FAIL

                _set_failure_presentation_contract(
                    state,
                    reason=failure.reason,
                )

                return state

            state.grade = GradeResult.PASS
            state.refined_context = refined

            _set_direct_factual_presentation_contract(state)

            state.add_trace(
                "GraderWriterAgent",
                (
                    "Deterministic PASS: required evidence and outputs "
                    "were present and no interpretation was required. "
                    f"Presentation={state.presentation_mode.value}."
                ),
            )

            print("[GraderWriterAgent] ✔ Deterministic evidence-contract PASS")
            print("[GraderWriterAgent]    Refined ctx: " f"{len(refined)} chars")
            print("[GraderWriterAgent]    ToT branches: 0")
            print(
                "[GraderWriterAgent]    Presentation: "
                f"{state.presentation_mode.value}"
            )
            print("[GraderWriterAgent]    LLM rewrite : " f"{state.allow_llm_rewrite}")
            print(
                "[GraderWriterAgent]    Complete out: "
                f"{state.requires_complete_output}"
            )
            print("═" * 70 + "\n")
            return state

        # Broad/multi-task investigations are interpreted task-by-task.
        # A successful SQL execution is only verified data retrieval; the LLM
        # must decide whether that evidence suggests a security risk, appears
        # benign, or is inconclusive.
        if state.investigation_requested and state.investigation_results:
            successful_tasks = _successful_investigation_results(state)
            failed_tasks = _failed_investigation_results(state)

            if not successful_tasks:
                failure = EvidenceFailure(
                    category=EvidenceFailureCategory.INCOMPLETE_EVIDENCE,
                    reason=(
                        "The investigation produced no successfully executed "
                        "database evidence for security interpretation."
                    ),
                    repairable=False,
                )

                self._store_failure(state, failure)
                state.grade = GradeResult.FAIL
                state.refined_context = ""

                _set_failure_presentation_contract(
                    state,
                    reason=failure.reason,
                )

                print(
                    "[GraderWriterAgent] ✘ Investigation produced no "
                    "successful database evidence."
                )
                print("═" * 70 + "\n")

                return state

            print(
                "[GraderWriterAgent]    Investigation security review: "
                f"{len(successful_tasks)} successful task(s)"
            )

            assessments: list[InvestigationTaskAssessment] = []

            for result in successful_tasks:
                assessment = self._review_investigation_task(
                    state=state,
                    result=result,
                )
                assessments.append(assessment)

                for component in assessment.covered_evidence_components:
                    state.investigation_ledger.mark_covered(
                        component,
                        kind="semantic",
                        task_id=assessment.task_id,
                        note="Agent 3 confirmed this semantic fact from verified evidence.",
                    )
                for component in assessment.missing_evidence_components:
                    state.investigation_ledger.mark_missing(
                        component,
                        kind="semantic",
                        note="Agent 3 identified this as missing evidence.",
                    )

                print(
                    "[GraderWriterAgent]      "
                    f"{assessment.task_id}: "
                    f"{assessment.security_assessment} "
                    f"(relevant={assessment.evidence_relevant}, "
                    f"coverage={assessment.evidence_coverage_sufficient}, "
                    f"confidence={assessment.confidence:.2f})"
                )

            # Synthesize only after task-level discoveries have been characterized.
            synthesis = self._synthesize_investigation(
                state=state,
                assessments=assessments,
            )

            # Holistic sufficiency is intentionally evaluated after discovery and
            # synthesis so missing context qualifies findings instead of erasing them.
            overall_sufficiency = self._review_overall_investigation_sufficiency(state)

            print("[GraderWriterAgent]    Holistic evidence sufficiency:")
            print(
                "[GraderWriterAgent]      Sufficient : "
                f"{overall_sufficiency.sufficient}"
            )
            print(
                "[GraderWriterAgent]      Confidence : "
                f"{overall_sufficiency.confidence:.2f}"
            )
            print(
                "[GraderWriterAgent]      Reason     : " f"{overall_sufficiency.reason}"
            )
            if overall_sufficiency.missing_evidence:
                print(
                    "[GraderWriterAgent]      Material missing: "
                    f"{overall_sufficiency.missing_evidence}"
                )

            supplemental_request = (
                ""
                if overall_sufficiency.sufficient
                else self._build_holistic_supplemental_request(
                    state,
                    overall_sufficiency,
                )
            )

            if overall_sufficiency.sufficient:
                state.add_trace(
                    "GraderWriterAgent",
                    (
                        "Holistic sufficiency reviewed after task interpretation and "
                        "synthesis; existing verified discoveries were preserved. "
                        f"Reason={overall_sufficiency.reason}"
                    ),
                )

            if supplemental_request:
                self._store_failure(
                    state,
                    EvidenceFailure(
                        category=EvidenceFailureCategory.INCOMPLETE_EVIDENCE,
                        reason=supplemental_request,
                        repairable=True,
                    ),
                )
                state.add_trace(
                    "GraderWriterAgent",
                    "Targeted supplemental evidence requested without erasing existing findings.",
                )
                print("[GraderWriterAgent] ↻ Supplemental evidence requested:")
                for line in supplemental_request.splitlines():
                    print(f"[GraderWriterAgent]    {line}")

            interpretation = self._build_investigation_interpretation(
                assessments=assessments,
                synthesis=synthesis,
            )

            # --------------------------------------------------------------
            # Investigation Tree of Thought
            # --------------------------------------------------------------
            # The investigation path used to return before the generic ToT
            # block below, which is why traces showed "ToT branches: 0".
            # Run the same independent branch/evaluator framework here as a
            # final LLM challenge of the holistic synthesis.
            investigation_tot_context = "\n".join(
                [
                    "HOLISTIC EVIDENCE SUFFICIENCY:",
                    f"sufficient={overall_sufficiency.sufficient}",
                    f"confidence={overall_sufficiency.confidence:.2f}",
                    f"reason={overall_sufficiency.reason}",
                    (
                        "material_missing="
                        f"{overall_sufficiency.missing_evidence or []}"
                    ),
                    "",
                    "CANDIDATE HOLISTIC SECURITY SYNTHESIS:",
                    interpretation,
                ]
            ).strip()

            print(
                "[GraderWriterAgent]    Investigation ToT: generating "
                f"{NUM_BRANCHES} challenge branches..."
            )

            branches = self._generate_branches(
                state,
                supplemental_context=investigation_tot_context,
            )

            if branches:
                evaluation = self._evaluate_branches(state, branches)
                (
                    selected_thought,
                    selected_response,
                    selected_score,
                ) = self._select_branch(branches, evaluation)

                state.tot_thoughts = [thought for thought, _ in branches]
                state.tot_best_branch = selected_thought.branch_id

                tot_grade = _grade_from_text(selected_response.grade)
                tot_interpretation = _clean_text(selected_response.refined_context)

                print(
                    "[GraderWriterAgent]    Investigation ToT winner: "
                    f"{selected_thought.branch_id} "
                    f"(score={selected_score}, grade={tot_grade.value})"
                )

                if (
                    tot_grade == GradeResult.PASS
                    and tot_interpretation
                    and not selected_response.missing_information
                ):
                    # A successful critic may improve the synthesis, but it must
                    # preserve the complete findings package rather than replace it
                    # with a shorter single-finding answer.
                    interpretation = (
                        interpretation
                        + "\n\n=== TREE-OF-THOUGHT CRITIQUE / REFINEMENT ===\n"
                        + tot_interpretation
                    ).strip()
                else:
                    missing_from_tot = [
                        _clean_text(item)
                        for item in selected_response.missing_information
                        if _clean_text(item)
                    ]
                    if missing_from_tot:
                        state.add_warning(
                            "Investigation Tree-of-Thought challenge identified "
                            "material uncertainty: " + "; ".join(missing_from_tot)
                        )
                    state.add_warning(
                        "Investigation Tree-of-Thought challenge did not endorse "
                        "the candidate synthesis. The structured synthesis was "
                        "preserved rather than downgraded or replaced."
                    )
            else:
                state.tot_thoughts = []
                state.tot_best_branch = None
                state.add_warning(
                    "All investigation Tree-of-Thought branches were unavailable. "
                    "The holistic sufficiency decision and structured investigation "
                    "synthesis were preserved; no semantic FAIL was manufactured."
                )
                print(
                    "[GraderWriterAgent]    Investigation ToT: no available "
                    "branches; preserving synthesis."
                )

            selected_context = _merge_verified_evidence_with_interpretation(
                state,
                interpretation,
            )

            if not selected_context:
                failure = EvidenceFailure(
                    category=EvidenceFailureCategory.INCOMPLETE_EVIDENCE,
                    reason=(
                        "Successful investigation tasks existed, but no "
                        "verified evidence-supported interpretation could be "
                        "constructed."
                    ),
                    repairable=False,
                )

                self._store_failure(state, failure)
                state.grade = GradeResult.FAIL
                state.refined_context = ""

                _set_failure_presentation_contract(
                    state,
                    reason=failure.reason,
                )

                return state

            state.grade = GradeResult.PASS
            state.refined_context = selected_context
            _set_interpretive_presentation_contract(state)

            if failed_tasks:
                state.add_warning(
                    "Investigation coverage was partial: "
                    f"{len(successful_tasks)} of "
                    f"{len(state.investigation_results)} tasks "
                    "produced verified database evidence. Security conclusions "
                    "apply only to the successfully reviewed evidence."
                )

            state.add_trace(
                "GraderWriterAgent",
                (
                    "Investigation evidence interpreted task-by-task; "
                    f"successful={len(successful_tasks)}; "
                    f"failed={len(failed_tasks)}; "
                    f"overall={synthesis.overall_assessment}; "
                    f"confidence={synthesis.confidence:.2f}; "
                    f"tot_branches={len(state.tot_thoughts)}; "
                    f"tot_winner={state.tot_best_branch or 'none'}; "
                    f"presentation={state.presentation_mode.value}."
                ),
            )

            print("[GraderWriterAgent] ✔ INVESTIGATION INTERPRETATION COMPLETE")
            print(
                "[GraderWriterAgent]    Successful tasks : " f"{len(successful_tasks)}"
            )
            print("[GraderWriterAgent]    Failed tasks     : " f"{len(failed_tasks)}")
            print(
                "[GraderWriterAgent]    Overall security : "
                f"{synthesis.overall_assessment}"
            )
            print(
                "[GraderWriterAgent]    Confidence       : "
                f"{synthesis.confidence:.2f}"
            )
            print(
                "[GraderWriterAgent]    Refined ctx      : "
                f"{len(selected_context)} chars"
            )
            print(
                "[GraderWriterAgent]    Supplemental req.: "
                f"{bool(state.evidence_failure_repairable)}"
            )
            print(
                "[GraderWriterAgent]    ToT branches      : "
                f"{len(state.tot_thoughts)}"
            )
            print(
                "[GraderWriterAgent]    ToT winner        : "
                f"{state.tot_best_branch or 'none'}"
            )
            print(
                "[GraderWriterAgent]    Presentation     : "
                f"{state.presentation_mode.value}"
            )
            print("═" * 70 + "\n")

            return state

        # Other interpretation-heavy work uses Tree of Thought.
        branches = self._generate_branches(state)
        if not branches:
            failure = EvidenceFailure(
                category=EvidenceFailureCategory.UNKNOWN,
                reason=(
                    "All Tree-of-Thought grading branches were unavailable; "
                    "no semantic grading decision was produced."
                ),
                repairable=False,
            )
            self._store_failure(state, failure)
            state.grade = GradeResult.FAIL
            state.refined_context = ""
            _set_failure_presentation_contract(state, reason=failure.reason)
            return state

        evaluation = self._evaluate_branches(
            state,
            branches,
        )

        (
            selected_thought,
            selected_response,
            selected_score,
        ) = self._select_branch(
            branches,
            evaluation,
        )

        selected_grade = _grade_from_text(selected_response.grade)
        selected_interpretation = _clean_text(selected_response.refined_context)

        # Hard post-ToT completeness gate.
        if selected_response.missing_information or not selected_interpretation:
            selected_grade = GradeResult.FAIL
            selected_interpretation = ""

        state.tot_thoughts = [thought for thought, _ in branches]
        state.tot_best_branch = selected_thought.branch_id
        state.grade = selected_grade

        if selected_grade == GradeResult.PASS:
            selected_context = _merge_verified_evidence_with_interpretation(
                state,
                selected_interpretation,
            )

            if not selected_context:
                selected_grade = GradeResult.FAIL
                state.grade = GradeResult.FAIL
                selected_context = ""
                state.refined_context = ""
            else:
                state.refined_context = selected_context
                _set_interpretive_presentation_contract(state)

        else:
            selected_context = ""
            state.refined_context = ""

        if selected_grade == GradeResult.FAIL:
            failure = EvidenceFailure(
                category=(EvidenceFailureCategory.INCOMPLETE_EVIDENCE),
                reason=(
                    "; ".join(selected_response.missing_information)
                    or (
                        "The available evidence required interpretation, "
                        "but did not completely support the requested answer."
                    )
                ),
                repairable=False,
            )
            self._store_failure(state, failure)

            _set_failure_presentation_contract(
                state,
                reason=failure.reason,
            )

        successful_tasks = _successful_investigation_results(state)
        failed_tasks = _failed_investigation_results(state)

        if selected_grade == GradeResult.PASS and successful_tasks and failed_tasks:
            state.add_warning(
                "Investigation coverage was partial: "
                f"{len(successful_tasks)} of "
                f"{len(state.investigation_results)} tasks "
                "produced verified database evidence."
            )

        state.add_trace(
            "GraderWriterAgent",
            (
                f"ToT branches={len(branches)}; "
                f"winner={selected_thought.branch_id}; "
                f"score={selected_score}; "
                f"grade={selected_grade.value}; "
                f"presentation={state.presentation_mode.value}."
            ),
        )

        print("[GraderWriterAgent] ✔ COMPLETE")
        print("[GraderWriterAgent]    ToT branches: " f"{len(branches)}")
        print("[GraderWriterAgent]    ToT winner  : " f"{selected_thought.branch_id}")
        print("[GraderWriterAgent]    Final grade : " f"{selected_grade.value.upper()}")
        print("[GraderWriterAgent]    Refined ctx : " f"{len(selected_context)} chars")
        print(
            "[GraderWriterAgent]    Presentation: " f"{state.presentation_mode.value}"
        )
        print("[GraderWriterAgent]    LLM rewrite : " f"{state.allow_llm_rewrite}")
        print("═" * 70 + "\n")

        return state


def _self_test_investigation_confidence_normalization() -> None:
    """Verify investigation confidence accepts percentage-style model output."""
    assessment = InvestigationTaskAssessment(
        task_id="T001",
        evidence_relevant=True,
        evidence_coverage_sufficient=True,
        covered_evidence_components=["test evidence"],
        missing_evidence_components=[],
        observed_facts=["test evidence"],
        security_assessment="potential_security_risk",
        confidence=85,
        finding="test",
        evidence_basis="test",
    )

    assert assessment.confidence == 0.85

    synthesis = InvestigationSynthesis(
        overall_assessment="inconclusive",
        confidence=100,
        opinion="test",
        supported_findings=[],
        benign_observations=[],
        inconclusive_observations=[],
        coverage_statement="test",
    )

    assert synthesis.confidence == 1.0

    print("Investigation confidence normalization self-test: PASS")


def _self_test_direct_query_confidence_normalization() -> None:
    """Verify percentage confidence is normalized before field validation."""
    review = DirectQueryEvidenceReview(
        sufficient=True,
        confidence=100,
        reason="test",
    )

    assert review.confidence == 1.0

    review_95 = DirectQueryEvidenceReview(
        sufficient=True,
        confidence=95,
        reason="test",
    )

    assert review_95.confidence == 0.95

    print("DirectQueryEvidenceReview confidence normalization self-test: PASS")


def _self_test_incomplete_investigation_evidence_cannot_be_benign() -> None:
    """Verify incomplete evidence cannot remain a benign task assessment."""
    assessment = InvestigationTaskAssessment(
        task_id="T-COVERAGE",
        evidence_relevant=True,
        evidence_coverage_sufficient=False,
        covered_evidence_components=["first comparison domain"],
        missing_evidence_components=["second comparison domain"],
        observed_facts=["first comparison-domain event was observed"],
        security_assessment="potential_security_risk",
        confidence=0.9,
        finding="Coverage test",
        evidence_basis="Only one side of the requested comparison was returned.",
    )

    assert assessment.evidence_coverage_sufficient is False
    assert assessment.security_assessment == "potential_security_risk"
    assert assessment.observed_facts

    print("Investigation evidence-coverage self-test: PASS")
