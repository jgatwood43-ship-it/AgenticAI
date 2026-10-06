"""
agents/retriever_agent.py
─────────────────────────
Agent 2 – Evidence Retrieval and Execution.

Responsibilities
----------------
Execute the evidence contract established by Agent 1:

    required_evidence_types
    requires_live_data
    requires_policy_evidence
    requires_interpretation
    investigation_requested

Agent 2 does not rediscover the user's intent from isolated words such as
"table", "schema", "record", or "policy". Query-text heuristics are retained
only as a backward-compatible fallback when no evidence contract exists.

Evidence execution order
------------------------
1. Retrieve requested policy evidence.
2. Retrieve authoritative schema grounding when required.
3. Execute either:
   - a multi-task investigation through InvestigationPlanner/SQLPlanner; or
   - one bounded DirectQuery through DirectQueryPlanner.
4. Build separated evidence context for Agent 3.

A structural schema request may use schema evidence without live data.
An analytical request mentioning tables may require schema, policy, database,
and investigation evidence simultaneously.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
from pathlib import Path
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage, LLM
from pydantic import BaseModel, Field, field_validator

from config.settings import settings as db_config
from core.direct_query_planner import (
    DirectQueryPlan,
    DirectQueryPlanner,
)
from core.direct_sql_validator import DirectSQLValidator
from core.investigation_planner import (
    InvestigationPlanner,
    is_investigation_query,
)
from core.schema_catalog import load_full_schema_catalog
from core.schema_graph import SchemaGraph
from core.schema_model import parse_schema_context
from core.schema_retriever import SchemaRetriever
from core.sql_planner import SQLPlanner, requires_mysql_explain
from core.system_prompts import EVIDENCE_REVIEW_SYSTEM_PROMPT, SQL_PLANNER_SYSTEM_PROMPT
from core.state import (
    EvidenceType,
    InvestigationTaskResult,
    RetrievalMode,
    SQLPlan,
    WorkflowState,
)
from tools.mcp_tools import mysql_query_tool


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
# Compatibility heuristics
# ============================================================================

NO_RETRIEVAL_PATTERNS = (
    r"what is your purpose",
    r"who are you",
    r"good morning",
    r"hello",
    r"hi",
)

_SCHEMA_INFORMATION_TERMS = (
    "schema",
    "schemas",
    "table structure",
    "database structure",
    "data model",
    "table relationships",
    "columns and relationships",
    "primary key",
    "foreign key",
    "describe the tables",
    "show the tables",
    "list the tables",
)

_ANALYTICAL_TERMS = (
    "review",
    "inspect",
    "assess",
    "analyze",
    "analyse",
    "evaluate",
    "compare",
    "correlate",
    "investigate",
    "determine if",
    "determine whether",
    "security concern",
    "security issue",
    "security risk",
    "suspicious",
    "anomaly",
    "inconsistency",
    "violation",
    "review for",
)

_POLICY_TERMS = (
    "nist",
    "cis",
    "policy",
    "policies",
    "standard",
    "standards",
    "control",
    "controls",
    "guidance",
    "documents on file",
)

_DATABASE_TERMS = (
    "employee",
    "employees",
    "badge",
    "access",
    "room",
    "clock",
    "punch",
    "department",
    "job title",
    "account",
    "user",
    "login",
    "key",
    "camera",
    "video",
    "record",
    "records",
    "activity",
    "events",
    "count",
    "how many",
    "who",
)

SIMILARITY_TOP_K = 5
POLICY_FALLBACK_TOP_K = 8
MIN_CONTEXT_CHARS = 20
MIN_PRINTABLE_RATIO = 0.80
MIN_ALPHANUMERIC_RATIO = 0.15


MAX_DIRECT_QUERY_EXECUTIONS = 2
MAX_INTERNAL_INVESTIGATION_PASSES = 3
MAX_INTERNAL_SQL_REFINEMENTS = 3
MAX_MYSQL_EXPLAIN_REPAIRS = 2


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


class InvestigationEvidenceSufficiencyReview(BaseModel):
    """LLM judgment of whether verified investigation evidence is enough to answer now."""

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


_INVESTIGATION_EVIDENCE_STOP_PROMPT = """
You are deciding whether a database investigation should STOP RETRIEVING and
move to security reasoning.

ORIGINAL USER QUESTION
----------------------
{query}

PERSISTENT LEDGER
-----------------
{ledger_context}

The ledger is advisory memory. Its semantic labels are NOT mandatory checklist
items and are NOT database identifiers.

INVESTIGATION PLAN COVERAGE
---------------------------
Planned tasks: {planned_task_count}
Successful verified tasks so far: {successful_task_count}
Remaining planned tasks: {remaining_task_count}

Remaining task objectives:
{remaining_tasks}

The plan is evidence-discovery context, not a mandatory checklist. Use it to
understand the breadth the investigation LLM itself considered relevant.

VERIFIED SUCCESSFUL INVESTIGATION EVIDENCE SO FAR
-------------------------------------------------
{evidence}

TASK
----
Decide whether the verified evidence gathered so far is sufficient for a
competent LLM to make a defensible, useful, evidence-grounded response to the
ORIGINAL USER QUESTION.

Reason about the scope of the user's request. Distinguish a narrow/existential
question (for example, whether a particular condition occurred) from a broad
review request (for example, review company records for security concerns).
Finding one concern may answer an existential question, but it does not by
itself establish that a broad review has adequate coverage.

Rules:
1. Judge the evidence AS A WHOLE, not task-by-task.
2. A successful SQL execution alone is not sufficient; inspect what the SQL
   tested and what the returned evidence contains.
3. Do NOT require every ledger label or every planned task to be completed.
4. For a narrow question, one successful query may be sufficient when it
   directly tests the requested condition and returns enough evidence to answer.
5. For a broad review, do not stop merely because the first potential concern
   was found. Decide whether the evidence provides reasonable breadth across the
   material areas identified by the investigation plan, OR whether the remaining
   planned areas are genuinely immaterial to a defensible answer.
6. A broad review may stop before every planned task is executed, but only when
   you can explain why the successful evidence adequately covers the original
   review objective rather than merely proving that at least one issue exists.
7. When a potential concern has been found, consider whether it is characterized
   well enough for a useful final answer. When supported and available from the
   documented data, useful characterization normally includes the affected actor
   or identifier, what happened, when, where, and the factual basis for concern.
   Do not require unavailable attributes and do not invent them.
8. Distinguish a suspicious or unauthorized ATTEMPT that was blocked from a
   control FAILURE that allowed unauthorized activity. A Denied event may show
   the control enforced a restriction even though the attempted activity itself
   can still be security-relevant.
9. Evidence from multiple successful queries may collectively be sufficient.
10. Do not demand additional evidence merely because another table, query, or
    representation could also be consulted.
11. If sufficient=false, missing_evidence must contain ONLY concrete factual
    evidence still required to answer the original question. Express each item as
    a POSITIVE real-world fact to retrieve or establish.
12. Before claiming that identity, timestamps, rooms, tables, or another factual
    field/domain is missing, inspect the Executed SQL, Columns, and Evidence.
13. Distinguish "the evidence domain is missing" from "the evidence is present
    but the correlation/predicate has not yet established the requested
    relationship." When necessary tables and fields are already present,
    describe the missing RELATIONSHIP or state-at-event-time fact.
14. The free-text reason may explain why the current evidence is insufficient,
    but missing_evidence must identify WHAT additional real-world fact is needed.
15. Do not generate SQL and do not make the final user-facing security judgment.

Return a structured sufficiency decision.
"""


_DIRECT_QUERY_RESULT_REVIEW_PROMPT = """
You are reviewing EXECUTED database evidence for a direct factual user question.

ORIGINAL USER QUESTION
----------------------
{query}

EXECUTED SQL
------------
{sql}

VERIFIED SQL FACTS
------------------
The SQL above has already passed deterministic authoritative validation.

Validated tables:
{validated_tables}

Validated columns:
{validated_columns}

Validated relationships:
{validated_relationships}

You MUST accept these as settled facts:
- every validated table exists;
- every validated column exists on the table shown;
- every validated relationship is documented;
- the SQL is read-only and structurally valid.

Do NOT contradict, reinterpret, or re-validate those schema facts.

DATABASE RESULT
---------------
Rows returned: {row_count}
Columns returned: {columns}

Evidence:
{evidence}

TASK
----
Your ONLY task is to decide whether the ACTUAL RETURNED DATABASE RESULT contains
the information needed to answer the ORIGINAL USER QUESTION.

Rules:
1. Successful SQL execution alone does not prove answer sufficiency.
2. Judge the returned evidence, not whether you would have chosen different
   tables, columns, joins, or SQL.
3. Never claim that a validated table, column, or relationship does not exist
   or belongs somewhere else.
4. You MAY identify a semantic mismatch when the returned evidence itself is
   about a different operational fact than the user requested.
5. If rows are returned and the returned fields directly contain the requested
   fact, treat that as sufficient unless the result itself shows a concrete
   semantic mismatch.
6. Zero rows MAY be sufficient. If the validated SQL directly tests the
   requested fact without an unsupported restriction, zero rows means no
   matching record was found and should normally be accepted as an answer.
7. Do not demand a second query merely because another table or representation
   might also contain related information.
8. If insufficient, explain only what requested information is missing from the
   returned result or what concrete semantic mismatch exists.
9. Do not generate SQL.
"""


_DIRECT_QUERY_RESULT_REPAIR_PROMPT = """
You are generating a fresh read-only MySQL query after an earlier query executed
successfully but the returned evidence did not answer the original question.

ORIGINAL USER QUESTION
----------------------
{query}

AUTHORITATIVE DATABASE SCHEMA
-----------------------------
{schema_context}

PREVIOUS EXECUTED SQL
---------------------
{prior_sql}

PREVIOUS DATABASE RESULT
------------------------
Rows returned: {row_count}
Columns returned: {columns}

Evidence:
{evidence}

WHY THE RESULT WAS INSUFFICIENT
-------------------------------
{reason}

INSTRUCTIONS
------------
1. Generate a NEW query that directly answers the ORIGINAL USER QUESTION.
2. Correct the semantic defect identified above.
3. Use only documented tables, columns, and relationships.
4. Do not substitute a different operational domain for the requested fact.
5. Do not invent date ranges, thresholds, status values, categories, business
   meanings, or other restrictions not supplied by the user or authoritative
   schema/business rules.
6. Choose the SQL construction yourself.
7. Return exactly one read-only MySQL query.
8. Return SQL only. Do not explain it and do not use Markdown fences.
"""


# ============================================================================
# General helpers
# ============================================================================


def _clean_text(
    value: Any,
) -> str:
    return str(value or "").strip()


def _normalize_confidence_dict(
    value: dict[str, Any],
) -> dict[str, Any]:
    """
    Normalize local-model confidence values.

    Some local models return confidence as a percentage (for example, 95 or
    100) even when the structured schema expects a 0.0-1.0 value. Match the
    normalization behavior used by GraderWriterAgent.
    """
    normalized = dict(value)

    confidence = normalized.get("confidence")

    if isinstance(confidence, (int, float)) and confidence > 1 and confidence <= 100:
        normalized["confidence"] = confidence / 100

    return normalized


def _extract_structured_model(
    response: Any,
    response_model: type[BaseModel],
) -> BaseModel:
    """Extract a structured Pydantic response from LlamaIndex wrappers."""
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
        raise ValueError("Structured LLM returned an empty response.")

    decoded = json.loads(text)

    if isinstance(decoded, dict):
        decoded = _normalize_confidence_dict(decoded)

    return response_model.model_validate(decoded)


def _extract_sql_candidate(
    value: Any,
) -> str:
    """Extract one SQL query from a possibly chatty local-model response."""
    text = _clean_text(getattr(value, "text", value))

    if not text:
        return ""

    fenced = re.search(
        r"```(?:sql|mysql)?\s*(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if fenced is not None:
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

    if start is not None:
        text = text[start.start() :].strip()

    semicolon = text.find(";")

    if semicolon >= 0:
        text = text[: semicolon + 1]

    return text.strip()


def _contains_any(
    text: str,
    terms: tuple[str, ...],
) -> bool:
    lowered = _clean_text(text).lower()

    return any(term in lowered for term in terms)


def should_skip_retrieval(
    query: str,
) -> bool:
    normalized_query = _clean_text(query)

    return any(
        re.fullmatch(
            rf"\s*{pattern}[?.!,]*\s*",
            normalized_query,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in NO_RETRIEVAL_PATTERNS
    )


def _text_quality_metrics(
    text: str,
) -> tuple[int, float, float]:
    cleaned = str(text or "").strip()

    if not cleaned:
        return (
            0,
            0.0,
            0.0,
        )

    printable_count = sum(character.isprintable() for character in cleaned)

    alphanumeric_count = sum(character.isalnum() for character in cleaned)

    return (
        len(cleaned),
        printable_count / len(cleaned),
        alphanumeric_count / len(cleaned),
    )


def _is_usable_text(
    text: str,
) -> bool:
    (
        text_length,
        printable_ratio,
        alphanumeric_ratio,
    ) = _text_quality_metrics(text)

    return (
        text_length >= MIN_CONTEXT_CHARS
        and printable_ratio >= MIN_PRINTABLE_RATIO
        and alphanumeric_ratio >= MIN_ALPHANUMERIC_RATIO
    )


# def _is_usable_text(
#    text: str,
# ) -> bool:
#    """
#    Reject empty or corrupted text without imposing a semantic relevance
#    threshold. Semantic relevance is supplied by vector ranking.
#    """
#    cleaned = _clean_text(text)

#    if len(cleaned) < MIN_CONTEXT_CHARS:
#        return False

#    printable_count = sum(character.isprintable() for character in cleaned)

#    alphanumeric_count = sum(character.isalnum() for character in cleaned)

#    return (
#        printable_count / len(cleaned) >= MIN_PRINTABLE_RATIO
#        and alphanumeric_count / len(cleaned) >= MIN_ALPHANUMERIC_RATIO
#    )


def _extract_policy_source(
    node_with_score: Any,
) -> str:
    node = getattr(
        node_with_score,
        "node",
        None,
    )

    if node is None:
        return ""

    metadata = (
        getattr(
            node,
            "metadata",
            {},
        )
        or {}
    )

    for key in (
        "file_name",
        "filename",
        "source",
        "document_title",
        "title",
        "file_path",
    ):
        value = metadata.get(key)

        if value:
            return _clean_text(value)

    return ""


def _normalize_mysql_result(
    raw_result: str,
) -> tuple[
    str,
    list[dict[str, Any]],
]:
    cleaned = _clean_text(raw_result)

    if not cleaned:
        return "", []

    try:
        parsed: Any = json.loads(cleaned)

    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(cleaned)

        except (
            ValueError,
            SyntaxError,
        ):
            return cleaned, []

    if isinstance(
        parsed,
        list,
    ):
        for item in parsed:
            if not isinstance(
                item,
                dict,
            ):
                continue

            inner_text = item.get("text")

            if not inner_text:
                continue

            try:
                inner = json.loads(inner_text)

            except json.JSONDecodeError:
                continue

            rows = inner.get(
                "rows",
                [],
            )

            if isinstance(
                rows,
                list,
            ):
                normalized_rows = [
                    row
                    for row in rows
                    if isinstance(
                        row,
                        dict,
                    )
                ]

                return (
                    json.dumps(
                        {
                            "rows": normalized_rows,
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    normalized_rows,
                )

    if isinstance(
        parsed,
        dict,
    ):
        rows = parsed.get(
            "rows",
            [],
        )

        normalized_rows = (
            [
                row
                for row in rows
                if isinstance(
                    row,
                    dict,
                )
            ]
            if isinstance(
                rows,
                list,
            )
            else []
        )

        return (
            json.dumps(
                parsed,
                indent=2,
                ensure_ascii=False,
            ),
            normalized_rows,
        )

    return cleaned, []


def _result_looks_like_error(
    result: str,
) -> bool:
    lowered = _clean_text(result).lower()

    return any(
        marker in lowered
        for marker in (
            "database error:",
            "mcpservererror",
            "sql parse error",
            "unknown column",
            "doesn't exist",
            "connection refused",
            "pool timed out",
            "expected: select",
            "found: eof",
        )
    )


_INFRASTRUCTURE_ERROR_MARKERS = (
    "pool timed out",
    "connection refused",
    "connection reset",
    "server has gone away",
    "lost connection",
    "transport error",
    "timed out waiting for",
    "mcp server is not running",
    "mcp stdin stream is unavailable",
    "mcp stdout stream is unavailable",
)


def _is_infrastructure_database_error(error_text: str) -> bool:
    """Return True when a failure says nothing about SQL validity."""
    lowered = _clean_text(error_text).lower()
    return any(marker in lowered for marker in _INFRASTRUCTURE_ERROR_MARKERS)


# ============================================================================
# Evidence-contract resolution
# ============================================================================


def _fallback_required_evidence_types(
    query: str,
) -> list[str]:
    """
    Backward-compatible inference for callers that bypass Agent 1.

    The operation is considered before the nouns. Analytical table requests
    are not treated as schema-only requests.
    """
    required: list[str] = []

    analytical = _contains_any(
        query,
        _ANALYTICAL_TERMS,
    )

    policy = _contains_any(
        query,
        _POLICY_TERMS,
    )

    schema = _contains_any(
        query,
        _SCHEMA_INFORMATION_TERMS,
    )

    database = _contains_any(
        query,
        _DATABASE_TERMS,
    )

    investigation = (analytical and database) or is_investigation_query(query)

    if policy:
        required.append(EvidenceType.POLICY.value)

    if schema or database or investigation:
        required.append(EvidenceType.SCHEMA.value)

    if database:
        required.append(EvidenceType.DATABASE.value)

    if investigation:
        required.append(EvidenceType.INVESTIGATION.value)

    return list(dict.fromkeys(required))


def _capture_evidence_contract(
    state: WorkflowState,
    query: str,
) -> dict[str, Any]:
    """
    Snapshot Agent 1's contract before retrieval state is cleared.

    Some existing WorkflowState.reset_retrieval_state implementations reset
    investigation_requested. Capturing and restoring the contract prevents
    Agent 2 from losing Agent 1's decision.
    """
    required_types = list(
        getattr(
            state,
            "required_evidence_types",
            [],
        )
        or []
    )

    if not required_types:
        required_types = _fallback_required_evidence_types(query)

    requires_live_data = bool(
        getattr(
            state,
            "requires_live_data",
            False,
        )
    )

    requires_policy = bool(
        getattr(
            state,
            "requires_policy_evidence",
            False,
        )
    )

    requires_interpretation = bool(
        getattr(
            state,
            "requires_interpretation",
            False,
        )
    )

    investigation_requested = bool(
        getattr(
            state,
            "investigation_requested",
            False,
        )
    )

    if EvidenceType.DATABASE.value in required_types:
        requires_live_data = True

    if EvidenceType.POLICY.value in required_types:
        requires_policy = True

    if EvidenceType.INVESTIGATION.value in required_types:
        investigation_requested = True
        requires_live_data = True
        requires_interpretation = True

    retrieval_mode = getattr(
        state,
        "retrieval_mode",
        RetrievalMode.EVIDENCE_ONLY,
    )

    if not isinstance(
        retrieval_mode,
        RetrievalMode,
    ):
        try:
            retrieval_mode = RetrievalMode(str(retrieval_mode).strip().lower())
        except ValueError:
            retrieval_mode = (
                RetrievalMode.INVESTIGATION
                if investigation_requested
                else (
                    RetrievalMode.DIRECT_QUERY
                    if requires_live_data
                    else RetrievalMode.EVIDENCE_ONLY
                )
            )

    # Legacy callers without the new routing mode are normalized here.
    if investigation_requested:
        retrieval_mode = RetrievalMode.INVESTIGATION
    elif requires_live_data and retrieval_mode == RetrievalMode.EVIDENCE_ONLY:
        retrieval_mode = RetrievalMode.DIRECT_QUERY

    return {
        "required_evidence_types": required_types,
        "requires_live_data": requires_live_data,
        "requires_policy_evidence": requires_policy,
        "requires_interpretation": requires_interpretation,
        "investigation_requested": investigation_requested,
        "retrieval_mode": retrieval_mode.value,
        "retrieval_mode_reason": _clean_text(
            getattr(
                state,
                "retrieval_mode_reason",
                "",
            )
        ),
        "evidence_requirement_reason": _clean_text(
            getattr(
                state,
                "evidence_requirement_reason",
                "",
            )
        ),
    }


def _restore_evidence_contract(
    state: WorkflowState,
    contract: dict[str, Any],
) -> None:
    state.required_evidence_types = list(contract["required_evidence_types"])

    state.requires_live_data = bool(contract["requires_live_data"])

    state.requires_policy_evidence = bool(contract["requires_policy_evidence"])

    state.requires_interpretation = bool(contract["requires_interpretation"])

    state.set_retrieval_mode(
        contract.get(
            "retrieval_mode",
            (
                RetrievalMode.INVESTIGATION.value
                if contract["investigation_requested"]
                else (
                    RetrievalMode.DIRECT_QUERY.value
                    if contract["requires_live_data"]
                    else RetrievalMode.EVIDENCE_ONLY.value
                )
            ),
        ),
        reason=contract.get(
            "retrieval_mode_reason",
            contract["evidence_requirement_reason"],
        ),
    )

    state.evidence_requirement_reason = _clean_text(
        contract["evidence_requirement_reason"]
    )

    state.normalize_evidence_requirements()


# ============================================================================
# RetrieverAgent
# ============================================================================


class RetrieverAgent:
    """Execute policy, schema, database, and investigation evidence contracts."""

    def __init__(
        self,
        llm: LLM,
        index: VectorStoreIndex,
        schema_retriever: SchemaRetriever,
    ) -> None:
        self._index = index
        self._schema_retriever = schema_retriever
        self._llm = llm
        self._direct_result_review_llm = llm.as_structured_llm(
            DirectQueryEvidenceReview
        )
        self._investigation_sufficiency_llm = llm.as_structured_llm(
            InvestigationEvidenceSufficiencyReview
        )

        # Structured SQLPlanner remains available only for investigations.
        self._sql_planner = SQLPlanner(llm=llm)

        project_root = Path(__file__).resolve().parents[1]
        schema_directory = project_root / "docs" / "schema"

        self._schema_graph = SchemaGraph.from_directory(schema_directory)

        self._direct_sql_validator = DirectSQLValidator(schema_graph=self._schema_graph)

        # DirectQueryPlanner is the single authority for:
        #   LLM generation → deterministic validation → one bounded LLM repair.
        # RetrieverAgent only executes SQL after planner approval.
        self._direct_query_planner = DirectQueryPlanner(
            llm=llm,
            validator=self._direct_sql_validator,
        )

        self._investigation_planner = InvestigationPlanner(
            llm=llm,
            schema_graph=self._schema_graph,
        )

        self._full_schema_catalog = load_full_schema_catalog()

        print(
            "\n[RetrieverAgent] ⚙  Initialising evidence-contract "
            "retrieval pipeline…"
        )
        print("[RetrieverAgent]    Embedding model : " f"{db_config.embed_model_name}")
        print(
            "[RetrieverAgent]    Policy Vector DB: "
            f"{db_config.pg_host}:{db_config.pg_port}/"
            f"{db_config.pg_database}"
        )
        print("[RetrieverAgent]    Policy table    : " f"{db_config.pg_table}")
        print(
            "[RetrieverAgent]    ANN index       : "
            f"HNSW (similarity_top_k={SIMILARITY_TOP_K})"
        )
        print(
            "[RetrieverAgent]    Schema catalog  : "
            f"{len(self._full_schema_catalog)} chars"
        )
        print(
            "[RetrieverAgent]    SchemaGraph     : "
            f"{len(self._schema_graph.tables)} tables / "
            f"{len(self._schema_graph.relationships)} relationships"
        )
        print("[RetrieverAgent]    Contract source : Agent 1")
        print(
            "[RetrieverAgent]    Fallback        : " "operation-aware query inference"
        )
        print(
            "[RetrieverAgent]    DirectQuery     : "
            "DirectQueryPlanner → approved SQL → MySQL"
        )
        print(
            "[RetrieverAgent]    Direct repairs  : "
            "planner corrections + evidence-only result review + one bounded re-query"
        )
        print(
            "[RetrieverAgent]    Investigation SQL validation: Python safety/preflight → MySQL EXPLAIN → bounded LLM repair"
        )
        print(
            "[RetrieverAgent]    Investigation loop: reason → query → observe → "
            "LLM sufficiency review → refine successful SQL → re-plan only if needed"
        )
        print(
            "[RetrieverAgent]    Internal evidence passes: "
            f"up to {MAX_INTERNAL_INVESTIGATION_PASSES} post-result re-plans"
        )
        print(
            "[RetrieverAgent]    SQL evidence refinement: up to "
            f"{MAX_INTERNAL_SQL_REFINEMENTS} successful-query refinements "
            "before opening a new evidence domain"
        )
        print(
            "[RetrieverAgent]    Investigation SQL: "
            "semantic evidence tasks + independent LLM table selection + "
            "complete-schema hard validation"
        )
        print(
            "[RetrieverAgent]    MCP execution   : maximum two DirectQuery executions"
        )

    # ------------------------------------------------------------------------
    # Schema evidence
    # ------------------------------------------------------------------------

    def _retrieve_schema_context(
        self,
        query: str,
        *,
        force_full_catalog: bool = False,
    ) -> tuple[
        str,
        list[str],
    ]:
        """
        Return authoritative schema evidence.

        RetrieverAgent now uses the complete Markdown schema catalog as the
        schema evidence source. SchemaGraph is responsible for deterministic
        table/relationship scoping for investigations, so Agent 2 no longer
        depends on a separate targeted schema-document loader.

        force_full_catalog is retained for interface compatibility.
        """
        del query
        del force_full_catalog

        schema_context = self._full_schema_catalog.strip()

        selected_documents = ["complete_schema_catalog"]

        print(
            "[RetrieverAgent] [Schema] " "Using complete authoritative schema catalog."
        )
        print(
            "[RetrieverAgent] [Schema] Context length: " f"{len(schema_context)} chars"
        )

        return (
            schema_context,
            selected_documents,
        )

    def _prepare_schema_evidence(
        self,
        state: WorkflowState,
        *,
        full_catalog: bool,
        query: str,
    ) -> None:
        """
        Store schema documentation as both grounding and answerable evidence.

        The complete catalog is currently used for schema grounding. DirectQuery
        and investigation planning both consume authoritative schema evidence.
        """
        (
            state.schema_context,
            state.schema_sources,
        ) = self._retrieve_schema_context(
            query,
            force_full_catalog=full_catalog,
        )

        state.schema_evidence = state.schema_context

        state.schema_query_succeeded = bool(state.schema_evidence.strip())

        print(
            "[RetrieverAgent] [Schema] Evidence prepared: "
            f"{state.schema_query_succeeded} "
            f"({len(state.schema_evidence)} chars)"
        )

    # ------------------------------------------------------------------------
    # Policy evidence
    # ------------------------------------------------------------------------

    def _retrieve_policy_evidence(
        self,
        query: str,
        state: WorkflowState,
        *,
        explicitly_required: bool,
    ) -> int:
        print(
            "[RetrieverAgent] [Policy RAG] Searching pgvector "
            "for relevant NIST/CIS content..."
        )

        top_k = POLICY_FALLBACK_TOP_K if explicitly_required else SIMILARITY_TOP_K

        try:
            policy_nodes = self._index.as_retriever(similarity_top_k=top_k).retrieve(
                query
            )

        except Exception as exc:
            policy_nodes = []

            warning = "Policy retrieval failed: " f"{type(exc).__name__}: {exc}"

            state.add_warning(warning)

            print(f"[RetrieverAgent] ⚠️ {warning}")

        usable_sections: list[str] = []
        policy_sources: list[str] = []

        for (
            node_number,
            node_with_score,
        ) in enumerate(
            policy_nodes,
            start=1,
        ):
            node = getattr(
                node_with_score,
                "node",
                None,
            )

            if node is None:
                continue

            content = _clean_text(node.get_content())

            score = getattr(
                node_with_score,
                "score",
                None,
            )

            score_text = (
                f"{score:.4f}"
                if isinstance(
                    score,
                    (int, float),
                )
                else "N/A"
            )

            usable = _is_usable_text(content)

            (
                text_length,
                printable_ratio,
                alphanumeric_ratio,
            ) = _text_quality_metrics(content)

            print(
                f"[RetrieverAgent] [Policy RAG] Node {node_number}: "
                f"score={score_text}, "
                f"usable={usable}, "
                f"chars={text_length}, "
                f"printable={printable_ratio:.3f}, "
                f"alphanumeric={alphanumeric_ratio:.3f}"
            )

            print("[RetrieverAgent] [Policy RAG] Preview: " f"{repr(content[:160])}")

            #            print(
            #                f"[RetrieverAgent] [Policy RAG] Node {node_number}: "
            #                f"score={score_text}, usable={usable}, "
            #                f"chars={len(content)}"
            #            )

            if not usable:
                continue

            source = _extract_policy_source(node_with_score)

            if source and source not in policy_sources:
                policy_sources.append(source)

            usable_sections.append(
                "\n".join(
                    [
                        f"POLICY PASSAGE {node_number}",
                        f"SOURCE: {source or 'unknown'}",
                        f"SIMILARITY SCORE: {score_text}",
                        "",
                        content,
                    ]
                )
            )

        state.policy_nodes = policy_nodes
        state.retrieved_nodes = policy_nodes
        state.policy_sources = policy_sources

        state.policy_context = "\n\n".join(usable_sections).strip()

        if explicitly_required and not state.policy_context:
            state.add_warning(
                (
                    "Policy evidence was explicitly required, but no usable "
                    "NIST/CIS passages were retrieved."
                )
            )

        print(
            "[RetrieverAgent] [Policy RAG] Usable passages: " f"{len(usable_sections)}"
        )
        print(
            "[RetrieverAgent] [Policy RAG] Policy context: "
            f"{len(state.policy_context)} chars"
        )

        return len(usable_sections)

    @staticmethod
    def _callable_accepts_keyword(
        callable_obj: Any,
        keyword: str,
    ) -> bool:
        """Return True when the installed SQLPlanner method accepts keyword."""
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return False

        if keyword in signature.parameters:
            return True

        return any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def _sql_planner_graph_kwargs(
        self,
        planner_method: Any,
        *,
        allowed_tables: list[str] | None,
        allowed_relationship_paths: list[str] | None,
    ) -> dict[str, Any]:
        """
        InvestigationPlanner table/path selections are metadata only.

        They are intentionally NOT forwarded to SQLPlanner. SQLPlanner owns
        physical table/column selection from the complete authoritative schema.
        """
        del planner_method
        del allowed_tables
        del allowed_relationship_paths
        return {}

    @staticmethod
    def _trace_graph_constraints(
        *,
        allowed_tables: list[str] | None,
        allowed_relationship_paths: list[str] | None,
    ) -> None:
        tables = list(allowed_tables or [])
        paths = list(allowed_relationship_paths or [])

        if not tables and not paths:
            return

        print(
            "[RetrieverAgent] [InvestigationPlanner metadata] Tables selected upstream: "
            f"{tables or '(none supplied)'}"
        )
        print(
            "[RetrieverAgent] [InvestigationPlanner metadata] Paths selected upstream:"
        )

        if paths:
            for path in paths:
                print("[RetrieverAgent] [InvestigationPlanner metadata]   - " f"{path}")
        else:
            print(
                "[RetrieverAgent] [InvestigationPlanner metadata]   - "
                "(single-table task; no relationship path required)"
            )

    # ------------------------------------------------------------------------
    # DirectQuery — bounded live-data lookups
    # ------------------------------------------------------------------------

    @staticmethod
    def _direct_query_plan_to_sql_plan(
        plan: DirectQueryPlan,
    ) -> SQLPlan:
        """
        Adapt DirectQueryPlan to the existing WorkflowState.SQLPlan structure.

        This preserves compatibility with Agent 3 / Agent 4 traces while
        keeping DirectQuery planning out of RetrieverAgent.
        """
        validation = plan.validation

        return SQLPlan(
            purpose=plan.question,
            database_type="mysql",
            tables=(list(validation.tables) if validation is not None else []),
            columns=(list(validation.columns) if validation is not None else []),
            relationships=(
                list(validation.relationships) if validation is not None else []
            ),
            filters=[],
            sql=plan.sql,
            is_valid=bool(
                plan.approved and validation is not None and validation.is_valid
            ),
            validation_errors=(
                []
                if plan.approved
                else (
                    list(validation.errors)
                    if validation is not None and validation.errors
                    else (
                        [plan.error]
                        if plan.error
                        else ["DirectQuery planning was not approved."]
                    )
                )
            ),
        )

    def _record_direct_query_plan(
        self,
        *,
        state: WorkflowState,
        plan: DirectQueryPlan,
    ) -> None:
        """Store DirectQuery planning metadata in existing workflow fields."""
        sql_plan = self._direct_query_plan_to_sql_plan(plan)

        self._record_plan(
            state,
            sql_plan,
        )

        if plan.validation is not None:
            state.schema_tables = list(plan.validation.tables)
            state.schema_relationships = list(plan.validation.relationships)

        print("[RetrieverAgent] [DirectQuery] Planner status : " f"{plan.status.value}")
        print("[RetrieverAgent] [DirectQuery] Approved       : " f"{plan.approved}")
        print(
            "[RetrieverAgent] [DirectQuery] Validation fixes: "
            f"{plan.validation_repairs}"
        )
        if plan.validation is not None:
            print(
                "[RetrieverAgent] [DirectQuery] Tables         : "
                f"{plan.validation.tables}"
            )
            print(
                "[RetrieverAgent] [DirectQuery] Columns        : "
                f"{plan.validation.columns}"
            )
            print(
                "[RetrieverAgent] [DirectQuery] Relationships  : "
                f"{plan.validation.relationships}"
            )

    def _execute_direct_sql(
        self,
        sql: str,
    ) -> tuple[
        bool,
        str,
    ]:
        """
        Execute SQL already approved by DirectQueryPlanner.

        No planning or repair occurs here.
        """
        cleaned_sql = _clean_text(sql)

        if not cleaned_sql:
            return (
                False,
                "DirectQueryPlanner approved an empty SQL statement.",
            )

        print("[RetrieverAgent] [DirectQuery/MCP] " "Executing planner-approved SQL...")
        print("[RetrieverAgent] [DirectQuery/MCP] SQL:")
        print(cleaned_sql)

        try:
            result = _clean_text(mysql_query_tool(cleaned_sql))

        except Exception as exc:
            return (
                False,
                f"{type(exc).__name__}: {exc}",
            )

        if _result_looks_like_error(result):
            return (
                False,
                result,
            )

        return (
            True,
            result,
        )

    @staticmethod
    def _preview_database_result(
        raw_result: str,
    ) -> tuple[str, list[dict[str, Any]], list[str]]:
        normalized_evidence, rows = _normalize_mysql_result(raw_result)

        columns = list(rows[0].keys()) if rows else []

        return (
            normalized_evidence,
            rows,
            columns,
        )

    def _review_direct_query_result(
        self,
        *,
        query: str,
        sql: str,
        raw_result: str,
        validated_tables: list[str],
        validated_columns: list[str],
        validated_relationships: list[str],
    ) -> DirectQueryEvidenceReview:
        """
        Ask the LLM only whether the executed result answers the question.

        Deterministic schema/safety validation is already complete. The reviewer
        receives those validated facts as authoritative and may not re-litigate
        them.
        """
        evidence, rows, columns = self._preview_database_result(raw_result)

        prompt = _DIRECT_QUERY_RESULT_REVIEW_PROMPT.format(
            query=query,
            sql=sql,
            validated_tables=validated_tables or ["(none)"],
            validated_columns=validated_columns or ["(none)"],
            validated_relationships=(validated_relationships or ["(none required)"]),
            row_count=len(rows),
            columns=columns,
            evidence=(
                evidence.strip()
                or "(query executed successfully and returned zero rows)"
            ),
        )

        try:
            parsed = _extract_structured_model(
                _complete_with_system(
                    self._direct_result_review_llm,
                    system_prompt=EVIDENCE_REVIEW_SYSTEM_PROMPT,
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
                    "DirectQuery result sufficiency review failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    def _generate_result_driven_retry_sql(
        self,
        *,
        query: str,
        schema_context: str,
        prior_sql: str,
        raw_result: str,
        reason: str,
    ) -> str:
        """Generate a fresh SQL candidate from post-execution semantic feedback."""
        evidence, rows, columns = self._preview_database_result(raw_result)

        physical_schema = parse_schema_context(schema_context).render_physical_schema(
            include_descriptions=True,
            include_relationships=True,
        )

        repair_prompt = _DIRECT_QUERY_RESULT_REPAIR_PROMPT.format(
            query=query,
            schema_context=physical_schema,
            prior_sql=prior_sql,
            row_count=len(rows),
            columns=columns,
            evidence=(
                evidence.strip() or "(query executed but returned no evidence rows)"
            ),
            reason=reason,
        )

        response = _complete_with_system(
            self._llm,
            system_prompt=SQL_PLANNER_SYSTEM_PROMPT,
            user_prompt=repair_prompt,
        )

        return _extract_sql_candidate(response)

    def _execute_direct_query(
        self,
        *,
        query: str,
        state: WorkflowState,
    ) -> None:
        """
        Plan and execute a DirectQuery with one bounded post-execution semantic
        re-query opportunity.

        DirectQueryPlanner owns SQL generation/self-review and deterministic
        schema/safety validation.

        RetrieverAgent owns execution and asks the LLM whether the ACTUAL
        returned evidence answers the original user question. If the first
        result is semantically insufficient, one fresh SQL query may be
        generated from that result feedback, hard-validated, and executed.

        Maximum database executions: two.
        """
        schema_context = (
            state.schema_context.strip() or self._full_schema_catalog.strip()
        )

        if not schema_context:
            state.database_query_succeeded = False
            state.database_error = "DirectQuery requires authoritative schema context."
            state.database_evidence = ""
            return

        print(
            "[RetrieverAgent] [DirectQuery] "
            "Delegating SQL planning to DirectQueryPlanner..."
        )

        try:
            direct_plan = self._direct_query_planner.plan(
                question=query,
                schema_context=schema_context,
            )
        except Exception as exc:
            state.database_query_succeeded = False
            state.database_error = (
                "DirectQueryPlanner failed unexpectedly: "
                f"{type(exc).__name__}: {exc}"
            )
            state.database_evidence = ""

            print(
                "[RetrieverAgent] [DirectQuery] Planner exception: "
                f"{state.database_error}"
            )
            return

        self._record_direct_query_plan(
            state=state,
            plan=direct_plan,
        )

        if not direct_plan.approved:
            state.database_query_succeeded = False
            state.database_error = (
                direct_plan.error
                or "DirectQueryPlanner did not approve SQL for database execution."
            )
            state.database_evidence = ""
            state.database_rows = []
            state.database_columns = []
            state.database_row_count = 0

            print(
                "[RetrieverAgent] [DirectQuery] "
                "SQL not approved; database execution blocked."
            )
            print("[RetrieverAgent] [DirectQuery] Reason: " f"{state.database_error}")
            return

        current_sql = direct_plan.sql
        current_validation = direct_plan.validation

        print(
            "[RetrieverAgent] [DirectQuery] " "Planner approval PASS; SQL may execute."
        )

        for execution_number in range(
            1,
            MAX_DIRECT_QUERY_EXECUTIONS + 1,
        ):
            success, result_or_error = self._execute_direct_sql(current_sql)

            if not success:
                state.database_query_succeeded = False
                state.database_error = result_or_error
                state.database_evidence = ""
                state.database_rows = []
                state.database_columns = []
                state.database_row_count = 0

                print(
                    "[RetrieverAgent] [DirectQuery/MCP] " "Database execution failed."
                )
                print("[RetrieverAgent] [DirectQuery/MCP] Error: " f"{result_or_error}")
                return

            review = self._review_direct_query_result(
                query=query,
                sql=current_sql,
                raw_result=result_or_error,
                validated_tables=(
                    list(current_validation.tables)
                    if current_validation is not None
                    else []
                ),
                validated_columns=(
                    list(current_validation.columns)
                    if current_validation is not None
                    else []
                ),
                validated_relationships=(
                    list(current_validation.relationships)
                    if current_validation is not None
                    else []
                ),
            )

            print(
                "[RetrieverAgent] [DirectQuery/Review] "
                f"Execution {execution_number}/{MAX_DIRECT_QUERY_EXECUTIONS}"
            )
            print(
                "[RetrieverAgent] [DirectQuery/Review] "
                f"Sufficient : {review.sufficient}"
            )
            print(
                "[RetrieverAgent] [DirectQuery/Review] "
                f"Confidence : {review.confidence:.2f}"
            )
            print(
                "[RetrieverAgent] [DirectQuery/Review] " f"Reason     : {review.reason}"
            )

            if review.sufficient:
                self._store_successful_database_result(
                    state,
                    result_or_error,
                )
                state.database_error = None
                return

            if execution_number >= MAX_DIRECT_QUERY_EXECUTIONS:
                state.database_query_succeeded = False
                state.database_error = (
                    "DirectQuery executed successfully, but the returned "
                    "evidence was judged insufficient after the bounded "
                    f"re-query: {review.reason}"
                )
                state.database_evidence = ""
                state.database_rows = []
                state.database_columns = []
                state.database_row_count = 0

                print(
                    "[RetrieverAgent] [DirectQuery/Review] "
                    "Evidence still insufficient; retry budget exhausted."
                )
                return

            print(
                "[RetrieverAgent] [DirectQuery/Review] "
                "Generating one result-driven re-query..."
            )

            retry_sql = self._generate_result_driven_retry_sql(
                query=query,
                schema_context=schema_context,
                prior_sql=current_sql,
                raw_result=result_or_error,
                reason=review.reason,
            )

            print("[RetrieverAgent] [DirectQuery/Review] Retry SQL candidate:")
            print(retry_sql)

            validation = self._direct_sql_validator.validate(retry_sql)

            if not validation.is_valid:
                state.database_query_succeeded = False
                state.database_error = (
                    "Result-driven DirectQuery retry failed hard validation: "
                    + "; ".join(
                        validation.errors or ["No validation reason was supplied."]
                    )
                )
                state.database_evidence = ""
                state.database_rows = []
                state.database_columns = []
                state.database_row_count = 0

                print(
                    "[RetrieverAgent] [DirectQuery/Review] "
                    "Retry SQL blocked by hard validation."
                )

                for error in validation.errors:
                    print("[RetrieverAgent] [DirectQuery/Review]   - " f"{error}")

                return

            retry_plan = SQLPlan(
                purpose=query,
                database_type="mysql",
                tables=list(validation.tables),
                columns=list(validation.columns),
                relationships=list(validation.relationships),
                filters=[],
                sql=retry_sql,
                is_valid=True,
                validation_errors=[],
            )

            self._record_plan(
                state,
                retry_plan,
            )

            state.schema_tables = list(validation.tables)
            state.schema_relationships = list(validation.relationships)

            current_sql = retry_sql
            current_validation = validation

            print(
                "[RetrieverAgent] [DirectQuery/Review] "
                "Retry SQL passed hard validation; executing bounded second query."
            )

    # ------------------------------------------------------------------------
    # SQL planning and execution
    # ------------------------------------------------------------------------

    @staticmethod
    def _record_plan(
        state: WorkflowState,
        plan: SQLPlan,
    ) -> None:
        state.sql_plan = plan
        state.sql_query = plan.sql

        state.sql_validation_errors = list(plan.validation_errors)

        if plan.is_valid:
            state.schema_tables = list(plan.tables)

            state.schema_relationships = list(plan.relationships)

    def _explain_plan(
        self,
        plan: SQLPlan,
    ) -> tuple[bool, str]:
        """
        Ask MySQL to parse/resolve/plan a safe read-only candidate without returning
        the candidate's result set.

        The exact database/MCP error is intentionally preserved so SQLPlanner's LLM
        repair turn can reason from the database-native observation.
        """
        sql = _clean_text(getattr(plan, "sql", ""))

        if not sql:
            return False, "MySQL EXPLAIN validation received empty SQL."

        explain_sql = f"EXPLAIN {sql.rstrip().rstrip(';')}"

        print("[RetrieverAgent] [MCP/EXPLAIN] Validating candidate with MySQL...")
        print("[RetrieverAgent] [MCP/EXPLAIN] SQL:")
        print(explain_sql)

        try:
            result = _clean_text(mysql_query_tool(explain_sql))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if _is_infrastructure_database_error(error):
                print(
                    "[RetrieverAgent] [MCP/EXPLAIN] Validation could not be completed."
                )
                print(
                    "[RetrieverAgent] [MCP/EXPLAIN] Infrastructure error; SQL validity remains UNKNOWN."
                )
                print(f"[RetrieverAgent] [MCP/EXPLAIN] Error: {error}")
                return False, f"INFRASTRUCTURE_ERROR: {error}"

            print("[RetrieverAgent] [MCP/EXPLAIN] MySQL rejected the candidate.")
            print(f"[RetrieverAgent] [MCP/EXPLAIN] Error: {error}")
            return False, error

        if _result_looks_like_error(result):
            if _is_infrastructure_database_error(result):
                print(
                    "[RetrieverAgent] [MCP/EXPLAIN] Validation could not be completed."
                )
                print(
                    "[RetrieverAgent] [MCP/EXPLAIN] Infrastructure error; SQL validity remains UNKNOWN."
                )
                print(f"[RetrieverAgent] [MCP/EXPLAIN] Error: {result}")
                return False, f"INFRASTRUCTURE_ERROR: {result}"

            print("[RetrieverAgent] [MCP/EXPLAIN] MySQL rejected the candidate.")
            print(f"[RetrieverAgent] [MCP/EXPLAIN] Error: {result}")
            return False, result

        print("[RetrieverAgent] [MCP/EXPLAIN] PASS — MySQL accepted the candidate.")
        return True, result

    def _execute_plan(
        self,
        plan: SQLPlan,
    ) -> tuple[
        bool,
        str,
    ]:
        if not plan.is_valid:
            return (
                False,
                (
                    "SQL planning did not produce a valid query: "
                    + "; ".join(
                        plan.validation_errors or ["No validation reason was supplied."]
                    )
                ),
            )

        if not plan.sql.strip():
            return (
                False,
                ("The SQL plan was marked valid but contained " "no SQL."),
            )

        print("[RetrieverAgent] [MCP] Executing validated SQL plan...")
        print("[RetrieverAgent] [MCP] SQL:")
        print(plan.sql)

        try:
            result = _clean_text(mysql_query_tool(plan.sql))

        except Exception as exc:
            return (
                False,
                f"{type(exc).__name__}: {exc}",
            )

        if _result_looks_like_error(result):
            return (
                False,
                result,
            )

        return (
            True,
            result,
        )

    def _store_successful_database_result(
        self,
        state: WorkflowState,
        raw_result: str,
    ) -> None:
        (
            normalized_evidence,
            database_rows,
        ) = _normalize_mysql_result(raw_result)

        state.database_query_succeeded = True
        state.database_error = None
        state.database_source = "mysql"
        state.database_evidence = normalized_evidence
        state.database_rows = database_rows
        state.database_row_count = len(database_rows)

        state.database_columns = list(database_rows[0].keys()) if database_rows else []

        print("[RetrieverAgent] [MCP] Database execution " "completed successfully.")
        print("[RetrieverAgent] [MCP] Rows returned: " f"{state.database_row_count}")
        print(
            "[RetrieverAgent] [MCP] Evidence length: "
            f"{len(normalized_evidence)} chars"
        )

    def _plan_repair_execute(
        self,
        *,
        query: str,
        schema_context: str,
        state: WorkflowState,
        allowed_tables: list[str] | None = None,
        allowed_relationship_paths: list[str] | None = None,
        ledger_context: str = "",
    ) -> None:
        """
        Plan -> Python safety/preflight -> MySQL EXPLAIN -> execute.

        Python may identify structural/schema concerns, but safe read-only candidates
        that MySQL can adjudicate are allowed to reach EXPLAIN. The exact MySQL
        validation/execution error is then supplied to SQLPlanner repair mode.
        """
        print(
            "[RetrieverAgent] [SQLPlanner] "
            "Requesting schema-grounded SQL with bounded repair..."
        )

        self._trace_graph_constraints(
            allowed_tables=allowed_tables,
            allowed_relationship_paths=allowed_relationship_paths,
        )

        print(
            "[RetrieverAgent] [SQLPlanner] Upstream table/path selections are "
            "NOT forwarded to the SQL LLM."
        )
        print(
            "[RetrieverAgent] [SQLPlanner] SQLPlanner selects physical evidence "
            "sources from the complete authoritative schema."
        )

        planner_kwargs = self._sql_planner_graph_kwargs(
            self._sql_planner.plan,
            allowed_tables=allowed_tables,
            allowed_relationship_paths=allowed_relationship_paths,
        )

        current_plan = self._sql_planner.plan(
            query=query,
            schema_context=schema_context,
            ledger_context=ledger_context,
            **planner_kwargs,
        )

        self._record_plan(state, current_plan)

        if not current_plan.is_valid:
            state.database_query_succeeded = False
            state.database_error = (
                "SQLPlanner safety/preflight validation blocked the SQL candidate: "
                + "; ".join(
                    current_plan.validation_errors
                    or ["No validation reason was supplied."]
                )
            )
            state.database_evidence = ""

            print(
                "[RetrieverAgent] [SQLPlanner] "
                "Candidate blocked before MySQL because it failed safety/preflight."
            )
            return

        repair_kwargs = self._sql_planner_graph_kwargs(
            self._sql_planner.repair,
            allowed_tables=allowed_tables,
            allowed_relationship_paths=allowed_relationship_paths,
        )

        # Every investigation candidate is checked by MySQL EXPLAIN. This includes
        # candidates that Python considered fully valid as well as provisional
        # candidates whose structural/schema concerns were intentionally deferred.
        for explain_cycle in range(1, MAX_MYSQL_EXPLAIN_REPAIRS + 2):
            if requires_mysql_explain(current_plan):
                python_notes = [
                    item
                    for item in (current_plan.validation_errors or [])
                    if item != "MYSQL_EXPLAIN_REQUIRED"
                ]
                if python_notes:
                    print(
                        "[RetrieverAgent] [SQLPlanner] Python preflight notes are "
                        "advisory until MySQL EXPLAIN adjudicates the SQL:"
                    )
                    for note in python_notes:
                        print(f"[RetrieverAgent] [SQLPlanner]   - {note}")

            explain_ok, explain_result = self._explain_plan(current_plan)

            if explain_ok:
                # MySQL has now established executability. Clear provisional
                # preflight notes before actual execution/recording.
                current_plan = current_plan.updated(
                    is_valid=True,
                    validation_errors=[],
                )

                print(
                    "[RetrieverAgent] [MCP] MySQL EXPLAIN accepted the SQL; "
                    "executing the actual read-only query."
                )

                success, result_or_error = self._execute_plan(current_plan)

                if success:
                    self._record_plan(state, current_plan)
                    self._store_successful_database_result(
                        state,
                        result_or_error,
                    )
                    return

                # EXPLAIN cannot expose every runtime property (for example a scalar
                # subquery that returns multiple rows). Preserve the exact execution
                # error and let the LLM repair the already accepted SQL.
                database_error = result_or_error

                if _is_infrastructure_database_error(database_error):
                    state.database_query_succeeded = False
                    state.database_error = f"INFRASTRUCTURE_ERROR: {database_error}"
                    state.database_evidence = ""
                    print(
                        "[RetrieverAgent] [MCP] Database execution could not complete."
                    )
                    print(
                        "[RetrieverAgent] [MCP] Infrastructure error; preserving the SQL "
                        "unchanged and skipping LLM repair."
                    )
                    print(f"[RetrieverAgent] [MCP] Error: {database_error}")
                    return

                error_kind = "MYSQL EXECUTION ERROR"
                print("[RetrieverAgent] [MCP] Database execution failed.")
                print(f"[RetrieverAgent] [MCP] Error: {database_error}")
            else:
                database_error = explain_result

                if database_error.startswith("INFRASTRUCTURE_ERROR:"):
                    state.database_query_succeeded = False
                    state.database_error = database_error
                    state.database_evidence = ""
                    print(
                        "[RetrieverAgent] [SQLPlanner] Infrastructure failure prevented "
                        "MySQL validation. Preserving the SQL unchanged; no LLM repair "
                        "will be attempted."
                    )
                    return

                error_kind = "MYSQL EXPLAIN ERROR"

            if explain_cycle >= MAX_MYSQL_EXPLAIN_REPAIRS + 1:
                state.database_query_succeeded = False
                state.database_error = f"{error_kind}: {database_error}"
                state.database_evidence = ""
                print(
                    "[RetrieverAgent] [SQLPlanner] "
                    "MySQL-native repair budget exhausted."
                )
                return

            exact_feedback = (
                f"{error_kind} (authoritative database observation): "
                f"{database_error}"
            )

            print(
                "[RetrieverAgent] [SQLPlanner] "
                "Sending the exact MySQL error to bounded LLM repair mode..."
            )

            repaired_plan = self._sql_planner.repair(
                query=query,
                schema_context=schema_context,
                prior_plan=current_plan,
                errors=[exact_feedback],
                ledger_context=ledger_context,
                **repair_kwargs,
            )

            self._record_plan(state, repaired_plan)

            if not repaired_plan.is_valid:
                state.database_query_succeeded = False
                state.database_error = (
                    "MySQL-error repair did not produce a safe SQL candidate: "
                    + "; ".join(
                        repaired_plan.validation_errors
                        or ["No validation reason was supplied."]
                    )
                )
                state.database_evidence = ""

                print(
                    "[RetrieverAgent] [SQLPlanner] "
                    "MySQL-error repair remained blocked by safety/preflight."
                )
                return

            current_plan = repaired_plan
            print(
                "[RetrieverAgent] [SQLPlanner] "
                "Repaired SQL will be re-checked by MySQL EXPLAIN."
            )

    def _review_investigation_evidence_sufficiency(
        self,
        *,
        state: WorkflowState,
        original_query: str,
    ) -> InvestigationEvidenceSufficiencyReview:
        """
        Ask the LLM whether successful evidence already answers the original
        investigation question.

        This is the investigation stopping rule. The ledger is supplied as
        memory, not as a mandatory checklist.
        """
        successful = [
            result
            for result in state.investigation_results
            if result.database_query_succeeded
        ]

        if not successful:
            return InvestigationEvidenceSufficiencyReview(
                sufficient=False,
                confidence=0.0,
                reason="No successfully executed investigation evidence exists yet.",
                missing_evidence=["verified database evidence"],
            )

        sections: list[str] = []

        # Keep the stopping review bounded while preserving the most recent and
        # most directly useful evidence. Local models do better with a focused
        # evidence window than with an ever-growing transcript.
        for result in successful[-8:]:
            evidence = _clean_text(result.database_evidence)

            if len(evidence) > 6000:
                evidence = evidence[:6000] + "\n...[evidence truncated for stop review]"

            sections.append(
                "\n".join(
                    [
                        f"Task: {result.title or result.task_id}",
                        f"Question: {result.question}",
                        f"Executed SQL: {result.sql_query or '(not recorded)'}",
                        f"Rows returned: {result.database_row_count}",
                        f"Columns: {result.database_columns}",
                        "Evidence:",
                        evidence or "(successful query returned zero rows)",
                    ]
                )
            )

        plan_data = getattr(state, "investigation_plan", {}) or {}
        planned_tasks = (
            plan_data.get("tasks", []) if isinstance(plan_data, dict) else []
        )
        successful_ids = {str(result.task_id) for result in successful}

        remaining_descriptions: list[str] = []
        for item in planned_tasks:
            if not isinstance(item, dict):
                continue
            task_id = str(item.get("task_id", "")).strip()
            if task_id and task_id in successful_ids:
                continue
            title = _clean_text(item.get("title"))
            question = _clean_text(item.get("question"))
            label = title or task_id or "planned task"
            remaining_descriptions.append(
                f"- {label}: {question}" if question else f"- {label}"
            )

        prompt = _INVESTIGATION_EVIDENCE_STOP_PROMPT.format(
            query=original_query,
            ledger_context=state.investigation_ledger.compact_summary(),
            planned_task_count=len(planned_tasks),
            successful_task_count=len(successful),
            remaining_task_count=len(remaining_descriptions),
            remaining_tasks=(
                "\n".join(remaining_descriptions[:16])
                if remaining_descriptions
                else "(none)"
            ),
            evidence="\n\n".join(sections),
        )

        try:
            parsed = _extract_structured_model(
                _complete_with_system(
                    self._investigation_sufficiency_llm,
                    system_prompt=EVIDENCE_REVIEW_SYSTEM_PROMPT,
                    user_prompt=prompt,
                ),
                InvestigationEvidenceSufficiencyReview,
            )
            assert isinstance(parsed, InvestigationEvidenceSufficiencyReview)
            return parsed
        except Exception as exc:
            # A failed stop review must not falsely terminate retrieval.
            return InvestigationEvidenceSufficiencyReview(
                sufficient=False,
                confidence=0.0,
                reason=(
                    "Investigation evidence sufficiency review failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                missing_evidence=[],
            )

    def _build_existing_investigation_evidence_summary(
        self,
        state: WorkflowState,
    ) -> str:
        """Build focused successful-evidence context for the next LLM plan."""
        sections: list[str] = []

        for result in state.investigation_results[-8:]:
            if not result.database_query_succeeded:
                continue

            evidence = _clean_text(result.database_evidence)
            if len(evidence) > 2500:
                evidence = evidence[:2500] + "\n...[evidence preview truncated]"

            sections.append(
                "\n".join(
                    [
                        f"Task: {result.title or result.task_id}",
                        f"Question: {result.question}",
                        f"Executed SQL: {result.sql_query or '(not recorded)'}",
                        f"Rows: {result.database_row_count}",
                        f"Columns: {result.database_columns}",
                        "Evidence preview:",
                        evidence or "(successful query returned zero rows)",
                    ]
                )
            )

        return (
            "\n\n".join(sections) if sections else "(no successful prior task metadata)"
        )

    @staticmethod
    def _missing_evidence_request_from_review(
        review: InvestigationEvidenceSufficiencyReview,
    ) -> str:
        """
        Convert the LLM sufficiency judgment into a POSITIVE factual evidence
        objective for the next planning pass.

        missing_evidence is authoritative for WHAT must be learned next.
        review.reason is diagnostic context only and must never become the next
        investigation question.
        """
        missing = [
            _clean_text(item) for item in review.missing_evidence if _clean_text(item)
        ]

        lines = [
            "NEXT FACTUAL EVIDENCE OBJECTIVE",
            "-------------------------------",
        ]

        if missing:
            lines.append(
                "Retrieve or establish the following concrete real-world fact(s):"
            )
            lines.extend(f"- {item}" for item in missing)
        else:
            # Fall back to the review reason only as context. Explicit wording
            # prevents the planner from turning an explanation into a meta-task.
            lines.append(
                "Determine the concrete factual evidence still needed to resolve "
                "the original user question."
            )

        reason = _clean_text(review.reason)
        if reason:
            lines.extend(
                [
                    "",
                    "DIAGNOSTIC CONTEXT FROM THE PRIOR EVIDENCE REVIEW",
                    "-------------------------------------------------",
                    reason,
                    (
                        "This diagnostic explanation is CONTEXT ONLY. Do not create "
                        "a task asking why the prior evidence was missing, why a "
                        "query failed, or why a field was absent."
                    ),
                ]
            )

        lines.extend(
            [
                "",
                "PLANNING REQUIREMENT",
                "--------------------",
                (
                    "Plan a NEW evidence task only if refining the already "
                    "successful query/evidence cannot establish the missing "
                    "operational fact. Ask what genuinely new records or domains "
                    "are required."
                ),
                (
                    "Do not repeat a prior successful query unless the new task "
                    "materially changes what is observed."
                ),
                (
                    "Do not ask a meta-question such as 'Why is the clock-in status "
                    "missing?' or 'Why did the earlier query fail?'."
                ),
            ]
        )

        return "\n".join(lines).strip()

    def _refine_successful_investigation_sql(
        self,
        *,
        original_query: str,
        state: WorkflowState,
        prior_result: InvestigationTaskResult,
        review: InvestigationEvidenceSufficiencyReview,
        refinement_number: int,
        refinement_feedback: list[str] | None = None,
    ) -> tuple[InvestigationTaskResult | None, str | None]:
        """Refine a successful investigation SQL query before opening a new domain."""
        prior_plan = SQLPlan(
            purpose=prior_result.question,
            tables=list(prior_result.required_tables),
            columns=list(prior_result.database_columns),
            sql=prior_result.sql_query,
            is_valid=True,
        )

        evidence_preview = _clean_text(prior_result.database_evidence)
        if len(evidence_preview) > 6000:
            evidence_preview = (
                evidence_preview[:6000]
                + "\n...[evidence preview truncated for SQL refinement]"
            )

        semantic_gap_parts = [_clean_text(review.reason)]
        semantic_gap_parts.extend(
            _clean_text(item) for item in review.missing_evidence if _clean_text(item)
        )
        semantic_gap = " | ".join(part for part in semantic_gap_parts if part)

        refined_plan = self._sql_planner.refine_from_evidence(
            original_query=original_query,
            schema_context=self._full_schema_catalog,
            prior_plan=prior_plan,
            row_count=prior_result.database_row_count,
            columns=list(prior_result.database_columns),
            evidence_preview=evidence_preview,
            semantic_gap=semantic_gap,
            ledger_context=state.investigation_ledger.compact_summary(),
            refinement_feedback=refinement_feedback,
        )

        if not refined_plan.is_valid:
            print(
                "[RetrieverAgent] [Investigation] SQL refinement did not produce "
                "a hard-valid query; falling back to evidence planning."
            )
            return None, "SQL refinement did not produce a hard-valid query."

        if (
            " ".join(refined_plan.sql.split()).lower()
            == " ".join(prior_result.sql_query.split()).lower()
        ):
            message = (
                "The refinement repeated the previously executed SQL unchanged. "
                "That SQL has already been observed and did not establish the "
                f"remaining semantic gap: {semantic_gap}. Produce a materially "
                "different investigative query on the next refinement."
            )
            print(
                "[RetrieverAgent] [Investigation] SQL refinement repeated the "
                "same successful SQL; feeding that repetition back to SQLPlanner."
            )
            return None, message

        print(
            "[RetrieverAgent] [Investigation] ↻ Executing refined successful "
            f"evidence query #{refinement_number}."
        )

        success, raw_result = self._execute_plan(refined_plan)
        if not success:
            print(
                "[RetrieverAgent] [Investigation] Refined SQL execution failed: "
                f"{raw_result}"
            )
            return None, (
                "The refined SQL was hard-valid but execution failed. "
                f"Execution result: {raw_result}"
            )

        temp_state = WorkflowState(query=original_query)
        self._record_plan(temp_state, refined_plan)
        self._store_successful_database_result(temp_state, raw_result)

        refined_result = InvestigationTaskResult(
            task_id=f"{prior_result.task_id}_refine_{refinement_number}",
            title=f"{prior_result.title} — evidence refinement {refinement_number}",
            question=original_query,
            sql_plan_valid=True,
            sql_query=refined_plan.sql,
            database_query_succeeded=True,
            database_error=None,
            database_evidence=temp_state.database_evidence,
            database_rows=temp_state.database_rows,
            database_columns=temp_state.database_columns,
            database_row_count=temp_state.database_row_count,
            required_tables=list(refined_plan.tables),
            requested_outputs=list(prior_result.requested_outputs),
        )

        state.investigation_results.append(refined_result)
        state.investigation_ledger.record_physical_result(
            task_id=refined_result.task_id,
            tables=list(refined_plan.tables),
            fields=list(refined_result.database_columns),
            row_count=refined_result.database_row_count,
        )

        return refined_result, None

    # ------------------------------------------------------------------------
    # Investigation
    # ------------------------------------------------------------------------

    def _execute_investigation(
        self,
        *,
        query: str,
        state: WorkflowState,
        append_results: bool = False,
        supplemental_label: str = "",
        preplanned_plan: Any | None = None,
        internal_pass: int = 0,
        internal_seen_requests: set[str] | None = None,
        internal_seen_refinements: set[str] | None = None,
    ) -> None:
        """
        Decompose and execute a broad or targeted investigation.

        Design rules:
        - Agent 1's investigation contract is authoritative for entering this path.
        - InvestigationPlanner proposes incremental factual evidence tasks.
        - Python validates schema/safety per task but does not require a perfect
          whole-investigation plan before execution.
        - Every SQLPlanner task receives the complete authoritative schema plus
          only the LLM-selected/SchemaGraph-approved task scope.
        - Each task is isolated so one failure does not abort the investigation.
        - Every successful task updates the persistent evidence ledger.
        - After every successful task, the LLM decides whether the ORIGINAL
          question is answerable from all verified evidence gathered so far.
        - If not, Agent 2 first gives SQLPlanner a bounded opportunity to refine
          the most recent SUCCESSFUL query using actual returned evidence and the
          LLM's semantic shortcoming.
        - Only when refinement cannot resolve the gap does Agent 2 ask
          InvestigationPlanner to open another evidence task/domain.
        - Agent 3 remains the final security interpreter after Agent 2 has either
          reached evidence sufficiency or exhausted the bounded internal loop.
        """
        state.investigation_requested = True

        if internal_seen_requests is None:
            internal_seen_requests = set()

        if internal_seen_refinements is None:
            internal_seen_refinements = set()

        last_stop_review: InvestigationEvidenceSufficiencyReview | None = None

        full_schema = self._full_schema_catalog

        ledger_summary = state.investigation_ledger.compact_summary()

        print(
            "[RetrieverAgent] [Investigation] Persistent ledger supplied to " "planner:"
        )
        for ledger_line in ledger_summary.splitlines():
            print("[RetrieverAgent] [Investigation]   " + ledger_line)

        investigation_plan = (
            preplanned_plan
            if preplanned_plan is not None
            else self._investigation_planner.plan(
                query=query,
                schema_context=full_schema,
                policy_context=state.policy_context,
                force_investigation=True,
                ledger_context=ledger_summary,
            )
        )

        if not append_results:
            state.investigation_plan = investigation_plan.model_dump()
            state.investigation_results = []
        else:
            state.add_trace(
                "RetrieverAgent",
                (
                    "Executing one bounded supplemental investigation plan"
                    + (f": {supplemental_label}" if supplemental_label else ".")
                ),
            )

        existing_result_count = len(state.investigation_results)

        print(
            "[RetrieverAgent] [Investigation] Tasks planned: "
            f"{len(investigation_plan.tasks)}"
        )

        if investigation_plan.tasks:
            planned_tables = sorted(
                {
                    table_name
                    for task in investigation_plan.tasks
                    for table_name in task.required_tables
                }
            )
            print(
                "[RetrieverAgent] [Investigation] Whole-plan table coverage: "
                f"{planned_tables}"
            )

        if investigation_plan.assumptions:
            print("[RetrieverAgent] [Investigation] Assumptions:")
            for assumption in investigation_plan.assumptions:
                print("[RetrieverAgent] [Investigation]   - " f"{assumption}")

        if investigation_plan.tasks and investigation_plan.missing_business_rules:
            print(
                "[RetrieverAgent] [Investigation] Executing valid tasks even "
                "though the planner reports remaining coverage/validation notes."
            )

        if investigation_plan.missing_business_rules:
            print(
                "[RetrieverAgent] [Investigation] " "Planning limitations/rejections:"
            )
            for limitation in investigation_plan.missing_business_rules:
                print("[RetrieverAgent] [Investigation]   - " f"{limitation}")

        if not investigation_plan.tasks:
            state.investigation_succeeded = False
            state.database_query_succeeded = False
            state.database_evidence = ""
            state.database_rows = []
            state.database_columns = []
            state.database_row_count = 0
            limitation_text = "; ".join(investigation_plan.missing_business_rules)

            state.database_error = (
                "The investigation planner did not produce any executable "
                "analysis tasks."
                + (
                    f" Planner limitations: {limitation_text}"
                    if limitation_text
                    else ""
                )
            )

            print(
                "[RetrieverAgent] [Investigation] No tasks produced. "
                f"{state.database_error}"
            )
            return

        evidence_sections: list[str] = []

        if append_results and state.database_evidence.strip():
            evidence_sections.append(state.database_evidence.strip())

        for task_number, task in enumerate(
            investigation_plan.tasks,
            start=existing_result_count + 1,
        ):
            print(
                "[RetrieverAgent] [Investigation] " f"Task {task_number}: {task.title}"
            )

            if task.required_schema_documents:
                print(
                    "[RetrieverAgent] [Investigation] "
                    "Planner schema refs: "
                    f"{task.required_schema_documents}"
                )

            task_allowed_tables = list(getattr(task, "required_tables", []) or [])
            task_relationship_paths = list(
                getattr(task, "relationship_paths", []) or []
            )

            print(
                "[RetrieverAgent] [Investigation] "
                "InvestigationPlanner tables (metadata only): "
                f"{task_allowed_tables or '(none)'}"
            )

            if task_relationship_paths:
                print(
                    "[RetrieverAgent] [Investigation] "
                    "InvestigationPlanner path(s) (metadata only):"
                )
                for relationship_path in task_relationship_paths:
                    print(
                        "[RetrieverAgent] [Investigation]   - " f"{relationship_path}"
                    )

            schema_context = full_schema

            print(
                "[RetrieverAgent] [Investigation] "
                "SQL schema grounding: complete catalog "
                f"({len(schema_context)} chars)"
            )

            task_state = WorkflowState(query=task.question)
            task_state.required_evidence_types = [
                EvidenceType.SCHEMA.value,
                EvidenceType.DATABASE.value,
            ]
            task_state.requires_live_data = True
            task_state.schema_context = schema_context
            task_state.schema_evidence = schema_context
            task_state.schema_query_succeeded = bool(schema_context.strip())

            try:
                self._plan_repair_execute(
                    query=task.question,
                    schema_context=schema_context,
                    state=task_state,
                    allowed_tables=task_allowed_tables,
                    allowed_relationship_paths=task_relationship_paths,
                    ledger_context=state.investigation_ledger.compact_summary(),
                )

            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                task_state.database_query_succeeded = False
                task_state.database_error = error_text
                task_state.database_evidence = ""
                task_state.database_rows = []
                task_state.database_columns = []
                task_state.database_row_count = 0

                print(
                    "[RetrieverAgent] [Investigation] "
                    f"Task {task_number} failed but the "
                    f"investigation will continue: {error_text}"
                )

            actual_tables = (
                list(task_state.sql_plan.tables)
                if task_state.sql_plan is not None and task_state.sql_plan.is_valid
                else list(task_allowed_tables)
            )

            actual_relationships = (
                list(task_state.sql_plan.relationships)
                if task_state.sql_plan is not None and task_state.sql_plan.is_valid
                else list(task_relationship_paths)
            )

            if actual_tables != task_allowed_tables:
                print(
                    "[RetrieverAgent] [Investigation] SQLPlanner expanded "
                    "beyond planner suggestions."
                )
                print(
                    "[RetrieverAgent] [Investigation]   Suggested tables: "
                    f"{task_allowed_tables}"
                )
                print(
                    "[RetrieverAgent] [Investigation]   Validated tables: "
                    f"{actual_tables}"
                )

            result = InvestigationTaskResult(
                task_id=task.task_id,
                title=task.title,
                question=task.question,
                sql_plan_valid=bool(
                    task_state.sql_plan and task_state.sql_plan.is_valid
                ),
                sql_query=task_state.sql_query,
                database_query_succeeded=(task_state.database_query_succeeded),
                database_error=task_state.database_error,
                database_evidence=task_state.database_evidence,
                database_rows=task_state.database_rows,
                database_columns=task_state.database_columns,
                database_row_count=task_state.database_row_count,
                required_tables=actual_tables,
                requested_outputs=list(getattr(task, "requested_outputs", []) or []),
            )

            state.investigation_results.append(result)

            if result.database_query_succeeded:
                state.investigation_ledger.record_physical_result(
                    task_id=result.task_id,
                    tables=actual_tables,
                    fields=result.database_columns,
                    row_count=result.database_row_count,
                )

            if result.database_query_succeeded:
                evidence_sections.append(
                    "\n".join(
                        [
                            (
                                "=== INVESTIGATION TASK "
                                f"{task_number}: "
                                f"{task.title} ==="
                            ),
                            f"Question: {task.question}",
                            f"Rationale: {task.rationale}",
                            (
                                "Requested outputs: "
                                + (
                                    ", ".join(task.requested_outputs)
                                    or "(none explicitly requested)"
                                )
                            ),
                            (
                                "Planner schema references: "
                                + (
                                    ", ".join(task.required_schema_documents)
                                    or "(none)"
                                )
                            ),
                            (
                                "InvestigationPlanner tables (metadata only): "
                                + (", ".join(task_allowed_tables) or "(none)")
                            ),
                            (
                                "Planner-suggested paths: "
                                + (" | ".join(task_relationship_paths) or "(none)")
                            ),
                            (
                                "SQLPlanner validated tables: "
                                + (", ".join(actual_tables) or "(none)")
                            ),
                            (
                                "SQLPlanner validated relationships: "
                                + (
                                    " | ".join(actual_relationships)
                                    or "(none required)"
                                )
                            ),
                            ("SQL plan valid: " f"{result.sql_plan_valid}"),
                            ("Rows returned: " f"{result.database_row_count}"),
                            "Database evidence:",
                            (result.database_evidence or "(no rows returned)"),
                        ]
                    )
                )
                stop_review = self._review_investigation_evidence_sufficiency(
                    state=state,
                    original_query=query,
                )
                last_stop_review = stop_review

                print(
                    "[RetrieverAgent] [Investigation] Evidence sufficiency checkpoint:"
                )
                print(
                    "[RetrieverAgent] [Investigation]   Sufficient : "
                    f"{stop_review.sufficient}"
                )
                print(
                    "[RetrieverAgent] [Investigation]   Confidence : "
                    f"{stop_review.confidence:.2f}"
                )
                print(
                    "[RetrieverAgent] [Investigation]   Reason     : "
                    f"{stop_review.reason}"
                )

                if stop_review.missing_evidence:
                    print(
                        "[RetrieverAgent] [Investigation]   Still needed: "
                        f"{stop_review.missing_evidence}"
                    )

                if stop_review.sufficient:
                    state.add_trace(
                        "RetrieverAgent",
                        (
                            "Investigation retrieval stopped early because the "
                            "LLM judged the verified evidence sufficient to "
                            "reason about the original question. "
                            f"Reason={stop_review.reason}"
                        ),
                    )
                    print(
                        "[RetrieverAgent] [Investigation] ✔ STOPPING TASK "
                        "GENERATION/EXECUTION — verified evidence is sufficient."
                    )
                    break

            else:
                print(
                    "[RetrieverAgent] [Investigation] "
                    f"Task {task_number} produced no verified "
                    "database evidence."
                )
                if result.database_error:
                    print(
                        "[RetrieverAgent] [Investigation]   Error: "
                        f"{result.database_error}"
                    )

        # If the task list is exhausted but the LLM says the original
        # question is still not answerable, refine the latest successful SQL
        # before asking InvestigationPlanner to open a new evidence domain.
        if last_stop_review is not None and not last_stop_review.sufficient:
            successful_for_refinement = [
                item
                for item in state.investigation_results
                if item.database_query_succeeded and _clean_text(item.sql_query)
            ]

            refinement_attempt = 0
            refinement_feedback: list[str] = []

            while (
                successful_for_refinement
                and not last_stop_review.sufficient
                and refinement_attempt < MAX_INTERNAL_SQL_REFINEMENTS
            ):
                prior_result = successful_for_refinement[-1]

                refinement_material = "\n".join(
                    [
                        _clean_text(prior_result.sql_query),
                        str(prior_result.database_row_count),
                        ",".join(prior_result.database_columns),
                        _clean_text(last_stop_review.reason),
                        "|".join(
                            _clean_text(item)
                            for item in last_stop_review.missing_evidence
                            if _clean_text(item)
                        ),
                        # Refinement feedback is part of the reasoning state.
                        # A duplicate proposal followed by explicit corrective
                        # feedback is therefore a NEW reasoning turn and must
                        # not be mistaken for an already-seen refinement state.
                        "|".join(
                            _clean_text(item)
                            for item in refinement_feedback
                            if _clean_text(item)
                        ),
                    ]
                )
                refinement_signature = hashlib.sha256(
                    refinement_material.encode("utf-8")
                ).hexdigest()

                if refinement_signature in internal_seen_refinements:
                    print(
                        "[RetrieverAgent] [Investigation] Refinement stopped only "
                        "because the same SQL/evidence/semantic-feedback state was "
                        "already attempted in an earlier investigation pass."
                    )
                    break

                internal_seen_refinements.add(refinement_signature)
                refinement_attempt += 1

                print(
                    "[RetrieverAgent] [Investigation] ↻ Missing fact may be "
                    "obtainable by refining the latest successful SQL before "
                    "opening another evidence domain."
                )

                refined_result, refinement_issue = (
                    self._refine_successful_investigation_sql(
                        original_query=query,
                        state=state,
                        prior_result=prior_result,
                        review=last_stop_review,
                        refinement_number=refinement_attempt,
                        refinement_feedback=refinement_feedback,
                    )
                )

                if refined_result is None:
                    if refinement_issue:
                        refinement_feedback.append(refinement_issue)
                        print(
                            "[RetrieverAgent] [Investigation]   Refinement feedback "
                            f"for next LLM turn: {refinement_issue}"
                        )

                    if refinement_attempt < MAX_INTERNAL_SQL_REFINEMENTS:
                        print(
                            "[RetrieverAgent] [Investigation]   Duplicate/non-terminal "
                            "refinement consumed one turn; another bounded LLM "
                            "refinement turn remains."
                        )
                        continue

                    break

                evidence_sections.append(
                    "\n".join(
                        [
                            (
                                "=== REFINED INVESTIGATION EVIDENCE "
                                f"{refined_result.task_id} ==="
                            ),
                            f"Question: {query}",
                            (
                                "Refinement source: prior successful SQL + actual "
                                "database evidence + LLM semantic gap"
                            ),
                            f"Rows returned: {refined_result.database_row_count}",
                            f"Columns: {refined_result.database_columns}",
                            "Refined SQL:",
                            refined_result.sql_query,
                            "Database evidence:",
                            (refined_result.database_evidence or "(no rows returned)"),
                        ]
                    )
                )

                last_stop_review = self._review_investigation_evidence_sufficiency(
                    state=state,
                    original_query=query,
                )

                print(
                    "[RetrieverAgent] [Investigation] Refined evidence "
                    "sufficiency checkpoint:"
                )
                print(
                    "[RetrieverAgent] [Investigation]   Sufficient : "
                    f"{last_stop_review.sufficient}"
                )
                print(
                    "[RetrieverAgent] [Investigation]   Confidence : "
                    f"{last_stop_review.confidence:.2f}"
                )
                print(
                    "[RetrieverAgent] [Investigation]   Reason     : "
                    f"{last_stop_review.reason}"
                )

                if last_stop_review.missing_evidence:
                    print(
                        "[RetrieverAgent] [Investigation]   Still needed: "
                        f"{last_stop_review.missing_evidence}"
                    )

                if last_stop_review.sufficient:
                    state.add_trace(
                        "RetrieverAgent",
                        (
                            "Investigation became sufficient after refining a "
                            "successful SQL query. "
                            f"Reason={last_stop_review.reason}"
                        ),
                    )
                    break

                refinement_feedback.append(
                    "The last refined SQL executed successfully but the "
                    "post-execution LLM still found the evidence insufficient. "
                    f"Reason: {last_stop_review.reason}. Missing fact(s): "
                    + "; ".join(last_stop_review.missing_evidence)
                )

                successful_for_refinement.append(refined_result)

            if last_stop_review.sufficient:
                print(
                    "[RetrieverAgent] [Investigation] ✔ Refined SQL evidence is "
                    "sufficient; no new evidence domain will be opened."
                )

            else:
                next_request = self._missing_evidence_request_from_review(
                    last_stop_review
                )
                normalized_request = " ".join(next_request.lower().split())

                latest_success = next(
                    (
                        item
                        for item in reversed(state.investigation_results)
                        if item.database_query_succeeded
                    ),
                    None,
                )

                progress_material = "\n".join(
                    [
                        normalized_request,
                        _clean_text(
                            latest_success.sql_query
                            if latest_success is not None
                            else ""
                        ),
                        str(
                            latest_success.database_row_count
                            if latest_success is not None
                            else 0
                        ),
                        ",".join(
                            latest_success.database_columns
                            if latest_success is not None
                            else []
                        ),
                    ]
                )
                request_progress_signature = hashlib.sha256(
                    progress_material.encode("utf-8")
                ).hexdigest()

                if internal_pass >= MAX_INTERNAL_INVESTIGATION_PASSES:
                    message = (
                        "Internal investigation evidence loop reached its bounded "
                        f"maximum of {MAX_INTERNAL_INVESTIGATION_PASSES} re-plans. "
                        f"Still missing: {last_stop_review.reason}"
                    )
                    state.add_warning(message)
                    state.add_trace("RetrieverAgent", message)
                    print("[RetrieverAgent] [Investigation] ⚠ " + message)

                elif request_progress_signature in internal_seen_requests:
                    message = (
                        "Internal investigation stopped because the LLM requested "
                        "the same missing evidence after materially identical "
                        "successful SQL/evidence. "
                        f"Reason: {last_stop_review.reason}"
                    )
                    state.add_warning(message)
                    state.add_trace("RetrieverAgent", message)
                    print("[RetrieverAgent] [Investigation] ⚠ " + message)

                else:
                    internal_seen_requests.add(request_progress_signature)

                    print(
                        "[RetrieverAgent] [Investigation] ↻ Evidence is still "
                        "insufficient after SQL refinement; asking the LLM to "
                        "plan a genuinely new evidence pass."
                    )
                    print(
                        "[RetrieverAgent] [Investigation]   Next FACTUAL evidence "
                        f"objective: {next_request.replace(chr(10), ' | ')}"
                    )

                    existing_summary = (
                        self._build_existing_investigation_evidence_summary(state)
                    )

                    next_plan = self._investigation_planner.plan_supplemental(
                        original_query=query,
                        supplemental_request=next_request,
                        existing_evidence_summary=existing_summary,
                        schema_context=full_schema,
                        policy_context=state.policy_context,
                        ledger_context=state.investigation_ledger.compact_summary(),
                    )

                    if next_plan.tasks:
                        state.database_evidence = "\n\n".join(evidence_sections).strip()
                        state.add_trace(
                            "RetrieverAgent",
                            (
                                "Continuing investigation internally after SQL "
                                "refinement did not resolve the evidence gap; "
                                f"pass={internal_pass + 1}; "
                                f"reason={last_stop_review.reason}"
                            ),
                        )

                        self._execute_investigation(
                            query=query,
                            state=state,
                            append_results=True,
                            supplemental_label=next_request,
                            preplanned_plan=next_plan,
                            internal_pass=internal_pass + 1,
                            internal_seen_requests=internal_seen_requests,
                            internal_seen_refinements=internal_seen_refinements,
                        )
                        return

                    message = (
                        "The LLM identified missing evidence, but "
                        "InvestigationPlanner could not produce a new executable "
                        "task for it."
                    )
                    if next_plan.missing_business_rules:
                        message += " " + "; ".join(next_plan.missing_business_rules)
                    state.add_warning(message)
                    state.add_trace("RetrieverAgent", message)
                    print("[RetrieverAgent] [Investigation] ⚠ " + message)

        successful_results = [
            result
            for result in state.investigation_results
            if result.database_query_succeeded
        ]

        failed_results = [
            result
            for result in state.investigation_results
            if not result.database_query_succeeded
        ]

        if successful_results and failed_results:
            print(
                "[RetrieverAgent] [Investigation] Partial progress preserved: "
                f"{len(successful_results)} successful task(s), "
                f"{len(failed_results)} failed task(s). "
                "Successful evidence will still be graded."
            )

        failed_results = [
            result
            for result in state.investigation_results
            if not result.database_query_succeeded
        ]

        state.investigation_succeeded = bool(successful_results)
        state.database_query_succeeded = bool(successful_results)
        state.database_evidence = "\n\n".join(evidence_sections).strip()

        aggregate_rows: list[dict[str, Any]] = []
        for result in successful_results:
            aggregate_rows.extend(result.database_rows)

        state.database_rows = aggregate_rows

        state.database_row_count = sum(
            result.database_row_count for result in successful_results
        )

        database_columns: list[str] = []
        for result in successful_results:
            for column in result.database_columns:
                if column not in database_columns:
                    database_columns.append(column)

        state.database_columns = database_columns
        state.database_source = "mysql" if successful_results else ""

        print(
            "[RetrieverAgent] [Investigation] "
            f"Successful tasks: {len(successful_results)}/"
            f"{len(state.investigation_results)}"
        )

        if successful_results:
            state.database_error = None

            if failed_results:
                failure_messages = [
                    (
                        f"{result.task_id}: "
                        f"{result.database_error or 'no verified evidence'}"
                    )
                    for result in failed_results
                ]
                state.add_warning(
                    (
                        "Some investigation tasks did not produce "
                        "verified database evidence: " + "; ".join(failure_messages)
                    )
                )
        else:
            errors = [
                result.database_error
                for result in state.investigation_results
                if result.database_error
            ]
            state.database_error = (
                "No investigation task produced verified evidence."
                + (" " + "; ".join(errors) if errors else "")
            )

    @staticmethod
    def _successful_investigation_evidence_signatures(
        state: WorkflowState,
    ) -> set[str]:
        """
        Return stable signatures for successfully retrieved investigation evidence.

        A supplemental pass counts as progress only when it adds at least one
        genuinely new successful evidence result.
        """
        signatures: set[str] = set()

        for result in state.investigation_results:
            if not result.database_query_succeeded:
                continue

            material = "\n".join(
                [
                    _clean_text(result.question).lower(),
                    "|".join(
                        sorted(
                            _clean_text(column).lower()
                            for column in result.database_columns
                            if _clean_text(column)
                        )
                    ),
                    str(result.database_row_count),
                    _clean_text(result.database_evidence),
                ]
            )

            signatures.add(hashlib.sha256(material.encode("utf-8")).hexdigest())

        return signatures

    async def run_supplemental_investigation(
        self,
        state: WorkflowState,
        supplemental_request: str,
    ) -> tuple[WorkflowState, bool, str]:
        """
        Execute one supplemental evidence-completion pass.

        Returns:
            (state, made_progress, progress_reason)

        Progress means at least one genuinely new successful investigation
        evidence result was added.
        """
        original_query = _clean_text(state.query)
        request = _clean_text(supplemental_request)

        if not request:
            reason = (
                "Supplemental investigation requested without a missing-"
                "evidence description."
            )
            state.add_warning(reason)
            return state, False, reason

        before_signatures = self._successful_investigation_evidence_signatures(state)
        before_success_count = len(before_signatures)
        before_ledger = state.investigation_ledger.fingerprint()

        existing_summary = self._build_existing_investigation_evidence_summary(state)

        supplemental_query = request

        print("\n" + "═" * 70)
        print(
            "[RetrieverAgent] ↻ SUPPLEMENTAL INVESTIGATION — "
            "adaptive evidence-completion pass"
        )
        print(f"[RetrieverAgent]    Original query : {original_query}")
        print(
            "[RetrieverAgent]    Missing need   : " f"{request.replace(chr(10), ' | ')}"
        )
        print("─" * 70)

        supplemental_plan = self._investigation_planner.plan_supplemental(
            original_query=original_query,
            supplemental_request=request,
            existing_evidence_summary=existing_summary,
            schema_context=self._full_schema_catalog,
            policy_context=state.policy_context,
            ledger_context=state.investigation_ledger.compact_summary(),
        )

        print(
            "[RetrieverAgent]    Supplemental tasks planned: "
            f"{len(supplemental_plan.tasks)}"
        )

        if not supplemental_plan.tasks:
            reason = (
                "Supplemental planner produced no self-reviewed executable "
                "evidence tasks."
            )

            if supplemental_plan.missing_business_rules:
                reason += " " + "; ".join(supplemental_plan.missing_business_rules)

            state.add_warning(reason)
            state.add_trace("RetrieverAgent", reason)
            state.query = original_query

            print(f"[RetrieverAgent]    No progress: {reason}")
            print("═" * 70 + "\n")

            return state, False, reason

        self._execute_investigation(
            query=supplemental_query,
            state=state,
            append_results=True,
            supplemental_label=request,
            preplanned_plan=supplemental_plan,
        )

        state.query = original_query
        state.reset_evidence_failure()

        after_signatures = self._successful_investigation_evidence_signatures(state)
        new_signatures = after_signatures - before_signatures
        after_ledger = state.investigation_ledger.fingerprint()
        made_progress = bool(new_signatures) or after_ledger != before_ledger

        if made_progress:
            progress_reason = (
                "Supplemental investigation added "
                f"{len(new_signatures)} new successful evidence result(s); "
                f"successful evidence sets increased from "
                f"{before_success_count} to {len(after_signatures)}."
            )
        else:
            progress_reason = (
                "Supplemental investigation added no new successful evidence. "
                "The pass either repeated existing evidence or failed to "
                "retrieve the missing evidence domain."
            )

        state.add_trace(
            "RetrieverAgent",
            progress_reason,
        )

        print(
            "[RetrieverAgent] ↻ Supplemental investigation complete. "
            f"Progress: {made_progress}. {progress_reason}"
        )
        print(
            "[RetrieverAgent]    Total investigation results: "
            f"{len(state.investigation_results)}"
        )
        print("═" * 70 + "\n")

        return state, made_progress, progress_reason

    # ------------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------------

    async def run(
        self,
        state: WorkflowState,
    ) -> WorkflowState:
        query = _clean_text(state.query)

        print("\n" + "═" * 70)
        print("[RetrieverAgent] ▶  STARTING — " "Agent 2: Evidence Contract Execution")
        print(
            "[RetrieverAgent]    Frameworks : "
            "Policy RAG + Schema + DirectQuery/Investigation + MCP"
        )
        print(f"[RetrieverAgent]    Query      : {query}")
        print("─" * 70)

        if should_skip_retrieval(query):
            state.retrieved_nodes = []
            state.retrieved_context = ""

            state.add_trace(
                "RetrieverAgent",
                ("Retrieval skipped for a simple " "conversational query."),
            )

            print("[RetrieverAgent] [GUARD] Retrieval skipped.")
            print("═" * 70 + "\n")

            return state

        contract = _capture_evidence_contract(
            state,
            query,
        )

        state.reset_retrieval_state()

        _restore_evidence_contract(
            state,
            contract,
        )

        required_types = set(state.required_evidence_types)

        retrieval_mode = getattr(
            state,
            "retrieval_mode",
            RetrievalMode.EVIDENCE_ONLY,
        )

        if not isinstance(
            retrieval_mode,
            RetrievalMode,
        ):
            try:
                retrieval_mode = RetrievalMode(str(retrieval_mode).strip().lower())
            except ValueError:
                retrieval_mode = (
                    RetrievalMode.INVESTIGATION
                    if state.investigation_requested
                    else (
                        RetrievalMode.DIRECT_QUERY
                        if state.requires_live_data
                        else RetrievalMode.EVIDENCE_ONLY
                    )
                )

        policy_required = (
            EvidenceType.POLICY.value in required_types
            or state.requires_policy_evidence
        )

        schema_required = EvidenceType.SCHEMA.value in required_types

        database_required = (
            EvidenceType.DATABASE.value in required_types or state.requires_live_data
        )

        investigation_required = (
            retrieval_mode == RetrievalMode.INVESTIGATION
            or EvidenceType.INVESTIGATION.value in required_types
            or state.investigation_requested
        )

        direct_query_required = (
            retrieval_mode == RetrievalMode.DIRECT_QUERY
            and database_required
            and not investigation_required
        )

        print(
            "[RetrieverAgent]    Required types : " f"{state.required_evidence_types}"
        )
        print("[RetrieverAgent]    Policy required: " f"{policy_required}")
        print("[RetrieverAgent]    Schema required: " f"{schema_required}")
        print("[RetrieverAgent]    Database req.  : " f"{database_required}")
        print("[RetrieverAgent]    Retrieval mode: " f"{retrieval_mode.value}")
        print("[RetrieverAgent]    Investigation : " f"{investigation_required}")
        print("[RetrieverAgent]    Direct query  : " f"{direct_query_required}")

        usable_policy_count = 0

        if policy_required:
            usable_policy_count = self._retrieve_policy_evidence(
                query=query,
                state=state,
                explicitly_required=True,
            )

        # Schema is answerable evidence for structural requests and grounding
        # evidence for live-data or investigation requests.
        if schema_required:
            full_catalog = investigation_required or not database_required

            self._prepare_schema_evidence(
                state=state,
                full_catalog=full_catalog,
                query=query,
            )

        if investigation_required:
            print(
                "[RetrieverAgent] [Investigation] "
                "Executing investigation evidence review (broad or targeted)."
            )

            # Investigation planning always receives complete schema grounding.
            if not state.schema_context.strip():
                self._prepare_schema_evidence(
                    state=state,
                    full_catalog=True,
                    query=query,
                )

            self._execute_investigation(
                query=query,
                state=state,
            )

        elif direct_query_required:
            print(
                "[RetrieverAgent] [DirectQuery] "
                "Executing bounded live-data request through direct SQL path."
            )

            if not state.schema_context.strip():
                self._prepare_schema_evidence(
                    state=state,
                    full_catalog=False,
                    query=query,
                )

            self._execute_direct_query(
                query=query,
                state=state,
            )

        elif database_required:
            # Backward-compatible fallback for callers that require live data
            # but did not provide the new RetrievalMode contract.
            print(
                "[RetrieverAgent] [DirectQuery] "
                "Live data required without explicit DirectQuery mode; "
                "using DirectQuery fallback."
            )

            if not state.schema_context.strip():
                self._prepare_schema_evidence(
                    state=state,
                    full_catalog=False,
                    query=query,
                )

            self._execute_direct_query(
                query=query,
                state=state,
            )

        else:
            print(
                "[RetrieverAgent] [Database] "
                "No live database execution required by the contract."
            )

        state.build_retrieved_context()

        missing_required: list[str] = []

        available_types = state.available_evidence_types()

        for required_type in state.required_evidence_types:
            if required_type not in available_types:
                missing_required.append(required_type)

        if missing_required:
            state.add_warning(
                (
                    "RetrieverAgent did not obtain all required evidence "
                    "types: " + ", ".join(missing_required)
                )
            )

        state.add_trace(
            "RetrieverAgent",
            (
                f"required={state.required_evidence_types}; "
                f"retrieval_mode={retrieval_mode.value}; "
                f"available={sorted(available_types)}; "
                f"missing={missing_required}; "
                f"policy passages={usable_policy_count}; "
                f"schema evidence={state.schema_query_succeeded}; "
                f"schema chars={len(state.schema_context)}; "
                f"investigation success="
                f"{state.investigation_succeeded}; "
                f"SQL valid="
                f"{bool(state.sql_plan and state.sql_plan.is_valid)}; "
                f"database success="
                f"{state.database_query_succeeded}; "
                f"combined context="
                f"{len(state.retrieved_context)} chars."
            ),
        )

        print("─" * 70)
        print("[RetrieverAgent] ✔  COMPLETE")
        print(
            "[RetrieverAgent]    Required types  : " f"{state.required_evidence_types}"
        )
        print("[RetrieverAgent]    Available types : " f"{sorted(available_types)}")
        print("[RetrieverAgent]    Missing types   : " f"{missing_required}")
        print("[RetrieverAgent]    Retrieval mode  : " f"{retrieval_mode.value}")
        print("[RetrieverAgent]    Policy passages : " f"{usable_policy_count}")
        print("[RetrieverAgent]    Policy sources  : " f"{len(state.policy_sources)}")
        print(
            "[RetrieverAgent]    Schema evidence : " f"{state.schema_query_succeeded}"
        )
        print(
            "[RetrieverAgent]    Schema context  : "
            f"{len(state.schema_context)} chars"
        )
        print(
            "[RetrieverAgent]    Investigation ok: " f"{state.investigation_succeeded}"
        )
        print(
            "[RetrieverAgent]    SQL plan valid  : "
            f"{bool(state.sql_plan and state.sql_plan.is_valid)}"
        )
        print(
            "[RetrieverAgent]    Database success: " f"{state.database_query_succeeded}"
        )
        print(
            "[RetrieverAgent]    Combined context: "
            f"{len(state.retrieved_context)} chars"
        )
        print("[RetrieverAgent]    Next agent      : " "GraderWriterAgent")
        print("═" * 70 + "\n")

        return state


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
