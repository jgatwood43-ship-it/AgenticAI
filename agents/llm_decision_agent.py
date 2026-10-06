"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  agents/llm_decision_agent.py                                               ║
║  Agent 1 – Structured Routing and Evidence Classification                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE
───────
Classify each user query into exactly one workflow route and establish the
initial evidence contract used by downstream agents.

Routes
------
Route.RETRIEVER
    Use when the request requires organizational data, schema documentation,
    policy retrieval, investigation, or evidence-backed analysis.

Route.ANSWER_GENERATOR
    Use only for greetings, application-purpose questions, or general
    cybersecurity knowledge that does not require organizational evidence or
    requested source retrieval.

Agent 1 also determines whether the request needs:

    * live organizational data;
    * policy evidence;
    * schema evidence;
    * multi-step investigation;
    * interpretation or analysis;
    * and, for Route.RETRIEVER, which Agent 2 execution mode applies:

        RetrievalMode.EVIDENCE_ONLY
            schema/policy retrieval without live database execution;

        RetrievalMode.DIRECT_QUERY
            one bounded live-data lookup or report;

        RetrievalMode.INVESTIGATION
            broad or multi-step security analysis.

Important distinction
---------------------
The presence of words such as "table", "schema", or "relationship" does not
automatically mean live data is needed.

Examples:

    "Describe the tables and relationships."
        schema evidence only; no live data; no interpretation.

    "List employees from the database."
        live database evidence; no interpretation.

    "Review table activity for security concerns using NIST and CIS."
        schema + policy + investigation; live data; interpretation.

This agent does not answer the user's question, retrieve evidence, generate SQL,
call MCP, or inspect database schemas.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from llama_index.core.llms import LLM
from pydantic import BaseModel, Field

from core.state import (
    EvidenceType,
    RetrievalMode,
    Route,
    WorkflowState,
)

# ============================================================================
# High-confidence classification vocabulary
# ============================================================================

_DIRECT_FULLMATCH_PATTERNS = (
    r"hello",
    r"hi",
    r"hey",
    r"good morning",
    r"good afternoon",
    r"good evening",
    r"who are you",
    r"what are you",
    r"what is your purpose",
    r"what can you do",
)

_GENERAL_KNOWLEDGE_PATTERNS = (
    r"what does .+ mean",
    r"define .+",
    r"explain .+",
    r"why is .+ important",
    r"what is least privilege",
    r"what is zero trust",
    r"what is multi-factor authentication",
    r"what is role-based access control",
)

_SCHEMA_TERMS = (
    "schema",
    "schemas",
    "table structure",
    "database structure",
    "data model",
    "table relationship",
    "table relationships",
    "columns and relationships",
    "primary key",
    "primary keys",
    "foreign key",
    "foreign keys",
    "describe the tables",
    "show the tables",
    "list the tables",
)

_POLICY_EVIDENCE_TERMS = (
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
    "according to nist",
    "according to cis",
    "cite nist",
    "cite cis",
    "based on nist",
    "based on cis",
)

_LIVE_DATA_TERMS = (
    "employee",
    "employees",
    "user account",
    "user accounts",
    "badge",
    "badged",
    "access event",
    "access events",
    "room access",
    "entered",
    "entry",
    "clocked in",
    "clocked-in",
    "clocked out",
    "clocked-out",
    "punch",
    "time clock",
    "record",
    "records",
    "row",
    "rows",
    "count",
    "how many",
    "current status",
    "actual data",
    "list of employees",
    "who accessed",
    "who entered",
    "department",
    "job title",
    "key access",
    "camera",
    "video footage",
    "activity",
    "events",
    "company data",
    "organizational data",
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
    "security concerns",
    "security issue",
    "security issues",
    "security risk",
    "security risks",
    "suspicious",
    "anomaly",
    "anomalies",
    "inconsistency",
    "inconsistencies",
    "violation",
    "violations",
    "review for",
)

_INVESTIGATION_TERMS = (
    "review the records",
    "review company records",
    "review the tables",
    "security concerns",
    "security issues",
    "security risks",
    "suspicious activity",
    "identify anomalies",
    "find anomalies",
    "investigate",
    "review for inconsistencies",
    "review for any possible",
    "provide as much detail as possible",
)

# Cross-domain correlation signals.
#
# These do not choose tables or SQL. They only identify questions whose answer
# requires comparing multiple organizational facts/events/states rather than
# performing one bounded factual lookup.
_CORRELATION_RELATION_TERMS = (
    "while",
    "at the time",
    "at that time",
    "when",
    "during",
    "before",
    "after",
    "without",
    "while not",
    "when not",
    "at the same time",
    "correlate",
    "compare",
    "compared with",
    "compared to",
    "in relation to",
)

_CORRELATION_DOMAIN_GROUPS = (
    (
        "badge",
        "badged",
        "access",
        "accessed",
        "room",
        "entered",
        "entry",
    ),
    (
        "clocked in",
        "clocked-in",
        "clocked out",
        "clocked-out",
        "time clock",
        "punch",
    ),
    (
        "employee status",
        "employment status",
        "active employee",
        "terminated",
        "termination",
        "employment",
    ),
    (
        "key access",
        "key activity",
        "key checkout",
        "key checked out",
        "returned key",
    ),
    (
        "camera",
        "video",
        "video footage",
        "footage",
    ),
    (
        "authorization",
        "authorized",
        "access rule",
        "access rules",
        "permission",
        "permissions",
    ),
)


_DIRECT_QUERY_ACTION_TERMS = (
    "list",
    "show",
    "provide",
    "give me",
    "what is",
    "what are",
    "who is",
    "who are",
    "when was",
    "when did",
    "how many",
    "count",
    "find",
)

_NAMED_PERSON_ORG_LOOKUP_PATTERNS = (
    # Possessive / named-person organizational facts.
    r"\bwhat\s+is\s+[A-Z][A-Za-z'’-]+\s+[A-Z][A-Za-z'’-]+'s\s+"
    r"(?:title|job title|department|status|current status)\b",
    # "What department is James Anderson in?"
    r"\bwhat\s+(?:department|job title|title|status)\s+is\s+"
    r"[A-Z][A-Za-z'’-]+\s+[A-Z][A-Za-z'’-]+\b",
    # Named-person clock activity.
    r"\bwhen\s+(?:was|did)\s+[A-Z][A-Za-z'’-]+\s+[A-Z][A-Za-z'’-]+\s+"
    r"(?:last\s+|most\s+recently\s+)?(?:clock(?:ed)?\s+in|clock(?:ed)?\s+out|punch)\b",
    # Named-person badge / room / key activity.
    r"\bwhen\s+(?:was|did)\s+[A-Z][A-Za-z'’-]+\s+[A-Z][A-Za-z'’-]+\s+"
    r"(?:last\s+|most\s+recently\s+)?(?:badge|badged|enter|entered|access|accessed|use|used)\b",
    # "Show/List James Anderson's ..."
    r"\b(?:show|list|provide|give me)\s+[A-Z][A-Za-z'’-]+\s+"
    r"[A-Z][A-Za-z'’-]+'s\s+"
    r"(?:title|job title|department|status|clock|badge|access|key|activity|records?)\b",
)


_DIRECT_QUERY_SHAPE_TERMS = (
    "each",
    "every",
    "per ",
    "last",
    "latest",
    "most recent",
    "first",
    "earliest",
    "all employees",
    "employee names",
    "job title",
    "department",
    "clocked in",
    "clocked out",
    "denied",
)

_BROAD_INVESTIGATION_TERMS = (
    "review",
    "assess",
    "analyze",
    "analyse",
    "evaluate",
    "investigate",
    "security concern",
    "security concerns",
    "security issue",
    "security issues",
    "security risk",
    "security risks",
    "suspicious",
    "anomaly",
    "anomalies",
    "inconsistency",
    "inconsistencies",
    "violation",
    "violations",
    "correlate",
    "review for",
)


_STRUCTURAL_ACTION_TERMS = (
    "describe",
    "show",
    "list",
    "provide",
    "explain",
)

_OPERATIONAL_ACTION_TERMS = (
    "review",
    "inspect",
    "assess",
    "analyze",
    "analyse",
    "evaluate",
    "compare",
    "correlate",
    "investigate",
    "find",
    "determine",
    "identify",
)


# ============================================================================
# Structured classification
# ============================================================================


class RoutingDecision(BaseModel):
    """One validated routing and evidence classification."""

    route: Literal[
        "retriever",
        "answer_generator",
    ] = Field(
        description=(
            "retriever when evidence or organizational data is required; "
            "answer_generator for general knowledge or conversation."
        )
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the route and evidence classification.",
    )

    requires_live_data: bool = Field(
        description=(
            "True only when actual organizational records or current database "
            "rows are needed. Structural metadata alone is not live data."
        )
    )

    requires_policy_evidence: bool = Field(
        description=(
            "True when the user requests NIST, CIS, standards, controls, "
            "policies, citations, or evidence-backed guidance."
        )
    )

    requires_schema_evidence: bool = Field(
        description=(
            "True when schema documentation is requested directly or is "
            "needed to ground organizational-data analysis."
        )
    )

    requires_investigation: bool = Field(
        description=(
            "True when the request is a broad or multi-step review rather "
            "than one bounded lookup."
        )
    )

    retrieval_mode: Literal[
        "evidence_only",
        "direct_query",
        "investigation",
    ] = Field(
        default="evidence_only",
        description=(
            "Agent 2 execution mode. direct_query is one bounded live-data "
            "lookup/report; investigation is a broad multi-step analysis; "
            "evidence_only is schema/policy retrieval without live SQL."
        ),
    )

    requires_interpretation: bool = Field(
        description=(
            "True when evidence must be compared, correlated, assessed, or "
            "used to reach a security conclusion."
        )
    )

    reason: str = Field(
        min_length=5,
        description=("Concise explanation of the route and evidence requirements."),
    )


_ROUTING_PROMPT = """
You are the routing and evidence-requirement classifier for a cybersecurity
user-access analysis system.

USER QUERY
----------
{query}

ROUTES
------
retriever:
Use when the request requires organizational records, schema documentation,
policy retrieval, investigation, or evidence-backed conclusions.

answer_generator:
Use only for greetings, application-purpose questions, or general
cybersecurity knowledge that requires no organizational evidence and no
requested NIST/CIS/policy retrieval.

CLASSIFICATION DIMENSIONS
-------------------------
requires_live_data:
True only when actual organizational records, events, counts, names, statuses,
dates, or database rows are needed.

requires_policy_evidence:
True when the request explicitly requires NIST, CIS, policies, standards,
controls, guidance, source citations, or documents on file.

requires_schema_evidence:
True when:
- database structure is directly requested; or
- schema grounding is needed to query or interpret organizational data.

requires_investigation:
True for:
- broad, multi-step reviews that may require several focused analyses; OR
- a bounded question whose answer requires correlating/comparing two or more
  organizational evidence domains or states.

A question can therefore be an investigation even when it asks for one final
yes/no answer or one list of employees.

requires_interpretation:
True when evidence must be assessed, compared, correlated, synthesized, or used
to determine concerns, risks, anomalies, violations, or whether one event/state
coincided with another.

retrieval_mode:
- evidence_only:
  schema/policy/document retrieval with no live database rows.
- direct_query:
  one bounded database lookup/report, even if it needs joins, grouping,
  "last/latest", "each/per", counts, or simple filtering.
- investigation:
  broad or multi-step review, OR a cross-domain correlation question that asks
  the system to compare organizational events/states across multiple evidence
  domains. This includes bounded questions where the final answer may be one
  yes/no result or one list, but reaching that answer requires correlation.

IMPORTANT DISTINCTIONS
----------------------
1. "Describe the tables and relationships"
   - retriever
   - schema evidence required
   - live data false
   - interpretation false

2. "List employee names"
   - retriever
   - retrieval_mode direct_query
   - live data true
   - schema evidence true
   - investigation false
   - interpretation false

2a. "What is James Anderson's title?"
   - retriever
   - retrieval_mode direct_query
   - live data true
   - schema evidence true
   - investigation false
   - interpretation false

2b. "For each employee, show the most recent clock-in time."
   - retriever
   - retrieval_mode direct_query
   - live data true
   - schema evidence true
   - investigation false
   - interpretation false

2c. "Are there employees who accessed a room while not clocked in?"
   - retriever
   - retrieval_mode investigation
   - live data true
   - schema evidence true
   - investigation true
   - interpretation true
   Reason: the final answer is bounded, but it requires temporal correlation
   between separate organizational activity domains.

3. "Review table activity for concerns based on NIST and CIS"
   - retriever
   - retrieval_mode investigation
   - live data true
   - policy evidence true
   - schema evidence true
   - investigation true
   - interpretation true

4. The word "table" does not by itself prove either schema-only or live-data
   intent. Classify the operation requested.

RULES
-----
1. This is classification only. Do not answer the question.
2. Do not retrieve evidence.
3. Do not generate SQL.
4. Do not name or call tools.
5. If organizational evidence may be required, choose retriever.
6. Explicit NIST/CIS/policy requests always require policy evidence.
7. Analytical requests about company tables or records require live data.
8. Structural metadata requests do not require live data.
9. A bounded lookup is direct_query even when it uses multiple tables,
   grouping, MAX/MIN, "each", "per", "last", or "most recent", PROVIDED the
   question does not require cross-domain evidence correlation.
10. Do not classify a simple lookup as investigation merely because it needs a
    join or aggregate. However, classify it as investigation when the requested
    answer depends on comparing two or more organizational events/states/domains.
11. Broad security review/assessment language and genuine cross-domain
    correlation take precedence and use investigation.
12. If no live data is required, retrieval_mode must be evidence_only.
13. If live data is required and investigation is false, retrieval_mode must be
    direct_query.
14. If investigation is true, retrieval_mode must be investigation.
15. A request for a factual organizational attribute or activity of a named
    person requires live data. Examples include:
    - "What is James Anderson's title?"
    - "What department is Maria Garcia in?"
    - "When was James Anderson last clocked in?"
    - "Show David Kim's badge-access records."
    These are direct_query requests, not schema-only requests.
16. Schema documentation may be needed to ground the query, but schema evidence
    alone cannot answer a question asking for the actual value stored for a
    named employee/person.
17. If uncertain about the top-level route, choose retriever.
"""


# ============================================================================
# Helpers
# ============================================================================


def _clean_text(
    value: Any,
) -> str:
    return str(value or "").strip()


def _contains_any(
    query: str,
    terms: tuple[str, ...],
) -> bool:
    query_lower = _clean_text(query).lower()

    return any(term in query_lower for term in terms)


def _matches_direct_conversation(
    query: str,
) -> bool:
    normalized = _clean_text(query)

    return any(
        re.fullmatch(
            rf"\s*{pattern}[?.!,]*\s*",
            normalized,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in _DIRECT_FULLMATCH_PATTERNS
    )


def _matches_general_knowledge(
    query: str,
) -> bool:
    normalized = _clean_text(query)

    return any(
        re.fullmatch(
            rf"\s*{pattern}[?.!]*\s*",
            normalized,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in _GENERAL_KNOWLEDGE_PATTERNS
    )


def _looks_structural_only(
    query: str,
) -> bool:
    """
    Return True for direct metadata/documentation requests with no analytical
    or operational-data operation.
    """
    query_lower = _clean_text(query).lower()

    has_schema_subject = _contains_any(
        query_lower,
        _SCHEMA_TERMS,
    )

    has_structural_action = _contains_any(
        query_lower,
        _STRUCTURAL_ACTION_TERMS,
    )

    has_analytical_action = _contains_any(
        query_lower,
        _ANALYTICAL_TERMS,
    )

    has_operational_action = _contains_any(
        query_lower,
        _OPERATIONAL_ACTION_TERMS,
    )

    return (
        has_schema_subject
        and has_structural_action
        and not has_analytical_action
        and not has_operational_action
    )


def _looks_named_person_org_lookup(
    query: str,
) -> bool:
    """
    Return True when the user asks for a factual organizational record about
    one explicitly named person.

    This helper determines only that live organizational data is required.
    It does NOT choose tables, columns, relationships, SQL, or the answer.
    """
    cleaned = _clean_text(query)

    if not cleaned:
        return False

    return any(
        re.search(
            pattern,
            cleaned,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in _NAMED_PERSON_ORG_LOOKUP_PATTERNS
    )


def _correlation_domain_count(
    query: str,
) -> int:
    """Count distinct organizational evidence domains referenced by the query."""
    lowered = _clean_text(query).lower()

    return sum(
        1
        for group in _CORRELATION_DOMAIN_GROUPS
        if any(term in lowered for term in group)
    )


def _looks_cross_domain_correlation(
    query: str,
) -> bool:
    """
    Return True when answering the question requires comparing organizational
    evidence across two or more operational domains/states.

    This is routing-only classification. It never chooses tables, columns,
    relationships, or SQL.
    """
    lowered = _clean_text(query).lower()

    if not lowered:
        return False

    if _correlation_domain_count(lowered) < 2:
        return False

    relational_language = any(term in lowered for term in _CORRELATION_RELATION_TERMS)

    analytical_language = any(
        term in lowered
        for term in (
            "compare",
            "correlate",
            "determine if",
            "determine whether",
            "identify",
            "find",
            "are there any",
            "is there any",
            "who",
            "which employees",
        )
    )

    return relational_language or analytical_language


def _looks_broad_investigation(
    query: str,
) -> bool:
    """
    Return True for broad reviews OR bounded questions that require genuine
    cross-domain correlation/interpretation.
    """
    return _contains_any(
        query,
        _BROAD_INVESTIGATION_TERMS,
    ) or _looks_cross_domain_correlation(query)


def _looks_direct_live_query(
    query: str,
) -> bool:
    """
    Return True for a bounded live-data lookup/report.

    A query may contain words such as "last", "each", or "find" and still be a
    direct query. Broad analytical/security-review language takes precedence.
    """
    if _looks_broad_investigation(query):
        return False

    if _looks_structural_only(query):
        return False

    live_data = _contains_any(
        query,
        _LIVE_DATA_TERMS,
    )

    if not live_data:
        return False

    has_direct_action = _contains_any(
        query,
        _DIRECT_QUERY_ACTION_TERMS,
    )

    has_direct_shape = _contains_any(
        query,
        _DIRECT_QUERY_SHAPE_TERMS,
    )

    return has_direct_action or has_direct_shape or live_data


def _retrieval_mode_from_decision(
    decision: RoutingDecision,
) -> RetrievalMode:
    """Map evidence classification to Agent 2 execution strategy."""
    if decision.requires_investigation:
        return RetrievalMode.INVESTIGATION

    if decision.requires_live_data:
        return RetrievalMode.DIRECT_QUERY

    return RetrievalMode.EVIDENCE_ONLY


def _normalize_confidence(
    value: float,
) -> float:
    if value > 1:
        return min(
            value / 100,
            1.0,
        )

    return max(
        min(
            value,
            1.0,
        ),
        0.0,
    )


def _extract_structured_response(
    response: Any,
) -> RoutingDecision:
    raw = getattr(
        response,
        "raw",
        None,
    )

    if isinstance(
        raw,
        RoutingDecision,
    ):
        return raw

    if isinstance(
        raw,
        dict,
    ):
        normalized = dict(raw)

        if isinstance(
            normalized.get("confidence"),
            (int, float),
        ):
            normalized["confidence"] = _normalize_confidence(
                float(normalized["confidence"])
            )

        return RoutingDecision.model_validate(normalized)

    if isinstance(
        response,
        RoutingDecision,
    ):
        return response

    additional_kwargs = (
        getattr(
            response,
            "additional_kwargs",
            {},
        )
        or {}
    )

    parsed = additional_kwargs.get("parsed")

    if isinstance(
        parsed,
        RoutingDecision,
    ):
        return parsed

    if isinstance(
        parsed,
        dict,
    ):
        normalized = dict(parsed)

        if isinstance(
            normalized.get("confidence"),
            (int, float),
        ):
            normalized["confidence"] = _normalize_confidence(
                float(normalized["confidence"])
            )

        return RoutingDecision.model_validate(normalized)

    response_text = _clean_text(
        getattr(
            response,
            "text",
            response,
        )
    )

    response_text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        response_text,
        flags=re.IGNORECASE,
    ).strip()

    if not response_text:
        raise ValueError("The routing model returned an empty response.")

    return RoutingDecision.model_validate_json(response_text)


def _route_enum(
    route_name: str,
) -> Route:
    if _clean_text(route_name).lower() == "answer_generator":
        return Route.ANSWER_GENERATOR

    return Route.RETRIEVER


def _required_evidence_types(
    decision: RoutingDecision,
) -> list[str]:
    evidence_types: list[str] = []

    if decision.requires_policy_evidence:
        evidence_types.append(EvidenceType.POLICY.value)

    if decision.requires_schema_evidence:
        evidence_types.append(EvidenceType.SCHEMA.value)

    if decision.requires_live_data:
        evidence_types.append(EvidenceType.DATABASE.value)

    if decision.requires_investigation:
        evidence_types.append(EvidenceType.INVESTIGATION.value)

    return list(dict.fromkeys(evidence_types))


def _initial_investigation_components(query: str) -> list[str]:
    """
    Derive a compact semantic evidence contract for an investigation.

    These names are reasoning concepts only. They are deliberately NOT table or
    column names. RetrieverAgent/InvestigationPlanner later determine which
    documented schema evidence can satisfy them.
    """
    lowered = _clean_text(query).lower()
    components: list[str] = []

    def add(name: str) -> None:
        if name not in components:
            components.append(name)

    # Shared correlation facts.
    if _looks_cross_domain_correlation(query):
        add("entity identity needed for correlation")
        add("event or state timing needed for correlation")
        add("cross-domain relationship at the relevant time")

    # Operational domains are semantic, not schema-specific.
    if any(term in lowered for term in _CORRELATION_DOMAIN_GROUPS[0]):
        add("access event evidence")

    if any(term in lowered for term in _CORRELATION_DOMAIN_GROUPS[1]):
        add("time-clock activity evidence")
        add("clock state at the correlated event time")

    if any(term in lowered for term in _CORRELATION_DOMAIN_GROUPS[2]):
        add("employment status evidence")
        add("employment state at the correlated event time")

    if any(term in lowered for term in _CORRELATION_DOMAIN_GROUPS[3]):
        add("key-access activity evidence")

    if any(term in lowered for term in _CORRELATION_DOMAIN_GROUPS[4]):
        add("video-footage evidence")

    if any(term in lowered for term in _CORRELATION_DOMAIN_GROUPS[5]):
        add("access-authorization evidence")

    # Broad investigations may not name specific operational domains. Give the
    # planner a minimal semantic objective rather than inventing domain details.
    if not components:
        add("operational evidence relevant to the investigation objective")

    return components


# ============================================================================
# LLMDecisionAgent
# ============================================================================


class LLMDecisionAgent:
    """
    Agent 1 – bounded route and evidence classification.

    ``schema_retriever`` remains optional for compatibility with the workflow
    constructor. It is intentionally unused because retrieval belongs to
    RetrieverAgent.
    """

    def __init__(
        self,
        llm: LLM,
        schema_retriever: Any | None = None,
    ) -> None:
        self._structured_llm = llm.as_structured_llm(RoutingDecision)

        print(
            "\n[LLMDecisionAgent] ⚙  Initialising structured routing "
            "and evidence classifier…"
        )
        print(
            "[LLMDecisionAgent]    Framework : "
            "deterministic classification + one structured LLM fallback"
        )
        print("[LLMDecisionAgent]    Output    : " "route + initial evidence contract")
        print("[LLMDecisionAgent]    ReAct     : none")
        print("[LLMDecisionAgent]    MCP tools : none")
        print("[LLMDecisionAgent]    Retrieval : none")
        print(
            "[LLMDecisionAgent]    Guard     : "
            "named-person facts require live data; cross-domain correlations require investigation"
        )
        print("[LLMDecisionAgent]    Fallback  : RETRIEVER")

    @staticmethod
    def _deterministic_decision(
        query: str,
    ) -> RoutingDecision | None:
        """Return a high-confidence deterministic decision when possible."""
        if _matches_direct_conversation(query):
            return RoutingDecision(
                route="answer_generator",
                confidence=1.0,
                requires_live_data=False,
                requires_policy_evidence=False,
                requires_schema_evidence=False,
                requires_investigation=False,
                requires_interpretation=False,
                retrieval_mode="evidence_only",
                reason=(
                    "The query is conversational or asks about the "
                    "application's purpose."
                ),
            )

        if _looks_structural_only(query):
            return RoutingDecision(
                route="retriever",
                confidence=0.99,
                requires_live_data=False,
                requires_policy_evidence=_contains_any(
                    query,
                    _POLICY_EVIDENCE_TERMS,
                ),
                requires_schema_evidence=True,
                requires_investigation=False,
                requires_interpretation=False,
                retrieval_mode="evidence_only",
                reason=(
                    "The request is for structural documentation rather than "
                    "live organizational records."
                ),
            )

        # High-confidence cross-domain correlation.
        if _looks_cross_domain_correlation(query):
            requires_policy = _contains_any(
                query,
                _POLICY_EVIDENCE_TERMS,
            )

            return RoutingDecision(
                route="retriever",
                confidence=0.99,
                requires_live_data=True,
                requires_policy_evidence=requires_policy,
                requires_schema_evidence=True,
                requires_investigation=True,
                requires_interpretation=True,
                retrieval_mode="investigation",
                reason=(
                    "The request requires correlation of multiple organizational "
                    "evidence domains/states, so live data, schema grounding, "
                    "multi-step investigation, and interpretation are required."
                ),
            )

        # High-confidence factual organizational lookup.
        #
        # Python enforces only the evidence requirement here: a named person's
        # current organizational fact/activity cannot be answered from schema
        # documentation alone. DirectQueryPlanner/LLM still performs the actual
        # schema reasoning and SQL generation.
        if _looks_named_person_org_lookup(query):
            requires_policy = _contains_any(
                query,
                _POLICY_EVIDENCE_TERMS,
            )

            return RoutingDecision(
                route="retriever",
                confidence=0.99,
                requires_live_data=True,
                requires_policy_evidence=requires_policy,
                requires_schema_evidence=True,
                requires_investigation=False,
                requires_interpretation=False,
                retrieval_mode="direct_query",
                reason=(
                    "The request asks for an actual organizational fact or "
                    "activity for a named person, so live database evidence "
                    "and schema grounding are required."
                ),
            )

        requires_policy = _contains_any(
            query,
            _POLICY_EVIDENCE_TERMS,
        )

        analytical = _contains_any(
            query,
            _ANALYTICAL_TERMS,
        )

        live_data = _contains_any(
            query,
            _LIVE_DATA_TERMS,
        )

        investigation = (
            _contains_any(
                query,
                _INVESTIGATION_TERMS,
            )
            or _looks_broad_investigation(query)
            or _looks_cross_domain_correlation(query)
        )

        # A bounded live-data lookup remains DirectQuery even when it uses
        # ordinary SQL reasoning such as joins, grouping, latest/last, each/per,
        # or simple filtering.
        direct_live_query = (
            live_data and not investigation and _looks_direct_live_query(query)
        )

        # Analytical references to company tables or records require actual
        # operational evidence even when the word "table" also appears.
        requires_live_data = live_data or (
            analytical and not _looks_structural_only(query)
        )

        requires_schema = (
            _contains_any(
                query,
                _SCHEMA_TERMS,
            )
            or requires_live_data
            or investigation
        )

        requires_interpretation = investigation or _looks_cross_domain_correlation(
            query
        )

        if any(
            (
                requires_live_data,
                requires_policy,
                requires_schema,
                investigation,
                requires_interpretation,
            )
        ):
            reasons: list[str] = []

            if requires_live_data:
                reasons.append("organizational records are required")

            if requires_policy:
                reasons.append("policy or standards evidence is explicitly requested")

            if requires_schema:
                reasons.append("schema grounding is required")

            if investigation:
                reasons.append("the request requires a multi-step review")

            if requires_interpretation:
                reasons.append("the evidence must be interpreted")

            return RoutingDecision(
                route="retriever",
                confidence=0.99,
                requires_live_data=requires_live_data,
                requires_policy_evidence=requires_policy,
                requires_schema_evidence=requires_schema,
                requires_investigation=investigation,
                requires_interpretation=requires_interpretation,
                retrieval_mode=(
                    "investigation"
                    if investigation
                    else "direct_query" if requires_live_data else "evidence_only"
                ),
                reason=("Retriever required because " + "; ".join(reasons) + "."),
            )

        if _matches_general_knowledge(query):
            return RoutingDecision(
                route="answer_generator",
                confidence=0.95,
                requires_live_data=False,
                requires_policy_evidence=False,
                requires_schema_evidence=False,
                requires_investigation=False,
                requires_interpretation=False,
                retrieval_mode="evidence_only",
                reason=(
                    "The query asks for general cybersecurity knowledge "
                    "without organizational or requested source evidence."
                ),
            )

        return None

    async def run(
        self,
        state: WorkflowState,
    ) -> WorkflowState:
        """Classify the query and store the initial evidence contract."""
        query = _clean_text(state.query)

        print("\n" + "═" * 70)
        print(
            "[LLMDecisionAgent] ▶  STARTING — "
            "Agent 1: Routing and Evidence Classification"
        )
        print(
            "[LLMDecisionAgent]    Framework : "
            "deterministic rules + one structured LLM call"
        )
        print("[LLMDecisionAgent]    ReAct     : none")
        print("[LLMDecisionAgent]    Tools     : none")
        print(f"[LLMDecisionAgent]    Query     : {query}")
        print("─" * 70)

        if not query:
            state.route = Route.ANSWER_GENERATOR
            state.required_evidence_types = []
            state.requires_live_data = False
            state.requires_policy_evidence = False
            state.requires_interpretation = False
            state.set_retrieval_mode(
                RetrievalMode.EVIDENCE_ONLY,
                reason="The query is empty.",
            )
            state.evidence_requirement_reason = "The query is empty."

            state.add_warning("DecisionAgent received an empty query.")

            state.add_trace(
                "LLMDecisionAgent",
                "Empty query routed to AnswerGeneratorAgent.",
            )

            print("[LLMDecisionAgent] ⚠️ Empty query.")
            print("[LLMDecisionAgent] ✔  COMPLETE — " "Route: answer_generator")
            print("═" * 70 + "\n")

            return state

        decision = self._deterministic_decision(query)

        decision_source = "deterministic"

        if decision is None:
            decision_source = "structured_llm"

            try:
                prompt = _ROUTING_PROMPT.format(query=query)

                decision = _extract_structured_response(
                    self._structured_llm.complete(prompt)
                )

            except Exception as exc:
                decision_source = "safe_fallback"

                decision = RoutingDecision(
                    route="retriever",
                    confidence=0.0,
                    requires_live_data=True,
                    requires_policy_evidence=False,
                    requires_schema_evidence=True,
                    requires_investigation=False,
                    requires_interpretation=False,
                    retrieval_mode="direct_query",
                    reason=(
                        "Structured routing failed; the safe retriever "
                        "fallback was selected. "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

                state.add_warning(
                    "Structured routing failed: " f"{type(exc).__name__}: {exc}"
                )

        # Evidence requirements cannot use direct-answer routing.
        if decision.route == "answer_generator" and any(
            (
                decision.requires_live_data,
                decision.requires_policy_evidence,
                decision.requires_schema_evidence,
                decision.requires_investigation,
            )
        ):
            decision = RoutingDecision(
                route="retriever",
                confidence=decision.confidence,
                requires_live_data=decision.requires_live_data,
                requires_policy_evidence=(decision.requires_policy_evidence),
                requires_schema_evidence=(decision.requires_schema_evidence),
                requires_investigation=(decision.requires_investigation),
                requires_interpretation=(decision.requires_interpretation),
                retrieval_mode=decision.retrieval_mode,
                reason=(
                    "Route overridden to retriever because the decision "
                    "requires organizational or source evidence."
                ),
            )

            decision_source += "+safety_override"

        # Deterministic correlation guard.
        if _looks_cross_domain_correlation(query) and (
            not decision.requires_investigation
            or not decision.requires_interpretation
            or decision.retrieval_mode != "investigation"
        ):
            decision = RoutingDecision(
                route="retriever",
                confidence=max(
                    decision.confidence,
                    0.99,
                ),
                requires_live_data=True,
                requires_policy_evidence=decision.requires_policy_evidence,
                requires_schema_evidence=True,
                requires_investigation=True,
                requires_interpretation=True,
                retrieval_mode="investigation",
                reason=(
                    decision.reason
                    + " Deterministic correlation guard: the requested answer "
                    "requires comparing multiple organizational evidence "
                    "domains/states, so the investigation path is required."
                ),
            )

            decision_source += "+cross_domain_correlation_guard"

        # Deterministic evidence guard for named-person organizational facts.
        #
        # The structured LLM remains the normal classifier for ambiguous
        # language. This guard only prevents the specific failure mode where
        # an obviously factual employee lookup is classified as schema-only.
        if (
            _looks_named_person_org_lookup(query)
            and not _looks_cross_domain_correlation(query)
            and (
                not decision.requires_live_data
                or decision.requires_investigation
                or decision.retrieval_mode != "direct_query"
            )
        ):
            decision = RoutingDecision(
                route="retriever",
                confidence=max(
                    decision.confidence,
                    0.99,
                ),
                requires_live_data=True,
                requires_policy_evidence=decision.requires_policy_evidence,
                requires_schema_evidence=True,
                requires_investigation=False,
                requires_interpretation=False,
                retrieval_mode="direct_query",
                reason=(
                    decision.reason
                    + " Deterministic evidence guard: the question asks for "
                    "an actual organizational fact/activity for a named "
                    "person, so live database evidence is required."
                ),
            )
            decision_source += "+named_person_live_data_guard"

        # Enforce internal consistency between the structured retrieval mode and
        # the boolean evidence contract. Booleans remain the source of truth.
        expected_mode = _retrieval_mode_from_decision(decision)

        if decision.retrieval_mode != expected_mode.value:
            decision = RoutingDecision(
                route=decision.route,
                confidence=decision.confidence,
                requires_live_data=decision.requires_live_data,
                requires_policy_evidence=decision.requires_policy_evidence,
                requires_schema_evidence=decision.requires_schema_evidence,
                requires_investigation=decision.requires_investigation,
                requires_interpretation=decision.requires_interpretation,
                retrieval_mode=expected_mode.value,
                reason=(
                    decision.reason
                    + " Retrieval mode normalized to match the evidence contract."
                ),
            )
            decision_source += "+mode_normalized"

        state.route = _route_enum(decision.route)

        state.route_reason = decision.reason
        state.route_confidence = decision.confidence
        state.requires_live_data = decision.requires_live_data
        state.requires_policy_evidence = decision.requires_policy_evidence
        state.requires_interpretation = decision.requires_interpretation

        state.required_evidence_types = _required_evidence_types(decision)

        state.evidence_requirement_reason = decision.reason

        # Agent 2 execution strategy belongs in state before RetrieverAgent runs.
        state.set_retrieval_mode(
            decision.retrieval_mode,
            reason=decision.reason,
        )

        # Investigation memory is initialized once by Agent 1 and then carried
        # through every downstream agent. Agent 1 records only SEMANTIC evidence
        # requirements; it does not choose tables, columns, joins, or SQL.
        if decision.requires_investigation:
            initial_components = _initial_investigation_components(query)

            state.investigation_ledger.initialize(
                state.query,
                required_components=initial_components,
            )

            state.add_trace(
                "LLMDecisionAgent",
                (
                    "Initialized investigation evidence ledger; "
                    f"missing_components={state.investigation_ledger.missing_names()}"
                ),
            )

        state.normalize_evidence_requirements()

        state.add_trace(
            "LLMDecisionAgent",
            (
                f"Source={decision_source}; "
                f"route={decision.route}; "
                f"confidence={decision.confidence:.2f}; "
                f"requires_live_data={decision.requires_live_data}; "
                f"requires_policy_evidence="
                f"{decision.requires_policy_evidence}; "
                f"requires_schema_evidence="
                f"{decision.requires_schema_evidence}; "
                f"requires_investigation="
                f"{decision.requires_investigation}; "
                f"retrieval_mode={decision.retrieval_mode}; "
                f"requires_interpretation="
                f"{decision.requires_interpretation}; "
                f"required_evidence_types="
                f"{state.required_evidence_types}; "
                f"reason={decision.reason}"
            ),
        )

        print("[LLMDecisionAgent]    Source        : " f"{decision_source}")
        print("[LLMDecisionAgent]    Route         : " f"{decision.route}")
        print("[LLMDecisionAgent]    Confidence    : " f"{decision.confidence:.2f}")
        print("[LLMDecisionAgent]    Live data     : " f"{decision.requires_live_data}")
        print(
            "[LLMDecisionAgent]    Policy RAG    : "
            f"{decision.requires_policy_evidence}"
        )
        print(
            "[LLMDecisionAgent]    Schema evidence: "
            f"{decision.requires_schema_evidence}"
        )
        print(
            "[LLMDecisionAgent]    Investigation : "
            f"{decision.requires_investigation}"
        )
        if decision.requires_investigation:
            print(
                "[LLMDecisionAgent]    Ledger missing : "
                f"{state.investigation_ledger.missing_names()}"
            )
        print("[LLMDecisionAgent]    Retrieval mode: " f"{decision.retrieval_mode}")
        print(
            "[LLMDecisionAgent]    Interpretation: "
            f"{decision.requires_interpretation}"
        )
        print(
            "[LLMDecisionAgent]    Evidence types: " f"{state.required_evidence_types}"
        )
        print("[LLMDecisionAgent]    Reason        : " f"{decision.reason}")
        print("[LLMDecisionAgent] ✔  COMPLETE — " f"Route: {decision.route}")
        print("═" * 70 + "\n")

        return state


def _self_test_cross_domain_correlation_routing() -> None:
    """Verify correlation requests route to investigation."""
    correlation_queries = [
        "Are there any employees who gained access to a room while not clocked in?",
        "Which employees accessed rooms while their employment status was terminated?",
        "Compare key access activity with badge access activity for the same employees.",
        "Were there badge entries at the same time as relevant video footage?",
    ]

    for query in correlation_queries:
        assert _looks_cross_domain_correlation(query), query

        decision = LLMDecisionAgent._deterministic_decision(query)

        assert decision is not None
        assert decision.route == "retriever"
        assert decision.requires_live_data is True
        assert decision.requires_schema_evidence is True
        assert decision.requires_investigation is True
        assert decision.requires_interpretation is True
        assert decision.retrieval_mode == "investigation"

    direct_queries = [
        "What is James Anderson's title?",
        "When was James Anderson last clocked in?",
        "For each employee, show the most recent clock-in time.",
    ]

    for query in direct_queries:
        assert not _looks_cross_domain_correlation(query), query

    print("LLMDecisionAgent cross-domain correlation self-test: PASS")
