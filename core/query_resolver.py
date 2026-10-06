"""
core/query_resolver.py
──────────────────────
Resolve conversational follow-up messages into complete standalone queries.

Primary objective
-----------------
Preserve the user's current operation, requested evidence, scope, outputs,
cardinality, grouping, and ordering/recency semantics.

The resolver distinguishes between structural requests, analytical requests,
and mixed requests that compare documentation or policy with operational data.

A prior structural question must never cause a later analytical request to be
resolved as another structural question.

The resolver does not retrieve evidence, generate SQL, classify security
findings, or invent tables, columns, people, dates, filters, thresholds,
business rules, or policy requirements.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from llama_index.core.llms import LLM
from pydantic import BaseModel, Field

MAX_HISTORY_MESSAGES = 10
MAX_MESSAGE_CHARS = 3000


class ResolvedQuery(BaseModel):
    """Result returned by QueryResolver."""

    is_follow_up: bool = Field(
        description=(
            "True when the current message refers to or depends on prior "
            "conversation context."
        )
    )
    standalone_query: str = Field(
        min_length=1,
        description=(
            "A complete natural-language query that preserves the user's "
            "current intent without requiring conversation history."
        ),
    )
    clarification_used: bool = Field(
        description=(
            "True when the current message modifies, narrows, corrects, or "
            "extends an earlier user request."
        )
    )
    reason: str = Field(
        min_length=5,
        description="Brief explanation of how the query was resolved.",
    )


_RESOLVER_PROMPT = """
You resolve conversational follow-ups for a cybersecurity evidence-analysis
workflow.

CURRENT USER MESSAGE
--------------------
{current_query}

MOST RECENT PRIOR USER QUESTION
-------------------------------
{prior_user_query}

OLDER USER QUESTIONS
--------------------
{older_user_context}

TASK
----
Return one complete natural-language standalone query.

MANDATORY RULES
---------------
1. The CURRENT USER MESSAGE is authoritative.
2. Preserve the current operation exactly: structural, retrieval, analytical,
   or transformation.
3. Never replace an analytical request with a prior structural request.
4. Use prior USER messages only to resolve an actual referent or modification.
5. When an analytical request refers to previously discussed tables, schemas,
   policies, records, or documents, preserve the analytical operation and use
   the prior material only as context.
6. Preserve every explicit evidence requirement, including NIST, CIS, policy
   documents, live records, company data, schemas, relationships, employee
   names, dates, and timestamps.
7. Preserve whether the user asks for documentation only, operational records
   only, documentation compared with operational records, or an investigation.
8. Do not infer that the word "tables" means table definitions:
   - "describe the tables" is structural;
   - "review the tables for concerns" is analytical;
   - "compare table activity with NIST" is mixed policy/data analysis.
9. Do not generate SQL or SQL fragments.
10. Do not invent table names, columns, people, dates, thresholds, filters,
    business values, findings, or policy requirements.
11. Do not narrow a broad review into one example query.
12. Do not broaden a direct factual request into an investigation.
13. Remove conversational prefixes only when doing so does not alter meaning.
14. If the current message is already standalone, return it substantially
    unchanged and set is_follow_up=false.
15. If a reference cannot be resolved safely, preserve the current wording
    rather than guessing.
16. standalone_query must not mention the conversation, the assistant, the
    previous answer, try again, knowing this, or based on this.
17. Preserve cardinality words exactly when they affect result shape:
    - all;
    - each;
    - every;
    - per;
    - one;
    - any;
    - only.
18. Preserve ordering and temporal-selection words exactly when they affect the
    requested result:
    - last;
    - latest;
    - most recent;
    - first;
    - earliest;
    - newest;
    - oldest;
    - before;
    - after;
    - since;
    - until.
19. Preserve aggregation/count language:
    - how many;
    - count;
    - total;
    - average;
    - minimum;
    - maximum.
20. Do not rewrite "for each employee" as one employee or one global result.
21. Do not rewrite "last time each employee clocked in" as "employees who were
    last clocked in." Preserve that the user wants one most-recent clock-in
    value independently for each employee.
22. When resolving words such as "those", "them", "these", or "of those", use
    prior context only to identify the referenced entity set. Do not alter the
    current operation, cardinality, or recency requirement.
"""
_FOLLOW_UP_PATTERNS = (
    r"^\s*no\b",
    r"^\s*yes\b",
    r"^\s*ok\b",
    r"^\s*okay\b",
    r"^\s*but\b",
    r"^\s*and\b",
    r"^\s*also\b",
    r"^\s*only\b",
    r"^\s*instead\b",
    r"^\s*what about\b",
    r"^\s*how about\b",
    r"\btry again\b",
    r"\buse that\b",
    r"\buse this\b",
    r"\busing this\b",
    r"\bknowing that\b",
    r"\bknowing this\b",
    r"\bbased on that\b",
    r"\bbased on this\b",
    r"\bwith that information\b",
    r"\bthe same\b",
    r"\bthose\b",
    r"\bthat result\b",
    r"\bprevious\b",
    r"\bearlier\b",
)

_INVESTIGATION_TERMS = (
    "review the company records",
    "review the records",
    "review company records",
    "security anomalies",
    "security anomaly",
    "possible security issues",
    "potential security issues",
    "possible security risks",
    "potential security risks",
    "identify anomalies",
    "find anomalies",
    "determine if there are anomalies",
    "determine whether there are anomalies",
    "suspicious activity",
    "investigate",
    "review for inconsistencies",
    "review for any possible",
    "provide as much detail as possible",
)

_STRUCTURAL_OPERATION_TERMS = (
    "list the tables",
    "show the tables",
    "describe the tables",
    "explain the tables",
    "table schema",
    "table schemas",
    "schemas and relationships",
    "columns and relationships",
    "database structure",
    "data model",
    "foreign keys",
    "primary keys",
)

_ANALYTICAL_OPERATION_TERMS = (
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
    "identify concerns",
    "security concerns",
    "security issues",
    "security risks",
    "suspicious activity",
    "anomalies",
    "inconsistencies",
    "violations",
    "review for",
)

_POLICY_REFERENCE_TERMS = (
    "nist",
    "cis",
    "policy",
    "policies",
    "standards",
    "controls",
    "guidance",
    "documents on file",
)

_OPERATIONAL_DATA_TERMS = (
    "records",
    "activity",
    "events",
    "employees",
    "users",
    "access",
    "badge",
    "keys",
    "time clock",
    "database rows",
    "company data",
    "organizational data",
)


_SCHEMA_REFERENCE_PREFIX_PATTERNS = (
    r"^\s*(?:ok|okay)\s*[,;:]?\s*",
    r"^\s*(?:knowing this|knowing that)\s*[,;:]?\s*",
    r"^\s*(?:based on this|based on that)\s*[,;:]?\s*",
    r"^\s*(?:using this|using that)\s*[,;:]?\s*",
    r"^\s*(?:with that information)\s*[,;:]?\s*",
)

_CARDINALITY_PATTERNS = (
    r"\ball\b",
    r"\beach\b",
    r"\bevery\b",
    r"\bper\b",
    r"\bone\b",
    r"\bany\b",
    r"\bonly\b",
)

_RECENCY_PATTERNS = (
    r"\blast\b",
    r"\blatest\b",
    r"\bmost recent\b",
    r"\bfirst\b",
    r"\bearliest\b",
    r"\bnewest\b",
    r"\boldest\b",
)

_TEMPORAL_BOUNDARY_PATTERNS = (
    r"\bbefore\b",
    r"\bafter\b",
    r"\bsince\b",
    r"\buntil\b",
)

_AGGREGATION_PATTERNS = (
    r"\bhow many\b",
    r"\bcount\b",
    r"\btotal\b",
    r"\baverage\b",
    r"\bminimum\b",
    r"\bmaximum\b",
    r"\bmin\b",
    r"\bmax\b",
)

_ENTITY_SET_REFERENCE_PATTERNS = (
    r"\bthose\b",
    r"\bthem\b",
    r"\bthese\b",
    r"\bof those\b",
    r"\bof them\b",
)


_SQLISH_PATTERNS = (
    r"\bselect\b",
    r"\bfrom\b",
    r"\bjoin\b",
    r"\bwhere\b",
    r"\bgroup\s+by\b",
    r"\border\s+by\b",
    r"\bhaving\b",
    r"\bwith\s+[A-Za-z_][A-Za-z0-9_]*\s+as\b",
    r"\binterval\b",
)

_PLACEHOLDER_PATTERNS = (
    r"\ba named employee\b",
    r"\bbefore a date\b",
    r"\bafter a date\b",
    r"\ba building\b",
    r"\ba room\b",
)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return _clean_text(message.get("role")).lower()
    role = getattr(message, "role", "")
    return _clean_text(getattr(role, "value", role)).lower()


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return _clean_text(message.get("content") or message.get("text"))
    content = getattr(message, "content", None)
    if content is not None:
        return _clean_text(content)
    text = getattr(message, "text", None)
    if text is not None:
        return _clean_text(text)
    return ""


def _normalize_history(history: Iterable[Any] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in list(history or [])[-MAX_HISTORY_MESSAGES:]:
        role = _message_role(message)
        content = _message_content(message)
        if role in {"user", "assistant"} and content:
            normalized.append({"role": role, "content": content[:MAX_MESSAGE_CHARS]})
    return normalized


def _prior_user_messages(history: list[dict[str, str]]) -> list[str]:
    return [item["content"] for item in history if item["role"] == "user"]


def _looks_like_follow_up(query: str) -> bool:
    normalized = _clean_text(query)
    return bool(normalized) and any(
        re.search(pattern, normalized, flags=re.IGNORECASE)
        for pattern in _FOLLOW_UP_PATTERNS
    )


def _is_investigation_request(query: str) -> bool:
    query_lower = _clean_text(query).lower()
    return any(term in query_lower for term in _INVESTIGATION_TERMS)


def _contains_any(
    text: str,
    terms: tuple[str, ...],
) -> bool:
    lowered = _clean_text(text).lower()
    return any(term in lowered for term in terms)


def _is_analytical_request(query: str) -> bool:
    return _contains_any(
        query,
        _ANALYTICAL_OPERATION_TERMS,
    )


def _is_structural_request(query: str) -> bool:
    return _contains_any(
        query,
        _STRUCTURAL_OPERATION_TERMS,
    ) and not _is_analytical_request(query)


def _requires_policy_context(query: str) -> bool:
    return _contains_any(
        query,
        _POLICY_REFERENCE_TERMS,
    )


def _references_operational_data(query: str) -> bool:
    return _contains_any(
        query,
        _OPERATIONAL_DATA_TERMS,
    )


def _matched_semantic_tokens(
    text: str,
    patterns: tuple[str, ...],
) -> set[str]:
    cleaned = _clean_text(text).lower()
    matched: set[str] = set()

    for pattern in patterns:
        match = re.search(
            pattern,
            cleaned,
            flags=re.IGNORECASE,
        )

        if match is not None:
            matched.add(
                re.sub(
                    r"\s+",
                    " ",
                    match.group(0).strip().lower(),
                )
            )

    return matched


def _semantic_signature(
    text: str,
) -> dict[str, set[str]]:
    """
    Capture user-language semantics that QueryResolver is not allowed to lose.
    """
    return {
        "cardinality": _matched_semantic_tokens(
            text,
            _CARDINALITY_PATTERNS,
        ),
        "recency": _matched_semantic_tokens(
            text,
            _RECENCY_PATTERNS,
        ),
        "temporal_boundary": _matched_semantic_tokens(
            text,
            _TEMPORAL_BOUNDARY_PATTERNS,
        ),
        "aggregation": _matched_semantic_tokens(
            text,
            _AGGREGATION_PATTERNS,
        ),
    }


def _semantic_preservation_errors(
    *,
    current_query: str,
    resolved_query: str,
) -> list[str]:
    """
    Detect semantic tokens present in the current user message but lost during
    follow-up resolution.

    The current message is authoritative; prior context may resolve referents
    but must not erase these result-shape requirements.
    """
    current = _semantic_signature(current_query)
    resolved = _semantic_signature(resolved_query)

    errors: list[str] = []

    for category in (
        "cardinality",
        "recency",
        "temporal_boundary",
        "aggregation",
    ):
        missing = sorted(current[category] - resolved[category])

        if missing:
            errors.append(
                f"Resolver removed {category} requirement(s): "
                + ", ".join(missing)
                + "."
            )

    return errors


def _references_entity_set(
    text: str,
) -> bool:
    cleaned = _clean_text(text)

    return any(
        re.search(
            pattern,
            cleaned,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in _ENTITY_SET_REFERENCE_PATTERNS
    )


def _strip_entity_set_prefix(
    text: str,
) -> str:
    """
    Remove only a leading conversational entity-set bridge while preserving the
    actual operation.

    Example:
        "Of those employees, when was the last time each employee clocked in?"
        ->
        "when was the last time each employee clocked in?"

    The prior query is used separately to resolve what "those employees" means.
    """
    cleaned = _clean_text(text)

    cleaned = re.sub(
        r"^\s*(?:of\s+)?(?:those|them|these)\s+(?:employees|users|records|rows)"
        r"\s*[,;:]?\s*",
        "",
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"^\s*(?:of\s+)?(?:those|them|these)\s*[,;:]?\s*",
        "",
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    )

    return re.sub(
        r"\s+",
        " ",
        cleaned,
    ).strip()


def _deterministic_current_intent_resolution(
    current_query: str,
) -> str | None:
    """
    Preserve analytical or mixed current intent before prior context is used.
    """
    mixed_policy_data = _requires_policy_context(
        current_query
    ) and _references_operational_data(current_query)

    if not (_is_analytical_request(current_query) or mixed_policy_data):
        return None

    resolved = _strip_conversational_prefix(current_query)
    return resolved or None


def _strip_conversational_prefix(text: str) -> str:
    cleaned = _clean_text(text)
    changed = True
    while changed:
        changed = False
        for pattern in _SCHEMA_REFERENCE_PREFIX_PATTERNS:
            updated = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE).strip()
            if updated != cleaned:
                cleaned = updated
                changed = True
    return re.sub(r"\s+", " ", cleaned).strip()


def _strip_retry_language(text: str) -> str:
    cleaned = _strip_conversational_prefix(text)
    cleaned = re.sub(
        r"\b(?:please\s+)?try again\b[.!]?", "", cleaned, flags=re.IGNORECASE
    )
    return re.sub(r"\s+", " ", cleaned).strip(" ,.;")


def _contains_sqlish_output(text: str) -> bool:
    return any(
        re.search(pattern, text, flags=re.IGNORECASE) for pattern in _SQLISH_PATTERNS
    )


def _contains_invented_placeholder(resolved_query: str, source_text: str) -> bool:
    resolved_lower = resolved_query.lower()
    source_lower = source_text.lower()
    return any(
        re.search(pattern, resolved_lower, flags=re.IGNORECASE)
        and not re.search(pattern, source_lower, flags=re.IGNORECASE)
        for pattern in _PLACEHOLDER_PATTERNS
    )


def _contains_unintroduced_identifier(resolved_query: str, source_text: str) -> bool:
    source_ids = {
        token.lower()
        for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", source_text)
        if "_" in token
    }
    resolved_ids = {
        token.lower()
        for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", resolved_query)
        if "_" in token
    }
    return bool(resolved_ids - source_ids)


def _deterministic_investigation_resolution(
    current_query: str,
) -> str | None:
    """Backward-compatible wrapper for current-intent preservation."""
    return _deterministic_current_intent_resolution(current_query)


def _deterministic_follow_up_merge(
    prior_user_query: str, current_query: str
) -> str | None:
    prior = _clean_text(prior_user_query).rstrip("?.! ")
    current = _clean_text(current_query)
    if not prior or not current:
        return None

    # Entity-set follow-ups should preserve the CURRENT operation/result shape.
    # Prior context is used only to resolve the referenced set.
    if _references_entity_set(current):
        operation = _strip_entity_set_prefix(current)

        if operation and operation != current:
            prior_subject = prior

            # Avoid producing a giant analytical rewrite. The explicit current
            # operation remains authoritative.
            return (
                f"{operation.rstrip('?.! ')}. "
                f"The referenced entity set is the result set from: "
                f"{prior_subject}."
            )

    current_intent = _deterministic_current_intent_resolution(current)
    if current_intent:
        return current_intent

    only_match = re.match(
        r"^\s*only\s+(?:include\s+)?(.+?)[.!]?\s*$", current, flags=re.IGNORECASE
    )
    if only_match:
        return f"{prior}, limited to {only_match.group(1).strip()}."

    if "try again" in current.lower() or current.lower().startswith(("no", "but")):
        clarification = _strip_retry_language(current)
        if clarification:
            return f"{prior}. Apply this clarification: {clarification}."

    return None


def _extract_structured_response(response: Any) -> ResolvedQuery:
    raw = getattr(response, "raw", None)
    if isinstance(raw, ResolvedQuery):
        return raw
    if isinstance(raw, dict):
        return ResolvedQuery.model_validate(raw)
    parsed = (getattr(response, "additional_kwargs", {}) or {}).get("parsed")
    if isinstance(parsed, ResolvedQuery):
        return parsed
    if isinstance(parsed, dict):
        return ResolvedQuery.model_validate(parsed)
    text = _clean_text(getattr(response, "text", response))
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    if not text:
        raise ValueError("The query resolver returned an empty response.")
    return ResolvedQuery.model_validate_json(text)


class QueryResolver:
    """Resolve current messages while preserving investigative intent."""

    def __init__(self, llm: LLM) -> None:
        self._structured_llm = llm.as_structured_llm(ResolvedQuery)
        print("\n[QueryResolver] ⚙  Initialising operation-preserving resolver…")
        print(
            "[QueryResolver]    Framework : operation-preserving deterministic resolution + structured LLM fallback"
        )
        print(
            "[QueryResolver]    Distinction: "
            "structure vs records vs policy/data analysis"
        )
        print("[QueryResolver]    Tools     : none")
        print("[QueryResolver]    SQL       : prohibited")
        print(
            "[QueryResolver]    Preserves : "
            "all/each/every/per + last/latest/most recent + count/aggregate"
        )

    def resolve(
        self,
        current_query: str,
        history: Iterable[Any] | None = None,
    ) -> ResolvedQuery:
        cleaned_query = _clean_text(current_query)
        if not cleaned_query:
            return ResolvedQuery(
                is_follow_up=False,
                standalone_query="",
                clarification_used=False,
                reason="The current query is empty.",
            )

        normalized_history = _normalize_history(history)
        prior_user_queries = _prior_user_messages(normalized_history)
        follow_up_detected = bool(prior_user_queries) and _looks_like_follow_up(
            cleaned_query
        )

        current_intent_resolution = _deterministic_current_intent_resolution(
            cleaned_query
        )

        if current_intent_resolution:
            return ResolvedQuery(
                is_follow_up=follow_up_detected,
                standalone_query=current_intent_resolution,
                clarification_used=follow_up_detected,
                reason=(
                    "The current analytical or mixed evidence request was "
                    "preserved without copying a prior structural request."
                ),
            )

        if not follow_up_detected:
            return ResolvedQuery(
                is_follow_up=False,
                standalone_query=cleaned_query,
                clarification_used=False,
                reason="The message is already standalone or no prior user question is available.",
            )

        prior_user_query = prior_user_queries[-1]
        deterministic = _deterministic_follow_up_merge(prior_user_query, cleaned_query)
        if deterministic:
            semantic_errors = _semantic_preservation_errors(
                current_query=cleaned_query,
                resolved_query=deterministic,
            )

            if not semantic_errors:
                return ResolvedQuery(
                    is_follow_up=True,
                    standalone_query=deterministic,
                    clarification_used=True,
                    reason=(
                        "The follow-up was merged conservatively with the most "
                        "recent prior user question while preserving current "
                        "cardinality and ordering semantics."
                    ),
                )

        older_user_context = "\n\n".join(prior_user_queries[:-1]) or "(none)"
        prompt = _RESOLVER_PROMPT.format(
            current_query=cleaned_query,
            prior_user_query=prior_user_query,
            older_user_context=older_user_context,
        )

        try:
            resolved = _extract_structured_response(
                self._structured_llm.complete(prompt)
            )
        except Exception as exc:
            fallback = _strip_conversational_prefix(cleaned_query) or cleaned_query
            return ResolvedQuery(
                is_follow_up=True,
                standalone_query=fallback,
                clarification_used=True,
                reason=(
                    "Structured resolution failed, so the current request was preserved "
                    f"after removing conversational prefixes. {type(exc).__name__}: {exc}"
                ),
            )

        resolved_text = _clean_text(resolved.standalone_query)
        source_text = f"{prior_user_query}\n{cleaned_query}\n{older_user_context}"
        invalid_reason: str | None = None

        if not resolved_text:
            invalid_reason = "The structured resolver returned an empty query."
        elif _contains_sqlish_output(resolved_text):
            invalid_reason = "The structured resolver returned SQL-like content."
        elif _contains_unintroduced_identifier(resolved_text, source_text):
            invalid_reason = "The structured resolver introduced an identifier not present in the user conversation."
        elif _contains_invented_placeholder(resolved_text, source_text):
            invalid_reason = (
                "The structured resolver introduced an unsupported "
                "placeholder or criterion."
            )
        elif _is_analytical_request(cleaned_query) and _is_structural_request(
            resolved_text
        ):
            invalid_reason = (
                "The structured resolver replaced an analytical request "
                "with a structural request."
            )
        elif _requires_policy_context(cleaned_query) and not _requires_policy_context(
            resolved_text
        ):
            invalid_reason = (
                "The structured resolver removed an explicit policy or "
                "standards requirement."
            )

        semantic_errors = _semantic_preservation_errors(
            current_query=cleaned_query,
            resolved_query=resolved_text,
        )

        if invalid_reason is None and semantic_errors:
            invalid_reason = " ".join(semantic_errors)

        if invalid_reason:
            fallback = _strip_conversational_prefix(cleaned_query) or cleaned_query

            if _references_entity_set(cleaned_query):
                stripped_operation = _strip_entity_set_prefix(cleaned_query)

                if stripped_operation:
                    fallback = (
                        f"{stripped_operation.rstrip('?.! ')}. "
                        f"The referenced entity set is the result set from: "
                        f"{prior_user_query.rstrip('?.! ')}."
                    )

            return ResolvedQuery(
                is_follow_up=True,
                standalone_query=fallback,
                clarification_used=True,
                reason=(
                    f"{invalid_reason} The current user operation, cardinality, "
                    "and ordering semantics were preserved instead."
                ),
            )

        resolved.is_follow_up = True
        resolved.clarification_used = True
        return resolved
