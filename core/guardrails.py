from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from datetime import datetime, timezone
import re
import time


# ─────────────────────────────────────────────────────────────────────────────
# Guardrail result object
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    allowed: bool
    sanitized_value: Any = None
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    fallback_value: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


def allow(value: Any = None, warnings: list[str] | None = None, metadata: dict | None = None) -> GuardrailResult:
    return GuardrailResult(
        allowed=True,
        sanitized_value=value,
        warnings=warnings or [],
        metadata=metadata or {}
    )


def block(reason: str, fallback_value: Any = None, metadata: dict | None = None) -> GuardrailResult:
    return GuardrailResult(
        allowed=False,
        reason=reason,
        fallback_value=fallback_value,
        metadata=metadata or {}
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runtime monitoring
# ─────────────────────────────────────────────────────────────────────────────

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_timer() -> float:
    return time.perf_counter()


def elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def record_guardrail_event(
    state: Any,
    agent_name: str,
    check_name: str,
    result: GuardrailResult,
    elapsed_time_ms: int | None = None,
) -> None:
    """
    Adds guardrail events to state.guardrail_trace.
    """

    if not hasattr(state, "guardrail_trace") or state.guardrail_trace is None:
        state.guardrail_trace = []

    state.guardrail_trace.append(
        {
            "timestamp_utc": utc_now(),
            "agent": agent_name,
            "check": check_name,
            "allowed": result.allowed,
            "reason": result.reason,
            "warnings": result.warnings,
            "elapsed_ms": elapsed_time_ms,
            "metadata": result.metadata,
        }
    )


def record_runtime_event(
    state: Any,
    agent_name: str,
    event_name: str,
    details: dict[str, Any] | None = None,
) -> None:
    """
    Adds runtime monitoring events to state.runtime_trace.
    """

    if not hasattr(state, "runtime_trace") or state.runtime_trace is None:
        state.runtime_trace = []

    state.runtime_trace.append(
        {
            "timestamp_utc": utc_now(),
            "agent": agent_name,
            "event": event_name,
            "details": details or {},
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Input checks
# ─────────────────────────────────────────────────────────────────────────────

MAX_QUERY_LENGTH = 3000

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"forget\s+your\s+instructions",
    r"you\s+are\s+now",
    r"system\s+prompt",
    r"developer\s+message",
    r"reveal\s+your\s+prompt",
    r"bypass\s+security",
    r"disable\s+guardrails",
    r"act\s+as\s+an\s+unrestricted",
]


def validate_user_query(query: str) -> GuardrailResult:
    """
    Input validation before the query reaches the ReAct agent.
    """

    if query is None or not str(query).strip():
        return block("Query is empty.", fallback_value="")

    clean_query = str(query).strip()

    if len(clean_query) > MAX_QUERY_LENGTH:
        return block(
            "Query exceeds maximum length.",
            fallback_value=clean_query[:MAX_QUERY_LENGTH],
            metadata={"max_length": MAX_QUERY_LENGTH}
        )

    lowered = clean_query.lower()

    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return block(
                "Possible prompt injection detected in user query.",
                fallback_value=clean_query,
                metadata={"matched_pattern": pattern}
            )

    return allow(clean_query)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tool access validation and limits
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_TOOLS = {
    "mysql_query",
    "postgres_query",
}

MAX_SQL_LENGTH = 2000
MAX_TOOL_RESULT_LENGTH = 5000

BLOCKED_SQL_PATTERNS = [
    r"\bdrop\b",
    r"\bdelete\b",
    r"\bupdate\b",
    r"\binsert\b",
    r"\balter\b",
    r"\btruncate\b",
    r"\bcreate\b",
    r"\bgrant\b",
    r"\brevoke\b",
    r"\bexecute\b",
    r"\bcall\b",
    r"\bmerge\b",
]


def validate_tool_request(tool_name: str, tool_args: dict[str, Any]) -> GuardrailResult:
    """
    Validates a requested MCP/database tool call.
    """

    if tool_name not in ALLOWED_TOOLS:
        return block(
            f"Tool is not allowed: {tool_name}",
            metadata={"tool_name": tool_name}
        )

    sql = tool_args.get("query") or tool_args.get("sql") or ""

    if not sql:
        return block("Database tool call is missing SQL text.")

    sql = str(sql).strip()

    if len(sql) > MAX_SQL_LENGTH:
        return block(
            "SQL query exceeds maximum length.",
            metadata={"max_sql_length": MAX_SQL_LENGTH}
        )

    lowered = sql.lower()

    for pattern in BLOCKED_SQL_PATTERNS:
        if re.search(pattern, lowered):
            return block(
                "Blocked unsafe SQL operation.",
                metadata={"matched_pattern": pattern}
            )

    if not lowered.startswith(("select", "show", "describe", "with")):
        return block("Only read-only SQL queries are allowed.")

    warnings = []

    if "information_schema" not in lowered and "pg_catalog" not in lowered:
        warnings.append("SQL is read-only but not clearly limited to schema inspection.")

    return allow(tool_args, warnings=warnings)


def validate_tool_observation(observation: Any) -> GuardrailResult:
    """
    Validates MCP/database result before it is used as ReAct observation.
    """

    text = str(observation)

    warnings = []

    if len(text) > MAX_TOOL_RESULT_LENGTH:
        text = text[:MAX_TOOL_RESULT_LENGTH]
        warnings.append("Tool result was truncated.")

    lowered = text.lower()

    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            text = re.sub(pattern, "[REDACTED_INSTRUCTION]", text, flags=re.IGNORECASE)
            warnings.append("Possible prompt injection removed from tool result.")

    return allow(text, warnings=warnings)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Source verification
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_SOURCE_FIELDS = {
    "source_id",
    "source_type",
    "created_at",
}

ALLOWED_SOURCE_TYPES = {
    "schema",
    "policy",
    "access_log",
    "role_definition",
    "document_chunk",
}


def verify_source(source: dict[str, Any]) -> GuardrailResult:
    """
    Verifies one source/document/log record.
    """

    if not isinstance(source, dict):
        return block("Source is not a dictionary.")

    missing = REQUIRED_SOURCE_FIELDS - set(source.keys())

    if missing:
        return block(
            "Source missing required metadata.",
            metadata={"missing_fields": list(missing)}
        )

    source_type = source.get("source_type")

    if source_type not in ALLOWED_SOURCE_TYPES:
        return block(
            "Source type is not allowed.",
            metadata={"source_type": source_type}
        )

    return allow(source)


def verify_sources(sources: list[dict[str, Any]]) -> GuardrailResult:
    """
    Verifies a list of retrieved sources.
    """

    verified = []
    rejected = []

    for source in sources:
        check = verify_source(source)

        if check.allowed:
            verified.append(check.sanitized_value)
        else:
            rejected.append(
                {
                    "source": source,
                    "reason": check.reason,
                    "metadata": check.metadata,
                }
            )

    if not verified and sources:
        return block(
            "No retrieved sources passed verification.",
            fallback_value=[],
            metadata={"rejected_count": len(rejected)}
        )

    return allow(
        verified,
        warnings=[f"{len(rejected)} source(s) rejected."] if rejected else [],
        metadata={
            "verified_count": len(verified),
            "rejected_count": len(rejected),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Output constraints
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_ROUTES = {
    "retriever",
    "answer_generator",
}

BLOCKED_OUTPUT_PHRASES = [
    "the user is malicious",
    "the employee is guilty",
    "this proves misconduct",
    "this proves data theft",
    "terminate the employee",
    "fire the employee",
    "disable the account immediately",
]


def validate_route_output(response_text: str) -> GuardrailResult:
    """
    Validates the final route emitted by the LLM decision agent.
    """

    if not response_text:
        return block(
            "Empty route output.",
            fallback_value="ROUTE: retriever"
        )

    matches = re.findall(
        r"ROUTE:\s*(retriever|answer_generator)",
        response_text,
        flags=re.IGNORECASE
    )

    if len(matches) == 1:
        route = matches[0].lower()

        if route not in ALLOWED_ROUTES:
            return block(
                "Route is not allowed.",
                fallback_value="ROUTE: retriever"
            )

        return allow(f"ROUTE: {route}")

    if len(matches) > 1:
        return block(
            "Multiple route tokens found.",
            fallback_value="ROUTE: retriever"
        )

    return block(
        "No valid route token found.",
        fallback_value="ROUTE: retriever"
    )


def sanitize_final_text(text: str) -> GuardrailResult:
    """
    Applies general safety constraints to generated text.
    Useful for AnswerGeneratorAgent and GraderWriterAgent.
    """

    if text is None:
        return block("Output text is empty.", fallback_value="")

    safe_text = str(text)

    lowered = safe_text.lower()

    for phrase in BLOCKED_OUTPUT_PHRASES:
        if phrase in lowered:
            return block(
                "Unsafe conclusion or prohibited action found in output.",
                fallback_value=safe_text.replace(phrase, "[REMOVED_UNSUPPORTED_CONCLUSION]"),
                metadata={"blocked_phrase": phrase}
            )

    return allow(safe_text)


def require_evidence_for_finding(finding: dict[str, Any]) -> GuardrailResult:
    """
    Requires security findings to have evidence before being escalated.
    """

    required = ["summary", "evidence", "risk_level", "confidence"]

    missing = [field for field in required if field not in finding]

    if missing:
        return block(
            "Finding is missing required fields.",
            metadata={"missing_fields": missing}
        )

    evidence = finding.get("evidence") or []

    if not evidence:
        return block("Finding has no supporting evidence.")

    return allow(finding)