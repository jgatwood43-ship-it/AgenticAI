"""
core/direct_query_checks.py
───────────────────────────
Compatibility layer for DirectQuery checks.

Semantic SQL reasoning now belongs to the LLM generation + self-review loop.
This module intentionally does not prescribe query shape, recency techniques,
aggregation techniques, joins, or result cardinality.

Hard SQL correctness is enforced by DirectSQLValidator:
- MySQL parsing;
- exactly one statement;
- read-only safety;
- documented tables;
- documented columns;
- documented relationships.

The class remains so existing imports do not break.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DirectQueryCheckIssue:
    code: str
    message: str


@dataclass
class DirectQueryCheckResult:
    is_valid: bool = True
    issues: list[DirectQueryCheckIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[str]:
        return [issue.message for issue in self.issues]


class DirectQueryChecks:
    """
    Do not perform semantic query-shape validation.

    The LLM is responsible for deciding how the SQL answers the user's request.
    Python remains the final authority for hard schema and safety constraints.
    """

    def validate(
        self,
        *,
        question: str,
        sql: str,
    ) -> DirectQueryCheckResult:
        return DirectQueryCheckResult(is_valid=True)


def _self_test() -> None:
    checker = DirectQueryChecks()

    result = checker.validate(
        question="Return the requested records.",
        sql="SELECT value FROM records",
    )

    assert result.is_valid
    assert result.issues == []

    print("DirectQueryChecks compatibility self-test: PASS")


if __name__ == "__main__":
    _self_test()
