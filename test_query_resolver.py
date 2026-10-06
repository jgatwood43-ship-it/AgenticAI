"""
test_query_resolver.py
──────────────────────
Standalone validation test for core/query_resolver.py.

The test verifies that:

    * standalone questions remain unchanged;
    * follow-up clarifications become complete standalone queries;
    * explicit table names, column names, and business values are preserved;
    * the resolver does not answer the question or generate SQL.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.llm_factory import (
    build_llm,
    configure_llama_globals,
)
from core.query_resolver import QueryResolver

SUMMARY_WIDTH = 72


@dataclass(frozen=True)
class ResolverTestCase:
    """One query-resolution scenario."""

    name: str
    current_query: str
    history: list[dict[str, str]]
    expect_follow_up: bool
    required_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()


TEST_CASES = (
    ResolverTestCase(
        name="Standalone organizational-data question",
        current_query=("Provide a list of employee first and last names."),
        history=[],
        expect_follow_up=False,
        required_terms=(
            "employee",
            "first",
            "last",
        ),
    ),
    ResolverTestCase(
        name="Punch-type clarification",
        current_query=(
            "No, the time_clock table uses punch_type IN and OUT. "
            "Knowing that, try again."
        ),
        history=[
            {
                "role": "user",
                "content": (
                    "Are there any employees who gained access to a "
                    "secure room while not clocked in?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "I could not produce a verified answer because "
                    "clock status was not successfully derived."
                ),
            },
        ],
        expect_follow_up=True,
        required_terms=(
            "secure room",
            "time_clock",
            "punch_type",
            "IN",
            "OUT",
            "access",
        ),
        forbidden_terms=(
            "as stated earlier",
            "previous conversation",
        ),
    ),
    ResolverTestCase(
        name="Follow-up filter",
        current_query=("Only include access events after 8 PM."),
        history=[
            {
                "role": "user",
                "content": ("Who entered the secure rooms yesterday?"),
            },
            {
                "role": "assistant",
                "content": ("The prior request requires database evidence."),
            },
        ],
        expect_follow_up=True,
        required_terms=("secure",),
    ),
    ResolverTestCase(
        name="Investigation after schema review",
        current_query=(
            "OK, knowing this please review the records and determine "
            "if there are any anomalies that could be potential security "
            "risks. Provide as much detail as possible including employee "
            "full name and dates."
        ),
        history=[
            {
                "role": "user",
                "content": (
                    "Please provide the schema and relationships "
                    "of the company tables."
                ),
            },
            {
                "role": "assistant",
                "content": ("The schema and relationships were provided."),
            },
        ],
        expect_follow_up=True,
        required_terms=(
            "review",
            "anomal",
            "security",
            "employee",
            "dates",
        ),
        forbidden_terms=(
            "named employee",
            "before a date",
            "building whose name",
            "select ",
            " where ",
        ),
    ),
)


def _heading(
    title: str,
    character: str = "=",
) -> None:
    print("\n" + character * SUMMARY_WIDTH)
    print(title)
    print(character * SUMMARY_WIDTH)


def _validate_case(
    test_case: ResolverTestCase,
    standalone_query: str,
    is_follow_up: bool,
) -> list[str]:
    failures: list[str] = []

    if is_follow_up != test_case.expect_follow_up:
        failures.append(
            f"Expected is_follow_up={test_case.expect_follow_up}, "
            f"received {is_follow_up}."
        )

    normalized = standalone_query.lower()

    for term in test_case.required_terms:
        if term.lower() not in normalized:
            failures.append(f"Resolved query is missing required term: {term!r}.")

    for term in test_case.forbidden_terms:
        if term.lower() in normalized:
            failures.append(f"Resolved query contains forbidden phrase: {term!r}.")

    if not standalone_query.strip():
        failures.append("Resolver returned an empty standalone query.")

    # The resolver should produce a question/instruction, not executable SQL.
    if normalized.lstrip().startswith(("select ", "with ")):
        failures.append("Resolver generated SQL instead of a standalone question.")

    return failures


def main() -> None:
    _heading(
        "INITIALIZING QUERY RESOLVER",
        "#",
    )

    configure_llama_globals()

    resolver = QueryResolver(
        llm=build_llm(),
    )

    all_failures: list[tuple[str, list[str]]] = []

    for test_case in TEST_CASES:
        _heading(
            f"TEST: {test_case.name}",
            "#",
        )

        print("Current query:")
        print(test_case.current_query)
        print("\nHistory messages:")

        if test_case.history:
            for message in test_case.history:
                print(f"{message['role'].upper()}: " f"{message['content']}")
        else:
            print("(none)")

        result = resolver.resolve(
            current_query=test_case.current_query,
            history=test_case.history,
        )

        _heading(
            "RESOLUTION RESULT",
            "-",
        )

        print(f"Follow-up         : " f"{result.is_follow_up}")
        print(f"Clarification used: " f"{result.clarification_used}")
        print(f"Reason            : " f"{result.reason}")
        print("Standalone query:")
        print(result.standalone_query)

        failures = _validate_case(
            test_case=test_case,
            standalone_query=result.standalone_query,
            is_follow_up=result.is_follow_up,
        )

        if failures:
            all_failures.append(
                (
                    test_case.name,
                    failures,
                )
            )

            print("\nTEST RESULT: FAIL")

            for failure in failures:
                print(f"  - {failure}")
        else:
            print("\nTEST RESULT: PASS")

    _heading(
        "QUERY RESOLVER TEST SUMMARY",
        "#",
    )

    print(f"Test cases run : {len(TEST_CASES)}")
    print(f"Passed         : " f"{len(TEST_CASES) - len(all_failures)}")
    print(f"Failed         : " f"{len(all_failures)}")

    if all_failures:
        for name, failures in all_failures:
            print(f"\n{name}:")

            for failure in failures:
                print(f"  - {failure}")

        raise RuntimeError(f"{len(all_failures)} query resolver test(s) failed.")


if __name__ == "__main__":
    main()
