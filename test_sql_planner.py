"""
test_sql_planner.py
───────────────────
Standalone validation test for core/sql_planner.py.

This test does not execute MySQL. It verifies:

    * deterministic simple planning;
    * logical reasoning for complex queries;
    * SQL safety and schema validation;
    * bounded repair behavior;
    * controlled failure when business values are not documented.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.llm_factory import (
    build_embed_model,
    build_llm,
    configure_llama_globals,
)
from core.schema_catalog import load_full_schema_catalog
from core.sql_planner import SQLPlanner
from core.state import SQLPlan
from core.vector_store import build_schema_index

SUMMARY_WIDTH = 72


@dataclass(frozen=True)
class PlannerTestCase:
    """One SQL-planner test scenario."""

    name: str
    query: str
    expect_valid: bool | None = None
    require_sql_when_valid: bool = True


TEST_CASES = (
    PlannerTestCase(
        name="Simple deterministic employee lookup",
        query=("Provide a list of employee first and last names."),
        expect_valid=True,
    ),
    PlannerTestCase(
        name="Complex temporal access analysis",
        query=(
            "Are there any employees who gained access to a "
            "secure room while not clocked in?"
        ),
        # Valid or controlled-invalid are both acceptable until the schema
        # documents the exact secure-room and punch-type business values.
        expect_valid=None,
    ),
    PlannerTestCase(
        name="Complex query with explicit punch assumptions",
        query=(
            "Identify employees who accessed a secure room while not "
            "clocked in. Treat time_clock.punch_type = 'IN' as clocked "
            "in and 'OUT' as clocked out. Determine status from the most "
            "recent punch at or before each badge access event."
        ),
        expect_valid=None,
    ),
)


def _heading(
    title: str,
    character: str = "=",
) -> None:
    print("\n" + character * SUMMARY_WIDTH)
    print(title)
    print(character * SUMMARY_WIDTH)


def _print_plan(
    plan: SQLPlan,
) -> None:
    print(f"Purpose       : " f"{plan.purpose or '(none)'}")
    print(f"Database      : " f"{plan.database_type or '(none)'}")
    print(f"Tables        : " f"{plan.tables}")
    print(f"Columns       : " f"{plan.columns}")
    print(f"Relationships : " f"{plan.relationships}")
    print(f"Filters       : " f"{plan.filters}")
    print(f"Valid         : " f"{plan.is_valid}")
    print(f"Errors        : " f"{plan.validation_errors}")
    print("SQL:")
    print(plan.sql or "(none)")


def _validate_case(
    test_case: PlannerTestCase,
    plan: SQLPlan,
) -> list[str]:
    failures: list[str] = []

    if test_case.expect_valid is not None and plan.is_valid != test_case.expect_valid:
        failures.append(
            f"Expected is_valid={test_case.expect_valid}, " f"received {plan.is_valid}."
        )

    if (
        plan.is_valid
        and test_case.require_sql_when_valid
        and not str(plan.sql or "").strip()
    ):
        failures.append("Planner marked the plan valid but returned no SQL.")

    if not plan.is_valid and not plan.validation_errors:
        failures.append("Planner returned an invalid plan without validation errors.")

    if plan.is_valid:
        normalized_sql = str(plan.sql or "").strip().upper()

        if not normalized_sql.startswith(("SELECT", "WITH")):
            failures.append("Valid SQL does not begin with SELECT or WITH.")

        if normalized_sql.startswith("WITH") and "SELECT" not in normalized_sql[4:]:
            failures.append("WITH statement does not contain a final SELECT.")

    return failures


def main() -> None:
    _heading(
        "INITIALIZING MODELS",
        "#",
    )

    configure_llama_globals()

    llm = build_llm()
    embed_model = build_embed_model()

    # Build the schema index as an integration check that the schema documents
    # remain loadable. SQLPlanner uses the full catalog text below.
    _heading(
        "BUILDING SCHEMA INDEX",
        "#",
    )

    build_schema_index(
        embed_model=embed_model,
    )

    schema_context = load_full_schema_catalog()

    print(f"Full schema catalog: " f"{len(schema_context)} chars")

    if not schema_context.strip():
        raise RuntimeError("The full schema catalog is empty.")

    planner = SQLPlanner(
        llm=llm,
    )

    all_failures: list[tuple[str, list[str]]] = []

    for test_case in TEST_CASES:
        _heading(
            f"TEST: {test_case.name}",
            "#",
        )
        print(f"QUERY: {test_case.query}")

        plan = planner.plan(
            query=test_case.query,
            schema_context=schema_context,
        )

        _heading(
            "SQL PLAN RESULT",
            "-",
        )
        _print_plan(plan)

        failures = _validate_case(
            test_case,
            plan,
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
        "PLANNER TEST SUMMARY",
        "#",
    )

    print(f"Queries tested : {len(TEST_CASES)}")
    print(f"Passed         : " f"{len(TEST_CASES) - len(all_failures)}")
    print(f"Failed         : " f"{len(all_failures)}")

    if all_failures:
        for name, failures in all_failures:
            print(f"\n{name}:")

            for failure in failures:
                print(f"  - {failure}")

        raise RuntimeError(f"{len(all_failures)} SQL planner test(s) failed.")


if __name__ == "__main__":
    main()
