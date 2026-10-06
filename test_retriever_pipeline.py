"""
test_retriever_pipeline.py
──────────────────────────
Integration test for the revised evidence pipeline without Streamlit.

Pipeline under test
-------------------
RetrieverAgent
    Policy RAG, schema selection, logical SQL planning, one bounded repair,
    and MCP database execution.

GraderWriterAgent
    Evidence grading and terminal failure classification.

AnswerGeneratorAgent
    Final verified answer or transparent failure response.

This test intentionally bypasses LLMDecisionAgent and QueryResolver so it can
focus on Agents 2, 3, and 4.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from agents.answer_generator_agent import AnswerGeneratorAgent
from agents.grader_writer_agent import GraderWriterAgent
from agents.retriever_agent import RetrieverAgent
from core.llm_factory import (
    build_embed_model,
    build_llm,
    configure_llama_globals,
)
from core.schema_retriever import SchemaRetriever
from core.state import GradeResult, WorkflowState
from core.vector_store import (
    build_index,
    build_schema_index,
    build_vector_store,
)

SUMMARY_WIDTH = 72
EVIDENCE_PREVIEW_CHARS = 2000
CONTEXT_PREVIEW_CHARS = 3000


@dataclass(frozen=True)
class PipelineTestCase:
    """One integration-test scenario."""

    name: str
    query: str
    expect_database_success: bool | None = None
    expect_grade: GradeResult | None = None
    expect_answer: bool = True


TEST_CASES = (
    PipelineTestCase(
        name="Simple employee-name lookup",
        query=("Provide a list of employee first and last names."),
        expect_database_success=True,
        expect_grade=GradeResult.PASS,
    ),
    PipelineTestCase(
        name="Complex temporal access analysis",
        query=(
            "Are there any employees who gained access to a "
            "secure room while not clocked in?"
        ),
        # This may still fail until the schema documents define the exact
        # secure-room value and clock-status business values. The test confirms
        # that failure is controlled, classified, and explained.
        expect_database_success=None,
        expect_grade=None,
    ),
)


def _print_heading(
    title: str,
    character: str = "=",
) -> None:
    print("\n" + character * SUMMARY_WIDTH)
    print(title)
    print(character * SUMMARY_WIDTH)


def _preview(
    value: str,
    limit: int,
) -> str:
    cleaned = str(value or "").strip()

    if not cleaned:
        return "(none)"

    if len(cleaned) <= limit:
        return cleaned

    return cleaned[:limit] + f"\n... [truncated {len(cleaned) - limit} chars]"


def _failure_category(
    state: WorkflowState,
) -> str:
    return str(
        getattr(
            state,
            "evidence_failure_category",
            "",
        )
        or ""
    )


def _failure_reason(
    state: WorkflowState,
) -> str:
    return str(
        getattr(
            state,
            "evidence_failure_reason",
            "",
        )
        or ""
    )


def _failure_repairable(
    state: WorkflowState,
) -> bool | None:
    value = getattr(
        state,
        "evidence_failure_repairable",
        None,
    )

    if value is None:
        return None

    return bool(value)


def _print_result_summary(
    state: WorkflowState,
) -> None:
    _print_heading(
        "RESULT SUMMARY",
        "-",
    )

    sql_plan_valid = bool(state.sql_plan and state.sql_plan.is_valid)

    print(f"Policy sources        : " f"{state.policy_sources}")
    print(f"Policy context chars  : " f"{len(state.policy_context)}")
    print(f"Schema sources        : " f"{state.schema_sources}")
    print(f"Schema context chars  : " f"{len(state.schema_context)}")
    print(f"SQL plan valid        : " f"{sql_plan_valid}")
    print(f"SQL validation errors : " f"{state.sql_validation_errors}")
    print("SQL query:")
    print(state.sql_query or "(none)")
    print(f"Database success      : " f"{state.database_query_succeeded}")
    print(f"Database row count    : " f"{state.database_row_count}")
    print(f"Database columns      : " f"{state.database_columns}")
    print(f"Database error        : " f"{state.database_error}")
    print("Database evidence:")
    print(
        _preview(
            state.database_evidence,
            EVIDENCE_PREVIEW_CHARS,
        )
    )
    print(f"Combined context chars: " f"{len(state.retrieved_context)}")
    print(f"Grade                 : " f"{state.grade.value if state.grade else None}")
    print(f"Failure category      : " f"{_failure_category(state) or '(none)'}")
    print(f"Failure repairable    : " f"{_failure_repairable(state)}")
    print(f"Failure reason        : " f"{_failure_reason(state) or '(none)'}")
    print(f"ToT branches          : " f"{len(state.tot_thoughts)}")
    print(f"ToT winner            : " f"{state.tot_best_branch}")
    print(f"Refined context chars : " f"{len(state.refined_context)}")
    print("Refined context:")
    print(
        _preview(
            state.refined_context,
            CONTEXT_PREVIEW_CHARS,
        )
    )
    print(f"Final answer chars    : " f"{len(state.answer)}")
    print("Final answer:")
    print(state.answer or "(none)")


def _validate_test_case(
    test_case: PipelineTestCase,
    state: WorkflowState,
) -> list[str]:
    """Return assertion failures without stopping later test cases."""
    failures: list[str] = []

    if (
        test_case.expect_database_success is not None
        and state.database_query_succeeded != test_case.expect_database_success
    ):
        failures.append(
            "Expected database success="
            f"{test_case.expect_database_success}, "
            "received "
            f"{state.database_query_succeeded}."
        )

    if test_case.expect_grade is not None and state.grade != test_case.expect_grade:
        actual_grade = state.grade.value if state.grade else None

        failures.append(
            "Expected grade="
            f"{test_case.expect_grade.value}, "
            f"received {actual_grade}."
        )

    if test_case.expect_answer and not str(state.answer or "").strip():
        failures.append("AnswerGeneratorAgent produced no final answer.")

    if state.grade == GradeResult.PASS and not str(state.refined_context or "").strip():
        failures.append("Grade PASS was returned with empty refined_context.")

    if state.database_query_succeeded and state.database_error:
        failures.append("Database success is True while database_error is populated.")

    if (
        state.sql_plan
        and state.sql_plan.is_valid
        and not str(state.sql_query or "").strip()
    ):
        failures.append("SQL plan is valid but sql_query is empty.")

    return failures


async def run_test_case(
    test_case: PipelineTestCase,
    retriever_agent: RetrieverAgent,
    grader_agent: GraderWriterAgent,
    answer_agent: AnswerGeneratorAgent,
) -> tuple[WorkflowState, list[str]]:
    """Run one isolated test case through Agents 2, 3, and 4."""
    _print_heading(
        f"TEST: {test_case.name}",
        "#",
    )
    print(f"QUERY: {test_case.query}")

    # A fresh WorkflowState is mandatory for every test query.
    state = WorkflowState(
        query=test_case.query,
    )

    _print_heading("RUNNING RETRIEVER AGENT")
    state = await retriever_agent.run(state)

    _print_heading("RUNNING GRADER WRITER AGENT")
    state = await grader_agent.run(state)

    _print_heading("RUNNING ANSWER GENERATOR AGENT")
    state = await answer_agent.run(state)

    _print_result_summary(state)

    failures = _validate_test_case(
        test_case,
        state,
    )

    if failures:
        print("\nTEST RESULT: FAIL")

        for failure in failures:
            print(f"  - {failure}")
    else:
        print("\nTEST RESULT: PASS")

    return state, failures


async def main() -> None:
    """Initialize shared resources and run all integration cases."""
    _print_heading(
        "INITIALIZING TEST RESOURCES",
        "#",
    )

    configure_llama_globals()

    llm = build_llm()
    embed_model = build_embed_model()

    vector_store = build_vector_store()

    policy_index = build_index(
        vector_store,
        embed_model,
    )

    schema_index = build_schema_index(
        embed_model=embed_model,
    )

    schema_retriever = SchemaRetriever(
        schema_index=schema_index,
        similarity_top_k=8,
    )

    # Agent wrappers may be reused in this standalone test because the revised
    # agents do not retain ReAct workflow history. WorkflowState remains fresh
    # for every query.
    retriever_agent = RetrieverAgent(
        llm=llm,
        index=policy_index,
        schema_retriever=schema_retriever,
    )

    grader_agent = GraderWriterAgent(
        llm=llm,
        schema_retriever=schema_retriever,
    )

    answer_agent = AnswerGeneratorAgent(
        llm=llm,
        schema_retriever=schema_retriever,
    )

    all_failures: list[tuple[str, list[str]]] = []

    for test_case in TEST_CASES:
        _, failures = await run_test_case(
            test_case=test_case,
            retriever_agent=retriever_agent,
            grader_agent=grader_agent,
            answer_agent=answer_agent,
        )

        if failures:
            all_failures.append(
                (
                    test_case.name,
                    failures,
                )
            )

    _print_heading(
        "INTEGRATION TEST SUMMARY",
        "#",
    )

    print(f"Test cases run : {len(TEST_CASES)}")
    print(f"Passed         : " f"{len(TEST_CASES) - len(all_failures)}")
    print(f"Failed         : " f"{len(all_failures)}")

    if all_failures:
        for test_name, failures in all_failures:
            print(f"\n{test_name}:")

            for failure in failures:
                print(f"  - {failure}")

        raise RuntimeError(f"{len(all_failures)} integration test case(s) failed.")


if __name__ == "__main__":
    asyncio.run(main())
