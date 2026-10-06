"""
test_direct_sql_reasoning.py
────────────────────────────
Compare standalone DirectQuery reasoning against the production
DirectQueryPlanner prompt byte-for-byte.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llama_index.llms.ollama import Ollama

from core.direct_query_planner import (
    build_direct_query_generation_prompt,
    direct_query_prompt_sha256,
    direct_query_schema_sha256,
)
from core.direct_sql_validator import DirectSQLValidator
from core.schema_catalog import load_full_schema_catalog
from core.schema_graph import SchemaGraph

MODEL_NAME = "llama3.2"


def _response_text(response: Any) -> str:
    return str(getattr(response, "text", response) or "").strip()


def ask_llm(
    *,
    llm: Ollama,
    schema: str,
    question: str,
) -> str:
    prompt = build_direct_query_generation_prompt(
        question=question,
        schema_context=schema,
    )

    print(f"Prompt chars   : {len(prompt)}")
    print(f"Prompt SHA256  : {direct_query_prompt_sha256(prompt)}")
    print(f"Schema SHA256  : {direct_query_schema_sha256(schema)}")

    response = llm.complete(prompt)
    return _response_text(response)


def print_validation(validation: Any) -> None:
    print("\nValidation:")
    print(f"Valid         : {validation.is_valid}")
    print(f"Tables        : {validation.tables}")
    print(f"Columns       : {validation.columns}")
    print(f"Relationships : {validation.relationships}")

    if validation.errors:
        print("Errors:")
        for error in validation.errors:
            print(f"  - {error}")

    if validation.warnings:
        print("Warnings:")
        for warning in validation.warnings:
            print(f"  - {warning}")


def main() -> None:
    print("\n# Direct SQL Reasoning Test — Prompt Identity")
    print("=" * 70)

    schema = load_full_schema_catalog()
    print(f"Schema length : {len(schema)} chars")
    print(f"Schema SHA256 : {direct_query_schema_sha256(schema)}")

    project_root = Path(__file__).resolve().parent
    schema_directory = project_root / "docs" / "schema"

    schema_graph = SchemaGraph.from_directory(schema_directory)
    validator = DirectSQLValidator(schema_graph=schema_graph)

    print()
    print(schema_graph.summary())

    if schema_graph.warnings:
        print("\nSchemaGraph warnings:")
        for warning in schema_graph.warnings:
            print(f"  - {warning}")

    llm = Ollama(
        model=MODEL_NAME,
        request_timeout=120.0,
        temperature=0.0,
    )

    questions = [
        "List all employees.",
        "What is James Anderson's title?",
        "When was the last time James Anderson clocked in?",
        "For each employee, show their most recent clock-in time.",
    ]

    for number, question in enumerate(questions, start=1):
        print("\n" + "=" * 70)
        print(f"TEST {number}")
        print(f"Question: {question}")
        print("-" * 70)

        sql = ask_llm(
            llm=llm,
            schema=schema,
            question=question,
        )

        print("\nRaw LLM response:")
        print(sql)

        validation = validator.validate(
            sql,
            question=question,
        )

        print_validation(validation)

    bad_tests = [
        (
            "BAD RELATIONSHIP TEST",
            """SELECT
e.first_name,
jt.job_title_name
FROM employees AS e
JOIN job_titles AS jt
ON e.employee_id = jt.job_title_id;""",
        ),
        (
            "BAD COLUMN TEST",
            """SELECT
employee_name
FROM employees;""",
        ),
        (
            "BAD TABLE TEST",
            """SELECT
first_name
FROM employee_master;""",
        ),
        (
            "UNSAFE SQL TEST",
            """UPDATE employees
SET first_name = 'Test'
WHERE employee_id = 1;""",
        ),
    ]

    for title, sql in bad_tests:
        print("\n" + "=" * 70)
        print(title)
        print("-" * 70)
        print("\nSQL:")
        print(sql)
        print_validation(validator.validate(sql))

    print("\n" + "=" * 70)
    print("Test complete.")
    print("No SQL statements were executed against MySQL.")


if __name__ == "__main__":
    main()
