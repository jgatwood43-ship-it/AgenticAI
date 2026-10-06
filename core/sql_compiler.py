"""
core/sql_compiler.py
────────────────────
Deterministic MySQL compiler for LogicalQueryPlan.

Completeness guarantees
-----------------------
The compiler rejects a plan when:

    * a required table is not the base table, a join participant, an existence
      predicate participant, a field-comparison participant, or a temporal
      operation participant;

    * required tables form disconnected executable components;

    * a required output field is not projected by selected_fields;

    * a selected field references a table that is not actually used;

    * a declared join or additional join condition does not appear in the
      compiled SQL;

    * a field comparison or existence predicate references undocumented schema;

    * a filter, grouping rule, ordering rule, or temporal dependency references
      undocumented schema.

The compiler does not infer business meaning and does not call an LLM.

For raw investigation SQL, deterministic validation recursively verifies
physical schema references in every SELECT/subquery and safety but does not veto
analytical cross-table correlations merely because they are not declared
SchemaGraph relationships.
"""

from __future__ import annotations

import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
from typing import Any

from core.logical_plan import (
    AggregatePredicate,
    Aggregation,
    ComparisonOperator,
    DerivedPredicate,
    ExistencePredicate,
    FieldComparison,
    FilterOperator,
    JoinCondition,
    LogicalFilter,
    LogicalJoin,
    LogicalQueryPlan,
    TemporalOperation,
)
from core.state import SQLPlan
from core.schema_model import SchemaCatalog

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ============================================================================
# General helpers
# ============================================================================


def _clean_text(
    value: Any,
) -> str:
    return str(value or "").strip()


def _unique_strings(
    values: list[str],
) -> list[str]:
    result: list[str] = []

    for value in values:
        cleaned = _clean_text(value)

        if cleaned and cleaned not in result:
            result.append(cleaned)

    return result


def _safe_identifier(
    value: str,
) -> str:
    cleaned = _clean_text(value)

    if not _IDENTIFIER_PATTERN.fullmatch(cleaned):
        raise ValueError(f"Unsafe SQL identifier: {value!r}")

    return cleaned


def _sql_literal(
    value: Any,
) -> str:
    if value is None:
        return "NULL"

    if isinstance(
        value,
        bool,
    ):
        return "1" if value else "0"

    if isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        return str(value)

    escaped = str(value).replace(
        "'",
        "''",
    )

    return f"'{escaped}'"


def _operator_sql(
    operator: FilterOperator,
) -> str:
    mapping = {
        FilterOperator.EQ: "=",
        FilterOperator.NE: "<>",
        FilterOperator.GT: ">",
        FilterOperator.GTE: ">=",
        FilterOperator.LT: "<",
        FilterOperator.LTE: "<=",
        FilterOperator.LIKE: "LIKE",
    }

    if operator not in mapping:
        raise ValueError("Unsupported scalar operator: " f"{operator.value}")

    return mapping[operator]


def _comparison_operator_sql(
    operator: ComparisonOperator,
) -> str:
    mapping = {
        ComparisonOperator.EQ: "=",
        ComparisonOperator.NE: "<>",
        ComparisonOperator.GT: ">",
        ComparisonOperator.GTE: ">=",
        ComparisonOperator.LT: "<",
        ComparisonOperator.LTE: "<=",
    }

    if operator not in mapping:
        raise ValueError("Unsupported field comparison operator: " f"{operator.value}")

    return mapping[operator]


def _join_condition_operator_sql(
    operator: str,
) -> str:
    normalized = _clean_text(operator).lower()

    mapping = {
        "eq": "=",
        "ne": "<>",
    }

    if normalized not in mapping:
        raise ValueError("Unsupported join-condition operator: " f"{operator!r}")

    return mapping[normalized]


def _field_parts(
    field: str,
) -> tuple[
    str,
    str,
]:
    cleaned = _clean_text(field).strip("`")

    parts = cleaned.split(
        ".",
        1,
    )

    if len(parts) != 2:
        raise ValueError("Field must use table.column notation: " f"{field!r}")

    table = _safe_identifier(parts[0]).lower()

    column = _safe_identifier(parts[1]).lower()

    return (
        table,
        column,
    )


# ============================================================================
# Plan completion and field normalization
# ============================================================================


def _complete_required_tables(
    plan: LogicalQueryPlan,
) -> None:
    """
    Add every explicitly referenced table to required_tables.
    """
    tables: list[str] = [
        plan.base_table,
        *plan.required_tables,
    ]

    for join in plan.joins:
        tables.extend(
            [
                join.left_table,
                join.right_table,
            ]
        )

    for comparison in plan.field_comparisons:
        tables.extend(
            [
                comparison.left_table,
                comparison.right_table,
            ]
        )

    for predicate in plan.existence_predicates:
        tables.extend(
            [
                predicate.left_table,
                predicate.right_table,
            ]
        )

    for item in plan.filters:
        tables.append(item.table)

    for aggregation in plan.aggregations:
        tables.append(aggregation.table)

    for ordering in plan.order_by:
        tables.append(ordering.table)

    for operation in plan.temporal_operations:
        tables.extend(
            [
                operation.source_table,
                operation.reference_table,
            ]
        )

        for correlation in operation.correlation_keys:
            tables.extend(
                [
                    correlation.source_table,
                    correlation.reference_table,
                ]
            )

    plan.base_table = _clean_text(plan.base_table).lower()

    plan.required_tables = _unique_strings(
        [table.lower() for table in tables if _clean_text(table)]
    )


def _candidate_tables(
    plan: LogicalQueryPlan,
) -> list[str]:
    return _unique_strings(
        [
            plan.base_table,
            *plan.required_tables,
        ]
    )


def _qualify_field(
    field: str,
    schema: SchemaCatalog,
    candidate_tables: list[str],
) -> tuple[
    str | None,
    str | None,
]:
    """
    Normalize a field to table.column notation.

    An unqualified field is accepted only when exactly one candidate table
    owns the column.
    """
    cleaned = _clean_text(field).strip("`")

    if not cleaned:
        return (
            None,
            "A logical-plan field is empty.",
        )

    if "." in cleaned:
        try:
            (
                table,
                column,
            ) = _field_parts(cleaned)

        except ValueError as exc:
            return (
                None,
                str(exc),
            )

        if not schema.has_table(table):
            return (
                None,
                f"Undocumented table in field: {table}.",
            )

        if not schema.has_column(
            table,
            column,
        ):
            return (
                None,
                f"Undocumented column: {table}.{column}.",
            )

        return (
            f"{table}.{column}",
            None,
        )

    column = cleaned.lower()

    matching_tables = [
        table
        for table in candidate_tables
        if schema.has_column(
            table,
            column,
        )
    ]

    if len(matching_tables) == 1:
        return (
            f"{matching_tables[0]}.{column}",
            None,
        )

    if not matching_tables:
        return (
            None,
            f"Undocumented unqualified column: {column}.",
        )

    return (
        None,
        (
            f"Ambiguous unqualified column: {column}. "
            f"Possible tables: {matching_tables}."
        ),
    )


def _normalize_field_list(
    fields: list[str],
    schema: SchemaCatalog,
    candidate_tables: list[str],
) -> tuple[
    list[str],
    list[str],
]:
    normalized_fields: list[str] = []
    errors: list[str] = []

    for field in fields:
        (
            normalized,
            error,
        ) = _qualify_field(
            field=field,
            schema=schema,
            candidate_tables=candidate_tables,
        )

        if error:
            errors.append(error)

        elif normalized and normalized not in normalized_fields:
            normalized_fields.append(normalized)

    return (
        normalized_fields,
        _unique_strings(errors),
    )


def _normalize_plan_fields(
    plan: LogicalQueryPlan,
    schema: SchemaCatalog,
) -> list[str]:
    """
    Normalize selected, required-output, and group-by fields in-place.
    """
    errors: list[str] = []

    candidates = _candidate_tables(plan)

    (
        plan.selected_fields,
        selected_errors,
    ) = _normalize_field_list(
        fields=plan.selected_fields,
        schema=schema,
        candidate_tables=candidates,
    )

    errors.extend(selected_errors)

    (
        plan.required_output_fields,
        required_output_errors,
    ) = _normalize_field_list(
        fields=plan.required_output_fields,
        schema=schema,
        candidate_tables=candidates,
    )

    errors.extend(required_output_errors)

    (
        plan.group_by,
        group_errors,
    ) = _normalize_field_list(
        fields=plan.group_by,
        schema=schema,
        candidate_tables=candidates,
    )

    errors.extend(group_errors)

    return _unique_strings(errors)


# ============================================================================
# Completeness validation
# ============================================================================


def _participating_tables(
    plan: LogicalQueryPlan,
) -> set[str]:
    """
    Return tables that actually participate in the executable plan.
    """
    participating = {plan.base_table.lower()}

    for join in plan.joins:
        participating.add(join.left_table.lower())
        participating.add(join.right_table.lower())

    for comparison in plan.field_comparisons:
        participating.add(comparison.left_table.lower())
        participating.add(comparison.right_table.lower())

    for predicate in plan.existence_predicates:
        participating.add(predicate.left_table.lower())
        participating.add(predicate.right_table.lower())

    for operation in plan.temporal_operations:
        participating.add(operation.source_table.lower())
        participating.add(operation.reference_table.lower())

        for correlation in operation.correlation_keys:
            participating.add(correlation.source_table.lower())
            participating.add(correlation.reference_table.lower())

    return participating


def _validate_required_table_usage(
    plan: LogicalQueryPlan,
) -> list[str]:
    """
    Reject required tables that never participate in executable logic.
    """
    participating = _participating_tables(plan)

    return [
        ("Required table is not used by the executable plan: " f"{table}.")
        for table in plan.required_tables
        if table.lower() not in participating
    ]


def _validate_required_outputs(
    plan: LogicalQueryPlan,
) -> list[str]:
    """
    Ensure every required evidence field appears in selected_fields.
    """
    selected = {field.lower() for field in plan.selected_fields}

    return [
        ("Required output field is missing from selected_fields: " f"{field}.")
        for field in plan.required_output_fields
        if field.lower() not in selected
    ]


def _validate_selected_table_usage(
    plan: LogicalQueryPlan,
) -> list[str]:
    """
    Reject selected fields from tables that are not executable participants.
    """
    participating = _participating_tables(plan)

    errors: list[str] = []

    for field in plan.selected_fields:
        (
            table,
            _,
        ) = _field_parts(field)

        if table not in participating:
            errors.append(
                (
                    "Selected field references a table that is not used "
                    f"by the executable plan: {field}."
                )
            )

    return errors


def _build_join_graph(
    plan: LogicalQueryPlan,
) -> dict[
    str,
    set[str],
]:
    graph: dict[
        str,
        set[str],
    ] = {}

    for table in plan.required_tables:
        graph.setdefault(
            table.lower(),
            set(),
        )

    graph.setdefault(
        plan.base_table.lower(),
        set(),
    )

    for join in plan.joins:
        left = join.left_table.lower()
        right = join.right_table.lower()

        graph.setdefault(
            left,
            set(),
        ).add(right)

        graph.setdefault(
            right,
            set(),
        ).add(left)

    for comparison in plan.field_comparisons:
        left = comparison.left_table.lower()
        right = comparison.right_table.lower()

        graph.setdefault(
            left,
            set(),
        ).add(right)

        graph.setdefault(
            right,
            set(),
        ).add(left)

    for predicate in plan.existence_predicates:
        left = predicate.left_table.lower()
        right = predicate.right_table.lower()

        graph.setdefault(
            left,
            set(),
        ).add(right)

        graph.setdefault(
            right,
            set(),
        ).add(left)

    for operation in plan.temporal_operations:
        source = operation.source_table.lower()
        reference = operation.reference_table.lower()

        graph.setdefault(
            source,
            set(),
        ).add(reference)

        graph.setdefault(
            reference,
            set(),
        ).add(source)

    return graph


def _validate_join_connectivity(
    plan: LogicalQueryPlan,
) -> list[str]:
    """
    Ensure every participating required table is connected to the base table.
    """
    graph = _build_join_graph(plan)

    base = plan.base_table.lower()

    visited: set[str] = set()
    stack = [base]

    while stack:
        current = stack.pop()

        if current in visited:
            continue

        visited.add(current)

        stack.extend(
            graph.get(
                current,
                set(),
            )
            - visited
        )

    participating = _participating_tables(plan)

    disconnected = sorted(table for table in participating if table not in visited)

    if not disconnected:
        return []

    return [
        (
            "Required table is disconnected from the base table "
            f"{plan.base_table}: {table}."
        )
        for table in disconnected
    ]


# ============================================================================
# Schema validation
# ============================================================================


def _relationship_is_documented(
    schema: SchemaCatalog,
    left_table: str,
    left_column: str,
    right_table: str,
    right_column: str,
) -> bool:
    relationship = (
        left_table.lower(),
        left_column.lower(),
        right_table.lower(),
        right_column.lower(),
    )

    reverse = (
        right_table.lower(),
        right_column.lower(),
        left_table.lower(),
        left_column.lower(),
    )

    return relationship in schema.relationships or reverse in schema.relationships


def _validate_join_condition(
    *,
    schema: SchemaCatalog,
    left_table: str,
    right_table: str,
    condition: JoinCondition,
    require_documented_relationship: bool,
) -> list[str]:
    errors: list[str] = []

    left_column = condition.left_column.lower()
    right_column = condition.right_column.lower()

    if not schema.has_column(
        left_table,
        left_column,
    ):
        errors.append(("Undocumented join column: " f"{left_table}.{left_column}."))

    if not schema.has_column(
        right_table,
        right_column,
    ):
        errors.append(("Undocumented join column: " f"{right_table}.{right_column}."))

    if (
        require_documented_relationship
        and condition.operator == "eq"
        and schema.has_column(
            left_table,
            left_column,
        )
        and schema.has_column(
            right_table,
            right_column,
        )
        and not _relationship_is_documented(
            schema=schema,
            left_table=left_table,
            left_column=left_column,
            right_table=right_table,
            right_column=right_column,
        )
    ):
        errors.append(
            (
                "Undocumented relationship: "
                f"{left_table}.{left_column} = "
                f"{right_table}.{right_column}."
            )
        )

    return errors


def _validate_schema_references(
    plan: LogicalQueryPlan,
    schema: SchemaCatalog,
) -> list[str]:
    errors: list[str] = []

    for table in _unique_strings(
        [
            plan.base_table,
            *plan.required_tables,
        ]
    ):
        if not schema.has_table(table):
            errors.append(f"Undocumented table: {table}.")

    for field in [
        *plan.selected_fields,
        *plan.required_output_fields,
        *plan.group_by,
    ]:
        try:
            (
                table,
                column,
            ) = _field_parts(field)

        except ValueError as exc:
            errors.append(str(exc))
            continue

        if not schema.has_column(
            table,
            column,
        ):
            errors.append(f"Undocumented column: {table}.{column}.")

    # Standard joins.
    for join in plan.joins:
        left_table = join.left_table.lower()
        right_table = join.right_table.lower()

        for condition in join.all_conditions():
            errors.extend(
                _validate_join_condition(
                    schema=schema,
                    left_table=left_table,
                    right_table=right_table,
                    condition=condition,
                    require_documented_relationship=True,
                )
            )

    # Field-to-field comparisons.
    for comparison in plan.field_comparisons:
        for (
            table,
            column,
        ) in (
            (
                comparison.left_table,
                comparison.left_column,
            ),
            (
                comparison.right_table,
                comparison.right_column,
            ),
        ):
            if not schema.has_column(
                table,
                column,
            ):
                errors.append(
                    ("Undocumented field-comparison column: " f"{table}.{column}.")
                )

    # EXISTS / NOT EXISTS relationship predicates.
    for predicate in plan.existence_predicates:
        left_table = predicate.left_table.lower()
        right_table = predicate.right_table.lower()

        for condition in predicate.all_conditions():
            errors.extend(
                _validate_join_condition(
                    schema=schema,
                    left_table=left_table,
                    right_table=right_table,
                    condition=condition,
                    require_documented_relationship=True,
                )
            )

    for item in plan.filters:
        if not schema.has_column(
            item.table,
            item.column,
        ):
            errors.append(
                ("Undocumented filter column: " f"{item.table}.{item.column}.")
            )

    for aggregation in plan.aggregations:
        if aggregation.column != "*" and not schema.has_column(
            aggregation.table,
            aggregation.column,
        ):
            errors.append(
                (
                    "Undocumented aggregation column: "
                    f"{aggregation.table}.{aggregation.column}."
                )
            )

    aggregation_aliases = {
        _clean_text(aggregation.alias).lower(): aggregation
        for aggregation in plan.aggregations
        if _clean_text(aggregation.alias)
    }

    for predicate in plan.aggregate_predicates:
        alias = _clean_text(predicate.aggregation_alias).lower()

        if not alias:
            errors.append("Aggregate predicate references an empty aggregation alias.")
            continue

        if alias not in aggregation_aliases:
            errors.append(
                "Aggregate predicate references unknown aggregation alias: "
                f"{predicate.aggregation_alias}."
            )

    for ordering in plan.order_by:
        if not schema.has_column(
            ordering.table,
            ordering.column,
        ):
            errors.append(
                (
                    "Undocumented ordering column: "
                    f"{ordering.table}.{ordering.column}."
                )
            )

    temporal_aliases: set[str] = set()

    for operation in plan.temporal_operations:
        temporal_aliases.add(operation.output_alias)

        if not schema.has_column(
            operation.source_table,
            operation.source_timestamp_column,
        ):
            errors.append(
                (
                    "Undocumented temporal source column: "
                    f"{operation.source_table}."
                    f"{operation.source_timestamp_column}."
                )
            )

        if not schema.has_column(
            operation.reference_table,
            operation.reference_timestamp_column,
        ):
            errors.append(
                (
                    "Undocumented temporal reference column: "
                    f"{operation.reference_table}."
                    f"{operation.reference_timestamp_column}."
                )
            )

        if operation.tie_breaker_column and not schema.has_column(
            operation.source_table,
            operation.tie_breaker_column,
        ):
            errors.append(
                (
                    "Undocumented temporal tie-breaker column: "
                    f"{operation.source_table}."
                    f"{operation.tie_breaker_column}."
                )
            )

        for correlation in operation.correlation_keys:
            for (
                table,
                column,
            ) in (
                (
                    correlation.source_table,
                    correlation.source_column,
                ),
                (
                    correlation.reference_table,
                    correlation.reference_column,
                ),
            ):
                if not schema.has_column(
                    table,
                    column,
                ):
                    errors.append(
                        (
                            "Undocumented temporal correlation column: "
                            f"{table}.{column}."
                        )
                    )

        for column in operation.selected_columns:
            if not schema.has_column(
                operation.source_table,
                column,
            ):
                errors.append(
                    (
                        "Undocumented temporal selected column: "
                        f"{operation.source_table}.{column}."
                    )
                )

    for predicate in plan.derived_predicates:
        if predicate.temporal_alias not in temporal_aliases:
            errors.append(
                (
                    "Derived predicate references unknown temporal alias: "
                    f"{predicate.temporal_alias}."
                )
            )

    return _unique_strings(errors)


def _validate_plan(
    plan: LogicalQueryPlan,
    schema: SchemaCatalog,
) -> list[str]:
    """
    Run schema, connectivity, usage, and output completeness checks.
    """
    errors: list[str] = []

    errors.extend(
        _validate_schema_references(
            plan,
            schema,
        )
    )

    errors.extend(_validate_required_table_usage(plan))

    errors.extend(_validate_join_connectivity(plan))

    errors.extend(_validate_selected_table_usage(plan))

    errors.extend(_validate_required_outputs(plan))

    if plan.aggregate_predicates and not plan.aggregations:
        errors.append("Aggregate predicates require at least one declared aggregation.")

    aggregation_aliases = {
        _clean_text(aggregation.alias).lower()
        for aggregation in plan.aggregations
        if _clean_text(aggregation.alias)
    }

    for predicate in plan.aggregate_predicates:
        alias = _clean_text(predicate.aggregation_alias).lower()

        if alias not in aggregation_aliases:
            errors.append(
                "Aggregate predicate references unknown aggregation alias: "
                f"{predicate.aggregation_alias}."
            )

    return _unique_strings(errors)


# ============================================================================
# SQL compilation helpers
# ============================================================================


def _compile_filter(
    item: LogicalFilter,
    aliases: dict[str, str],
) -> str:
    table = item.table.lower()
    alias = aliases[table]

    column = _safe_identifier(item.column)

    reference = f"{alias}.{column}"

    if item.operator == FilterOperator.IS_NULL:
        return f"{reference} IS NULL"

    if item.operator == FilterOperator.IS_NOT_NULL:
        return f"{reference} IS NOT NULL"

    if item.operator in {
        FilterOperator.IN,
        FilterOperator.NOT_IN,
    }:
        if (
            not isinstance(
                item.value,
                list,
            )
            or not item.value
        ):
            raise ValueError(
                (f"{item.operator.value} requires " "a non-empty list value.")
            )

        values = ", ".join(_sql_literal(value) for value in item.value)

        keyword = "IN" if item.operator == FilterOperator.IN else "NOT IN"

        return f"{reference} {keyword} ({values})"

    return (
        f"{reference} " f"{_operator_sql(item.operator)} " f"{_sql_literal(item.value)}"
    )


def _compile_field_comparison(
    item: FieldComparison,
    aliases: dict[str, str],
) -> str:
    left_table = item.left_table.lower()
    right_table = item.right_table.lower()

    left_column = _safe_identifier(item.left_column)
    right_column = _safe_identifier(item.right_column)

    return (
        f"{aliases[left_table]}.{left_column} "
        f"{_comparison_operator_sql(item.operator)} "
        f"{aliases[right_table]}.{right_column}"
    )


def _compile_join_condition(
    *,
    left_alias: str,
    right_alias: str,
    condition: JoinCondition,
) -> str:
    return (
        f"{left_alias}."
        f"{_safe_identifier(condition.left_column)} "
        f"{_join_condition_operator_sql(condition.operator)} "
        f"{right_alias}."
        f"{_safe_identifier(condition.right_column)}"
    )


def _compile_logical_join(
    join: LogicalJoin,
    aliases: dict[str, str],
) -> str:
    left_table = join.left_table.lower()
    right_table = join.right_table.lower()

    conditions = [
        _compile_join_condition(
            left_alias=aliases[left_table],
            right_alias=aliases[right_table],
            condition=condition,
        )
        for condition in join.all_conditions()
    ]

    if not conditions:
        raise ValueError(
            (
                "Logical join has no executable conditions: "
                f"{left_table} -> {right_table}."
            )
        )

    return (
        f"{join.join_type} JOIN "
        f"{_safe_identifier(right_table)} "
        f"AS {aliases[right_table]} ON " + " AND ".join(conditions)
    )


def _compile_existence_predicate(
    item: ExistencePredicate,
    aliases: dict[str, str],
) -> str:
    left_table = item.left_table.lower()
    right_table = item.right_table.lower()

    left_alias = aliases[left_table]

    sub_alias = f"ex_{_safe_identifier(right_table)}"

    conditions: list[str] = []

    for condition in item.all_conditions():
        operator_sql = _join_condition_operator_sql(condition.operator)

        conditions.append(
            (
                f"{sub_alias}."
                f"{_safe_identifier(condition.right_column)} "
                f"{operator_sql} "
                f"{left_alias}."
                f"{_safe_identifier(condition.left_column)}"
            )
        )

    if not conditions:
        raise ValueError(
            (
                "Existence predicate has no relationship conditions: "
                f"{left_table} -> {right_table}."
            )
        )

    keyword = "EXISTS" if item.must_exist else "NOT EXISTS"

    return (
        f"{keyword} (\n"
        f"        SELECT 1\n"
        f"        FROM {_safe_identifier(right_table)} AS {sub_alias}\n"
        f"        WHERE " + "\n          AND ".join(conditions) + "\n    )"
    )


def _aggregation_expression(
    aggregation: Aggregation,
    aliases: dict[str, str],
) -> str:
    """Compile an aggregation expression without its output alias."""
    alias = aliases[aggregation.table.lower()]

    column = (
        "*"
        if aggregation.column == "*"
        else (f"{alias}." f"{_safe_identifier(aggregation.column)}")
    )

    if aggregation.function == "COUNT_DISTINCT":
        return f"COUNT(DISTINCT {column})"

    distinct = "DISTINCT " if aggregation.distinct else ""

    return f"{aggregation.function}" f"({distinct}{column})"


def _compile_aggregation(
    aggregation: Aggregation,
    aliases: dict[str, str],
) -> str:
    expression = _aggregation_expression(
        aggregation,
        aliases,
    )

    return f"{expression} AS " f"{_safe_identifier(aggregation.alias)}"


def _compile_aggregate_predicate(
    predicate: AggregatePredicate,
    aggregations_by_alias: dict[str, Aggregation],
    aliases: dict[str, str],
) -> str:
    """
    Compile a logical aggregate predicate as a HAVING expression.

    The underlying aggregate expression is compiled directly rather than
    relying on MySQL alias handling in HAVING.
    """
    aggregation_alias = _clean_text(predicate.aggregation_alias).lower()

    aggregation = aggregations_by_alias.get(aggregation_alias)

    if aggregation is None:
        raise ValueError(
            "Unknown aggregation alias in aggregate predicate: "
            f"{predicate.aggregation_alias!r}"
        )

    expression = _aggregation_expression(
        aggregation,
        aliases,
    )

    return (
        f"{expression} "
        f"{_comparison_operator_sql(predicate.operator)} "
        f"{_sql_literal(predicate.value)}"
    )


def _compile_latest_before(
    operation: TemporalOperation,
    aliases: dict[str, str],
) -> str:
    source_table = _safe_identifier(operation.source_table)

    reference_alias = aliases[operation.reference_table.lower()]

    output_alias = _safe_identifier(operation.output_alias)

    source_timestamp = _safe_identifier(operation.source_timestamp_column)

    reference_timestamp = _safe_identifier(operation.reference_timestamp_column)

    conditions = [
        (
            f"temporal_source.{source_timestamp} "
            f"<= {reference_alias}.{reference_timestamp}"
        )
    ]

    for key in operation.correlation_keys:
        conditions.append(
            (
                "temporal_source."
                f"{_safe_identifier(key.source_column)} "
                f"= {aliases[key.reference_table.lower()]}."
                f"{_safe_identifier(key.reference_column)}"
            )
        )

    ordering = ["temporal_source." f"{source_timestamp} DESC"]

    if operation.tie_breaker_column:
        ordering.append(
            (
                "temporal_source."
                f"{_safe_identifier(operation.tie_breaker_column)} "
                "DESC"
            )
        )

    selected_columns = operation.selected_columns or [operation.source_timestamp_column]

    select_list = ",\n            ".join(
        ("temporal_source." f"{_safe_identifier(column)}")
        for column in selected_columns
    )

    return (
        "LEFT JOIN LATERAL (\n"
        "        SELECT\n"
        f"            {select_list}\n"
        f"        FROM {source_table} AS temporal_source\n"
        f"        WHERE {' AND '.join(conditions)}\n"
        f"        ORDER BY {', '.join(ordering)}\n"
        "        LIMIT 1\n"
        f"    ) AS {output_alias} ON TRUE"
    )


def _compile_derived_predicate(
    predicate: DerivedPredicate,
) -> str:
    alias = _safe_identifier(predicate.temporal_alias)

    column = _safe_identifier(predicate.column)

    reference = f"{alias}.{column}"

    if predicate.operator == FilterOperator.IS_NULL:
        expression = f"{reference} IS NULL"

    elif predicate.operator == FilterOperator.IS_NOT_NULL:
        expression = f"{reference} IS NOT NULL"

    else:
        expression = (
            f"{reference} "
            f"{_operator_sql(predicate.operator)} "
            f"{_sql_literal(predicate.value)}"
        )

    if predicate.allow_missing:
        return f"({reference} IS NULL " f"OR {expression})"

    return expression


# ============================================================================
# Reporting helpers
# ============================================================================


def _relationship_descriptions(
    plan: LogicalQueryPlan,
) -> list[str]:
    descriptions: list[str] = []

    for join in plan.joins:
        for condition in join.all_conditions():
            descriptions.append(
                (
                    f"{join.left_table}."
                    f"{condition.left_column} "
                    f"{condition.operator} "
                    f"{join.right_table}."
                    f"{condition.right_column}"
                )
            )

    for comparison in plan.field_comparisons:
        descriptions.append(
            (
                f"{comparison.left_table}."
                f"{comparison.left_column} "
                f"{comparison.operator.value} "
                f"{comparison.right_table}."
                f"{comparison.right_column}"
            )
        )

    for predicate in plan.existence_predicates:
        existence_label = "must_exist" if predicate.must_exist else "must_not_exist"

        for condition in predicate.all_conditions():
            descriptions.append(
                (
                    f"{existence_label}: "
                    f"{predicate.left_table}."
                    f"{condition.left_column} "
                    f"{condition.operator} "
                    f"{predicate.right_table}."
                    f"{condition.right_column}"
                )
            )

    return _unique_strings(descriptions)


def _filter_descriptions(
    plan: LogicalQueryPlan,
) -> list[str]:
    descriptions = [
        (f"{item.table}.{item.column} " f"{item.operator.value} " f"{item.value!r}")
        for item in plan.filters
    ]

    descriptions.extend(
        [
            (
                f"{item.left_table}.{item.left_column} "
                f"{item.operator.value} "
                f"{item.right_table}.{item.right_column}"
            )
            for item in plan.field_comparisons
        ]
    )

    descriptions.extend(
        [
            (f"{'EXISTS' if item.must_exist else 'NOT EXISTS'} " f"{item.right_table}")
            for item in plan.existence_predicates
        ]
    )

    descriptions.extend(
        [
            (
                f"aggregate:{item.aggregation_alias} "
                f"{item.operator.value} "
                f"{item.value!r}"
            )
            for item in plan.aggregate_predicates
        ]
    )

    return _unique_strings(descriptions)


# ============================================================================
# Main compiler
# ============================================================================


def compile_logical_plan(
    plan: LogicalQueryPlan,
    schema: SchemaCatalog,
) -> SQLPlan:
    """
    Validate and compile one LogicalQueryPlan into a read-only SQLPlan.
    """
    if not plan.is_supported:
        return SQLPlan(
            purpose=plan.purpose,
            database_type="mysql",
            tables=_unique_strings(plan.required_tables),
            columns=_unique_strings(plan.selected_fields),
            validation_errors=(
                _unique_strings(plan.missing_information)
                or ["Logical plan is unsupported."]
            ),
            is_valid=False,
        )

    _complete_required_tables(plan)

    normalization_errors = _normalize_plan_fields(
        plan=plan,
        schema=schema,
    )

    validation_errors = _validate_plan(
        plan=plan,
        schema=schema,
    )

    errors = _unique_strings(normalization_errors + validation_errors)

    if errors:
        return SQLPlan(
            purpose=plan.purpose,
            database_type="mysql",
            tables=_unique_strings(plan.required_tables),
            columns=_unique_strings(plan.selected_fields),
            relationships=_relationship_descriptions(plan),
            filters=_filter_descriptions(plan),
            validation_errors=errors,
            is_valid=False,
        )

    tables = _unique_strings(
        [
            plan.base_table,
            *plan.required_tables,
        ]
    )

    aliases = {
        table: f"t{index}"
        for (
            index,
            table,
        ) in enumerate(
            tables,
            start=1,
        )
    }

    select_items: list[str] = []

    for field in plan.selected_fields:
        (
            table,
            column,
        ) = _field_parts(field)

        select_items.append(f"{aliases[table]}.{column}")

    for aggregation in plan.aggregations:
        select_items.append(
            _compile_aggregation(
                aggregation,
                aliases,
            )
        )

    if not select_items:
        select_items.append(f"{aliases[plan.base_table]}.*")

    join_statements: list[str] = []

    for join in plan.joins:
        join_statements.append(
            _compile_logical_join(
                join,
                aliases,
            )
        )

    for operation in plan.temporal_operations:
        if operation.operation != "latest_before":
            return SQLPlan(
                purpose=plan.purpose,
                database_type="mysql",
                tables=tables,
                columns=plan.selected_fields,
                relationships=_relationship_descriptions(plan),
                filters=_filter_descriptions(plan),
                validation_errors=[
                    "Temporal operation not yet supported: " f"{operation.operation}."
                ],
                is_valid=False,
            )

        join_statements.append(
            _compile_latest_before(
                operation,
                aliases,
            )
        )

    conditions = [
        _compile_filter(
            item,
            aliases,
        )
        for item in plan.filters
    ]

    conditions.extend(
        [
            _compile_field_comparison(
                item,
                aliases,
            )
            for item in plan.field_comparisons
        ]
    )

    conditions.extend(
        [
            _compile_existence_predicate(
                item,
                aliases,
            )
            for item in plan.existence_predicates
        ]
    )

    conditions.extend(
        [_compile_derived_predicate(item) for item in plan.derived_predicates]
    )

    sql_lines = [
        "SELECT",
        ("    " + ",\n    ".join(select_items)),
        (
            f"FROM "
            f"{_safe_identifier(plan.base_table)} "
            f"AS {aliases[plan.base_table]}"
        ),
    ]

    if join_statements:
        sql_lines.extend(join_statements)

    if conditions:
        sql_lines.extend(
            [
                "WHERE",
                ("    " + "\n    AND ".join(conditions)),
            ]
        )

    if plan.group_by:
        group_items: list[str] = []

        for field in plan.group_by:
            (
                table,
                column,
            ) = _field_parts(field)

            group_items.append(f"{aliases[table]}.{column}")

        sql_lines.extend(
            [
                "GROUP BY",
                ("    " + ",\n    ".join(group_items)),
            ]
        )

    if plan.aggregate_predicates:
        aggregations_by_alias = {
            _clean_text(aggregation.alias).lower(): aggregation
            for aggregation in plan.aggregations
            if _clean_text(aggregation.alias)
        }

        having_items = [
            _compile_aggregate_predicate(
                predicate,
                aggregations_by_alias,
                aliases,
            )
            for predicate in plan.aggregate_predicates
        ]

        sql_lines.extend(
            [
                "HAVING",
                ("    " + "\n    AND ".join(having_items)),
            ]
        )

    if plan.order_by:
        order_items = [
            (
                f"{aliases[item.table.lower()]}."
                f"{_safe_identifier(item.column)} "
                f"{item.direction}"
            )
            for item in plan.order_by
        ]

        sql_lines.extend(
            [
                "ORDER BY",
                ("    " + ",\n    ".join(order_items)),
            ]
        )

    if plan.limit is not None:
        sql_lines.append(f"LIMIT {plan.limit}")

    sql = "\n".join(sql_lines).rstrip(";") + ";"

    return SQLPlan(
        purpose=plan.purpose,
        database_type="mysql",
        tables=tables,
        columns=plan.selected_fields,
        relationships=_relationship_descriptions(plan),
        filters=_filter_descriptions(plan),
        sql=sql,
        is_valid=True,
        validation_errors=[],
    )


# ============================================================================
# Hard validation for LLM-generated raw SQL
# ============================================================================


def _parse_allowed_path_edges(
    allowed_relationship_paths: list[str] | None,
) -> set[tuple[str, str, str, str]]:
    edges: set[tuple[str, str, str, str]] = set()

    pattern = re.compile(
        r"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)"
        r"\s*(?:->|→|=|joins\s+to)\s*"
        r"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)",
        flags=re.IGNORECASE,
    )

    for item in allowed_relationship_paths or []:
        for left_table, left_col, right_table, right_col in pattern.findall(
            _clean_text(item)
        ):
            edge = (
                left_table.lower(),
                left_col.lower(),
                right_table.lower(),
                right_col.lower(),
            )
            edges.add(edge)
            edges.add((edge[2], edge[3], edge[0], edge[1]))

    return edges


def _nearest_select(
    node: exp.Expression | None,
) -> exp.Select | None:
    """Return the SELECT scope that directly owns an AST node."""
    current = node

    while current is not None:
        if isinstance(current, exp.Select):
            return current

        current = getattr(current, "parent", None)

    return None


def _parent_select(
    select: exp.Select | None,
) -> exp.Select | None:
    """Return the nearest enclosing SELECT scope, if any."""
    if select is None:
        return None

    current = getattr(select, "parent", None)

    while current is not None:
        if isinstance(current, exp.Select):
            return current

        current = getattr(current, "parent", None)

    return None


def _scope_tables(
    select: exp.Select,
) -> list[exp.Table]:
    """
    Return only table-like sources directly visible in this SELECT scope.

    Nested subquery sources are excluded because their nearest SELECT differs.
    A returned exp.Table may represent either a physical table or a CTE reference;
    callers decide which namespace it belongs to.
    """
    return [
        table
        for table in select.find_all(exp.Table)
        if _nearest_select(table) is select
    ]


def _projection_name(expression: exp.Expression) -> str:
    """Return the externally visible name of one SELECT projection."""
    alias = _clean_text(getattr(expression, "alias", ""))
    if alias:
        return alias.lower()

    if isinstance(expression, exp.Column):
        return _clean_text(expression.name).lower()

    alias_or_name = _clean_text(getattr(expression, "alias_or_name", ""))
    return alias_or_name.lower()


def _cte_output_columns(cte: exp.CTE) -> set[str] | None:
    """
    Return columns projected by a CTE.

    None means the projection contains a wildcard and therefore cannot be safely
    enumerated from syntax alone. In that case the CTE is still a valid SQL-local
    relation and MySQL remains the final authority on its wildcard-expanded shape.
    """
    alias_expression = cte.args.get("alias")
    explicit_columns = []

    if alias_expression is not None:
        explicit_columns = [
            _clean_text(getattr(column, "name", column)).lower()
            for column in (alias_expression.args.get("columns") or [])
            if _clean_text(getattr(column, "name", column))
        ]

    if explicit_columns:
        return set(explicit_columns)

    query = cte.this
    select = query if isinstance(query, exp.Select) else query.find(exp.Select)

    if select is None:
        return set()

    projected: set[str] = set()

    for item in select.expressions:
        if isinstance(item, exp.Star):
            return None

        # Handles forms such as table.*.
        if isinstance(item, exp.Column) and _clean_text(item.name) == "*":
            return None

        name = _projection_name(item)
        if name:
            projected.add(name)

    return projected


def _collect_cte_registry(
    expression: exp.Expression,
) -> dict[str, set[str] | None]:
    """
    Collect query-local CTE relation names and their projected columns.

    CTEs are SQL-scope objects, not physical schema tables. They must never be
    validated with schema.has_table().
    """
    registry: dict[str, set[str] | None] = {}

    for cte in expression.find_all(exp.CTE):
        name = _clean_text(cte.alias_or_name).lower()
        if name:
            registry[name] = _cte_output_columns(cte)

    return registry


def _scope_source_map(
    select: exp.Select,
    cte_registry: dict[str, set[str] | None],
) -> dict[str, tuple[str, str]]:
    """
    Map source names/aliases visible in one SELECT to:

        ("physical", physical_table_name)
        ("cte", cte_name)
    """
    mapping: dict[str, tuple[str, str]] = {}

    for table in _scope_tables(select):
        source_name = _clean_text(table.name).lower()
        alias = _clean_text(table.alias_or_name).lower()

        if not source_name:
            continue

        if source_name in cte_registry:
            source = ("cte", source_name)
        else:
            source = ("physical", source_name)

        mapping[source_name] = source

        if alias:
            mapping[alias] = source

    return mapping


def _source_has_column(
    source: tuple[str, str],
    column_name: str,
    *,
    schema: SchemaCatalog,
    cte_registry: dict[str, set[str] | None],
) -> bool:
    kind, name = source
    column = _clean_text(column_name).lower()

    if kind == "physical":
        return schema.has_column(name, column)

    projected = cte_registry.get(name)

    # Wildcard projection: shape cannot be fully enumerated deterministically.
    if projected is None:
        return True

    return column in projected


def _scope_alias_map(
    select: exp.Select,
) -> dict[str, str]:
    """
    Map local physical table names and aliases to physical table names.

    This compatibility helper intentionally excludes CTE awareness because callers
    use it only when they need a physical table identity (for example relationship
    metadata). CTE-backed aliases therefore do not masquerade as schema tables.
    """
    mapping: dict[str, str] = {}

    for table in _scope_tables(select):
        table_name = _clean_text(table.name).lower()
        alias = _clean_text(table.alias_or_name).lower()

        if table_name:
            mapping[table_name] = table_name

        if alias:
            mapping[alias] = table_name

    return mapping


def _resolve_qualified_table(
    qualifier: str,
    select: exp.Select | None,
) -> str | None:
    """
    Resolve a qualified physical table/alias from current SQL scope outward.

    This compatibility helper is used for physical relationship metadata. CTE
    validation is handled separately by _validate_raw_sql_schema_references().
    """
    cleaned = _clean_text(qualifier).lower()
    current = select

    while current is not None:
        resolved = _scope_alias_map(current).get(cleaned)

        if resolved:
            return resolved

        current = _parent_select(current)

    return None


def _resolve_unqualified_column(
    column_name: str,
    select: exp.Select | None,
    schema: SchemaCatalog,
) -> tuple[str | None, list[str]]:
    """
    Compatibility resolver for physical-column metadata.

    CTE-aware correctness validation is performed separately.
    """
    cleaned = _clean_text(column_name).lower()
    current = select

    while current is not None:
        local_tables = _unique_strings(
            [
                _clean_text(table.name).lower()
                for table in _scope_tables(current)
                if _clean_text(table.name)
                and schema.has_table(_clean_text(table.name).lower())
            ]
        )

        owners = [table for table in local_tables if schema.has_column(table, cleaned)]

        if len(owners) == 1:
            return owners[0], owners

        if len(owners) > 1:
            return None, owners

        current = _parent_select(current)

    return None, []


def _nearest_select_scope(node: exp.Expression) -> exp.Select | None:
    """Return the nearest SELECT ancestor for scope-aware raw-SQL validation."""
    current = node.parent

    while current is not None:
        if isinstance(current, exp.Select):
            return current

        current = current.parent

    return None


def _validate_raw_sql_schema_references(
    expression: exp.Expression,
    schema: SchemaCatalog,
) -> list[str]:
    """
    Validate raw SQL using SQL lexical scope rather than treating every exp.Table
    as a physical database table.

    Physical sources are checked against SchemaCatalog.
    CTE names and table aliases are query-local relations.
    CTE column references are checked against the CTE projection when determinable.
    Correlated outer aliases remain visible to nested SELECT scopes.

    This remains structural validation only; it does not judge business meaning.
    """
    errors: list[str] = []
    cte_registry = _collect_cte_registry(expression)

    def child_selects(select: exp.Select) -> list[exp.Select]:
        result: list[exp.Select] = []

        for child in select.find_all(exp.Select):
            if child is select:
                continue

            if _nearest_select_scope(child) is select:
                result.append(child)

        return result

    def validate_select(
        select: exp.Select,
        inherited_sources: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        inherited_sources = dict(inherited_sources or {})
        local_sources = _scope_source_map(select, cte_registry)

        # Validate only physical sources against the authoritative schema.
        for source_kind, source_name in set(local_sources.values()):
            if source_kind == "physical" and not schema.has_table(source_name):
                errors.append(f"Undocumented table: {source_name}.")

        visible_sources = {**inherited_sources, **local_sources}

        for column in select.find_all(exp.Column):
            if _nearest_select_scope(column) is not select:
                continue

            column_name = _clean_text(column.name).lower()
            qualifier = _clean_text(column.table).lower()

            if not column_name or column_name == "*":
                continue

            if qualifier:
                source = visible_sources.get(qualifier)

                if source is None:
                    errors.append(
                        "Column qualifier is not visible in this SQL scope: "
                        f"{qualifier}.{column_name}."
                    )
                    continue

                if not _source_has_column(
                    source,
                    column_name,
                    schema=schema,
                    cte_registry=cte_registry,
                ):
                    source_kind, source_name = source

                    if source_kind == "physical":
                        errors.append(
                            f"Undocumented column: {source_name}.{column_name}."
                        )
                    else:
                        errors.append(
                            f"CTE `{source_name}` does not project column: "
                            f"{column_name}."
                        )

                continue

            # Unqualified columns resolve against local sources only. Do not silently
            # bind an inner unqualified reference to an outer query.
            owners = sorted(
                source_name
                for source_name, source in local_sources.items()
                if source_name == source[1]  # de-duplicate aliases
                and _source_has_column(
                    source,
                    column_name,
                    schema=schema,
                    cte_registry=cte_registry,
                )
            )

            if len(owners) == 1:
                continue

            if not owners:
                errors.append(
                    "Column is not documented/projected by any source visible "
                    f"to this SQL scope: {column_name}."
                )
            else:
                errors.append(
                    "Ambiguous unqualified column in its SQL scope: "
                    f"{column_name}. Possible sources: {owners}."
                )

        for child in child_selects(select):
            validate_select(child, visible_sources)

    roots = [
        select
        for select in expression.find_all(exp.Select)
        if _nearest_select_scope(select) is None
    ]

    if isinstance(expression, exp.Select) and expression not in roots:
        roots.insert(0, expression)

    for root in roots:
        validate_select(root)

    return _unique_strings(errors)


def validate_read_only_sql(
    sql: str,
    schema: SchemaCatalog,
    *,
    purpose: str = "",
    allowed_tables: list[str] | None = None,
    allowed_relationship_paths: list[str] | None = None,
) -> SQLPlan:
    """
    Hard-validate raw LLM SQL without deciding how the query should be written.

    Enforced:
    - MySQL parse validity;
    - exactly one statement;
    - read-only query;
    - documented tables;
    - allowed table scope when supplied;
    - documented qualified columns;
    - SQL-scope-aware resolution of unqualified columns;
    - valid aliases / resolvable physical column references.

    Advisory only for raw investigation SQL:
    - whether a cross-table JOIN/comparison is a documented SchemaGraph
      relationship;
    - whether a supplied SchemaGraph path was used.

    A raw LLM query may intentionally correlate two documented columns for
    analytical purposes even when that comparison is not a declared relational
    edge. Whether that correlation is semantically useful is evaluated
    downstream after execution.

    Not enforced:
    - whether the LLM chose a particular aggregation technique;
    - whether recency uses a particular SQL idiom;
    - whether the evidence "proves" a security conclusion.
    """
    cleaned_sql = _clean_text(sql)
    errors: list[str] = []

    if not cleaned_sql:
        return SQLPlan(
            purpose=purpose,
            database_type="mysql",
            validation_errors=["SQL is empty."],
            is_valid=False,
        )

    try:
        statements = [
            statement
            for statement in sqlglot.parse(cleaned_sql, read="mysql")
            if statement is not None
        ]
    except ParseError as exc:
        return SQLPlan(
            purpose=purpose,
            database_type="mysql",
            sql=cleaned_sql,
            validation_errors=[f"MySQL parse error: {exc}"],
            is_valid=False,
        )

    if len(statements) != 1:
        return SQLPlan(
            purpose=purpose,
            database_type="mysql",
            sql=cleaned_sql,
            validation_errors=["Exactly one SQL statement is allowed."],
            is_valid=False,
        )

    expression = statements[0]

    # Recursively hard-validate every SELECT/subquery schema reference before
    # continuing with the existing raw-SQL safety and structural checks.
    errors.extend(
        _validate_raw_sql_schema_references(
            expression=expression,
            schema=schema,
        )
    )

    forbidden = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Create,
        exp.Drop,
        exp.Alter,
        exp.Command,
    )

    if not isinstance(expression, (exp.Select, exp.Union, exp.Subquery)):
        errors.append("Only read-only SELECT/query SQL is permitted.")

    for forbidden_type in forbidden:
        if expression.find(forbidden_type) is not None:
            errors.append(
                f"Modifying SQL operation is not permitted: "
                f"{forbidden_type.__name__}."
            )

    cte_registry = _collect_cte_registry(expression)

    # Report only physical database tables. Query-local CTE names are not schema
    # tables and must not appear as undocumented-table failures.
    table_nodes = list(expression.find_all(exp.Table))
    tables = _unique_strings(
        [
            _clean_text(table.name).lower()
            for table in table_nodes
            if _clean_text(table.name)
            and _clean_text(table.name).lower() not in cte_registry
        ]
    )

    allowed = {
        _clean_text(table).lower()
        for table in (allowed_tables or [])
        if _clean_text(table)
    }

    for table in tables:
        if not schema.has_table(table):
            errors.append(f"Undocumented table: {table}.")

        if allowed and table not in allowed:
            errors.append(f"Table is outside the SchemaGraph-approved scope: {table}.")

    # Column metadata contains physical columns only. CTE-projected columns are
    # query-local and were already structurally validated above.
    columns: list[str] = []

    for column in expression.find_all(exp.Column):
        column_name = _clean_text(column.name).lower()

        if not column_name or column_name == "*":
            continue

        select_scope = _nearest_select(column)
        qualifier = _clean_text(column.table).lower()

        if qualifier:
            table_name = _resolve_qualified_table(
                qualifier,
                select_scope,
            )

            if not table_name or not schema.has_table(table_name):
                continue

            if not schema.has_column(table_name, column_name):
                continue

            qualified = f"{table_name}.{column_name}"

            if qualified not in columns:
                columns.append(qualified)

        else:
            resolved_table, _owners = _resolve_unqualified_column(
                column_name,
                select_scope,
                schema,
            )

            if resolved_table:
                qualified = f"{resolved_table}.{column_name}"

                if qualified not in columns:
                    columns.append(qualified)

    approved_edges = _parse_allowed_path_edges(allowed_relationship_paths)

    schema_edges: set[tuple[str, str, str, str]] = set()

    for left_table, left_col, right_table, right_col in schema.relationships:
        edge = (
            left_table.lower(),
            left_col.lower(),
            right_table.lower(),
            right_col.lower(),
        )
        schema_edges.add(edge)
        schema_edges.add((edge[2], edge[3], edge[0], edge[1]))

    relationship_descriptions: list[str] = []

    for join in expression.find_all(exp.Join):
        on_expression = join.args.get("on")

        if on_expression is None:
            errors.append("Every explicit JOIN must contain an ON condition.")
            continue

        # Raw investigation SQL may correlate documented columns even when the
        # comparison is not a declared SchemaGraph relationship. Record
        # documented edges when present, but do not reject undocumented
        # analytical correlations solely for lacking a declared relationship.
        for equality in on_expression.find_all(exp.EQ):
            left = equality.left
            right = equality.right

            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                continue

            left_alias = str(left.table or "").strip().lower()
            right_alias = str(right.table or "").strip().lower()

            join_scope = _nearest_select(join)

            left_table = _resolve_qualified_table(
                left_alias,
                join_scope,
            )
            right_table = _resolve_qualified_table(
                right_alias,
                join_scope,
            )

            if (
                not left_table
                or not right_table
                or left_table == right_table
                or not schema.has_table(left_table)
                or not schema.has_table(right_table)
            ):
                continue

            left_column = str(left.name or "").strip().lower()
            right_column = str(right.name or "").strip().lower()

            # Column existence is already hard-validated above. This block only
            # records relationship metadata when the comparison corresponds to
            # a documented edge.
            edge = (
                left_table,
                left_column,
                right_table,
                right_column,
            )

            if edge in schema_edges:
                description = f"{edge[0]}.{edge[1]} -> " f"{edge[2]}.{edge[3]}"

                if description not in relationship_descriptions:
                    relationship_descriptions.append(description)

                # If a caller supplied approved paths, keep that information as
                # metadata guidance only for raw investigation SQL. Do not turn
                # it into a hard execution blocker.
                if approved_edges and edge not in approved_edges:
                    advisory = (
                        "advisory: documented relationship is outside supplied "
                        "SchemaGraph path hints: "
                        f"{edge[0]}.{edge[1]} -> "
                        f"{edge[2]}.{edge[3]}"
                    )
                    if advisory not in relationship_descriptions:
                        relationship_descriptions.append(advisory)

    unique_errors = _unique_strings(errors)

    return SQLPlan(
        purpose=purpose,
        database_type="mysql",
        tables=tables,
        columns=columns,
        relationships=relationship_descriptions,
        sql=cleaned_sql,
        validation_errors=unique_errors,
        is_valid=not unique_errors,
    )
