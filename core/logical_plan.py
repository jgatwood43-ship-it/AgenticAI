"""
core/logical_plan.py
────────────────────
Reusable logical-query models for schema-grounded database analysis.

The LLM produces a LogicalQueryPlan. Python validates and compiles it into SQL.

## Completeness contract

A plan must explicitly state:

* required_tables:
  every table needed to answer the question;

* required_output_fields:
  every database field that must appear in the final evidence;

* selected_fields:
  fields actually projected by the query;

* joins, comparisons, existence predicates, temporal operations, and
  aggregate predicates:
  how required tables participate and how the requested evidence condition is
  tested.

The compiler rejects plans when required tables or required outputs disappear.

This module describes logical intent only. It does not contain SQL.

Compatibility note
------------------
LogicalJoin retains required left_column/right_column fields because the current
sql_compiler.py still consumes those fields directly. Multi-column join support
is additive through additional_conditions so existing compiler behavior remains
valid while the compiler is extended.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# ============================================================================
# Core enums
# ============================================================================


class ResultShape(str, Enum):
    ROWS = "rows"
    COUNT = "count"
    BOOLEAN = "boolean"
    GROUPED_ROWS = "grouped_rows"
    SUMMARY = "summary"


class FilterOperator(str, Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    LIKE = "like"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class ComparisonOperator(str, Enum):
    """
    Operators for field-to-field comparisons.

    LogicalFilter compares a field to a literal value.
    FieldComparison compares one documented database field to another.
    """

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"


# ============================================================================
# Join models
# ============================================================================


class JoinCondition(BaseModel):
    """
    One additional column-to-column condition in a logical relationship.

    Table names are supplied by the containing LogicalJoin or
    ExistencePredicate.
    """

    left_column: str
    right_column: str
    operator: Literal["eq", "ne"] = "eq"
    purpose: str = ""


class LogicalJoin(BaseModel):
    """
    Join two documented tables.

    left_column/right_column remain the primary join condition and are required
    for compatibility with the current sql_compiler.py.

    additional_conditions allow a documented relationship to require more than
    one pair of matching columns.
    """

    left_table: str
    left_column: str

    right_table: str
    right_column: str

    join_type: Literal[
        "INNER",
        "LEFT",
    ] = "INNER"

    additional_conditions: list[JoinCondition] = Field(default_factory=list)

    purpose: str = ""

    def all_conditions(
        self,
    ) -> list[JoinCondition]:
        """
        Return the primary join plus any additional conditions.
        """
        return [
            JoinCondition(
                left_column=self.left_column,
                right_column=self.right_column,
                operator="eq",
                purpose=self.purpose,
            ),
            *self.additional_conditions,
        ]


# ============================================================================
# Literal filters
# ============================================================================


class LogicalFilter(BaseModel):
    """
    Compare one documented field to a literal value.
    """

    table: str
    column: str
    operator: FilterOperator

    value: str | int | float | bool | list[str | int | float] | None = None

    source: Literal[
        "user",
        "schema",
        "business_rule",
    ] = "user"

    purpose: str = ""


# ============================================================================
# Field-to-field comparisons
# ============================================================================


class FieldComparison(BaseModel):
    """
    Compare two documented database fields directly.

    This allows logical tests such as:
        table_a.status != table_b.status

    It deliberately contains no raw SQL.
    """

    left_table: str
    left_column: str

    operator: ComparisonOperator

    right_table: str
    right_column: str

    purpose: str = ""


# ============================================================================
# Existence / non-existence checks
# ============================================================================


class ExistencePredicate(BaseModel):
    """
    Require that a related record either exists or does not exist.

    This is the logical representation for generalized checks such as:
        - a child/reference record has a corresponding parent;
        - an event has a matching authorization rule;
        - an event has no matching authorization rule.

    SQLCompiler decides whether this becomes EXISTS, NOT EXISTS, or an
    equivalent deterministic anti-join.
    """

    left_table: str
    right_table: str

    # Primary relationship condition.
    left_column: str
    right_column: str

    additional_conditions: list[JoinCondition] = Field(default_factory=list)

    must_exist: bool = True

    purpose: str = ""

    def all_conditions(
        self,
    ) -> list[JoinCondition]:
        return [
            JoinCondition(
                left_column=self.left_column,
                right_column=self.right_column,
                operator="eq",
                purpose=self.purpose,
            ),
            *self.additional_conditions,
        ]


# ============================================================================
# Aggregation / ordering
# ============================================================================


class Aggregation(BaseModel):
    function: Literal[
        "COUNT",
        "COUNT_DISTINCT",
        "SUM",
        "AVG",
        "MIN",
        "MAX",
    ]

    table: str
    column: str
    alias: str

    distinct: bool = False


class AggregatePredicate(BaseModel):
    """
    Predicate applied to a declared aggregation.

    This models SQL HAVING semantics without storing raw SQL. The predicate
    references an Aggregation by alias.

    Example:
        Aggregation(
            function="COUNT",
            table="video_footage",
            column="*",
            alias="occurrence_count",
        )

        AggregatePredicate(
            aggregation_alias="occurrence_count",
            operator=ComparisonOperator.GT,
            value=1,
            purpose="Return only duplicate groups.",
        )
    """

    aggregation_alias: str
    operator: ComparisonOperator
    value: int | float
    purpose: str = ""


class OrderingRule(BaseModel):
    table: str
    column: str

    direction: Literal[
        "ASC",
        "DESC",
    ] = "ASC"


# ============================================================================
# Temporal operations
# ============================================================================


class CorrelationKey(BaseModel):
    source_table: str
    source_column: str

    reference_table: str
    reference_column: str


class TemporalOperation(BaseModel):
    """
    Reusable temporal operator.

    latest_before:
        Select the most recent source row whose source timestamp is less than
        or equal to the reference timestamp and whose correlation keys match.

    earliest_after:
        Select the earliest source row whose source timestamp is greater than
        or equal to the reference timestamp and whose correlation keys match.

    within_interval:
        Associate records where a documented timestamp falls within a
        documented interval. Compiler support is added independently.
    """

    operation: Literal[
        "latest_before",
        "earliest_after",
        "within_interval",
    ]

    source_table: str
    source_timestamp_column: str

    reference_table: str
    reference_timestamp_column: str

    correlation_keys: list[CorrelationKey] = Field(default_factory=list)

    output_alias: str

    tie_breaker_column: str | None = None

    selected_columns: list[str] = Field(default_factory=list)


class DerivedPredicate(BaseModel):
    """
    Predicate applied to the row selected by a temporal operation.
    """

    temporal_alias: str
    column: str
    operator: FilterOperator

    value: str | int | float | bool | None = None

    allow_missing: bool = False

    purpose: str = ""


# ============================================================================
# Logical query plan
# ============================================================================


class LogicalQueryPlan(BaseModel):
    """
    Structured, schema-grounded plan compiled deterministically by Python.

    The plan describes WHAT database evidence must be tested, not raw SQL.
    """

    purpose: str

    database_type: Literal["mysql",] = "mysql"

    result_shape: ResultShape

    base_table: str

    # Every table needed to answer the question.
    required_tables: list[str]

    # Every table.column field the final evidence must return.
    required_output_fields: list[str] = Field(default_factory=list)

    # Fields actually projected by the logical query.
    selected_fields: list[str]

    # Direct documented table relationships.
    joins: list[LogicalJoin] = Field(default_factory=list)

    # Field-to-literal predicates.
    filters: list[LogicalFilter] = Field(default_factory=list)

    # Field-to-field predicates.
    field_comparisons: list[FieldComparison] = Field(default_factory=list)

    # Related-record existence/non-existence predicates.
    existence_predicates: list[ExistencePredicate] = Field(default_factory=list)

    # Temporal/correlation operations.
    temporal_operations: list[TemporalOperation] = Field(default_factory=list)

    derived_predicates: list[DerivedPredicate] = Field(default_factory=list)

    # Aggregation/grouping.
    aggregations: list[Aggregation] = Field(default_factory=list)

    # Predicates over declared aggregations. Compiled deterministically as
    # HAVING expressions.
    aggregate_predicates: list[AggregatePredicate] = Field(default_factory=list)

    group_by: list[str] = Field(default_factory=list)

    order_by: list[OrderingRule] = Field(default_factory=list)

    limit: int | None = Field(
        default=None,
        ge=1,
        le=10000,
    )

    assumptions: list[str] = Field(default_factory=list)

    missing_information: list[str] = Field(default_factory=list)

    is_supported: bool
