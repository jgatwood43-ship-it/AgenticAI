"""
core/direct_sql_validator.py
────────────────────────────
Permissive hard guard for LLM-generated DirectQuery SQL.

Design principle
----------------
The LLM owns SQL reasoning and self-review.
MySQL is the final authority on SQL execution and scope resolution.
Python blocks only conditions it can prove are unsafe or schema-invalid.

BLOCKING checks:
1. SQL is empty.
2. SQL cannot be parsed as MySQL at all.
3. More than one statement is present.
4. SQL is not read-only SELECT/query SQL.
5. A modifying operation appears anywhere in the AST.
6. A clearly referenced table is not documented.
7. A clearly qualified table.column reference is provably undocumented.
8. A clearly resolved explicit JOIN edge is provably undocumented.

NON-BLOCKING warnings:
- ambiguous unqualified columns;
- unresolved local/outer aliases;
- correlated-subquery scope uncertainty;
- complex or unsupported JOIN syntax;
- cases Python cannot prove valid or invalid.

If Python is uncertain, it warns and allows the read-only query to reach MySQL.
MySQL then determines whether the SQL is actually executable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.optimizer.scope import traverse_scope

from core.schema_graph import SchemaGraph


class DirectSQLValidationCategory(str, Enum):
    """Stable machine-readable validation categories."""

    EMPTY_SQL = "empty_sql"
    PARSE_ERROR = "parse_error"
    MULTIPLE_STATEMENTS = "multiple_statements"
    NON_READ_ONLY = "non_read_only"
    MODIFYING_SQL = "modifying_sql"
    UNDOCUMENTED_TABLE = "undocumented_table"
    UNDOCUMENTED_COLUMN = "undocumented_column"
    AMBIGUOUS_COLUMN = "ambiguous_column"
    UNRESOLVED_COLUMN_SCOPE = "unresolved_column_scope"
    JOIN_WITHOUT_ON = "join_without_on"
    UNSUPPORTED_JOIN = "unsupported_join"
    UNRESOLVED_JOIN_ALIAS = "unresolved_join_alias"
    UNDOCUMENTED_RELATIONSHIP = "undocumented_relationship"
    QUERY_SHAPE = "query_shape"


@dataclass(frozen=True)
class DirectSQLValidationIssue:
    """One deterministic validation finding."""

    category: DirectSQLValidationCategory
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class DirectSQLValidationResult:
    """
    Result returned by DirectSQLValidator.

    `errors` and `warnings` remain string lists for compatibility with the
    current test harness and future RetrieverAgent integration.

    `issues` provides stable machine-readable categories for repair logic.
    """

    is_valid: bool = False
    sql: str = ""

    tables: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    issues: list[DirectSQLValidationIssue] = field(default_factory=list)

    @property
    def error_categories(self) -> list[str]:
        return list(dict.fromkeys(issue.category.value for issue in self.issues))

    @property
    def is_repairable(self) -> bool:
        """
        Return True when an LLM reconsideration is reasonable.

        Unsafe modifying SQL and multi-statement SQL are deliberately not
        repairable through the normal DirectQuery repair loop.
        """
        if self.is_valid:
            return False

        non_repairable = {
            DirectSQLValidationCategory.MULTIPLE_STATEMENTS,
            DirectSQLValidationCategory.NON_READ_ONLY,
            DirectSQLValidationCategory.MODIFYING_SQL,
        }

        return bool(self.issues) and not any(
            issue.category in non_repairable for issue in self.issues
        )

    def repair_feedback(self) -> str:
        """
        Produce concise feedback for an LLM reconsideration.

        The validator itself never invokes the LLM.
        """
        if self.is_valid:
            return ""

        if not self.errors:
            return "The SQL failed validation. Generate a corrected SELECT query."

        lines = [
            "The candidate SQL was rejected by deterministic validation.",
            "",
            "VALIDATION ERRORS:",
        ]

        for error in self.errors:
            lines.append(f"- {error}")

        lines.extend(
            [
                "",
                "Repair the SQL using only the authoritative schema.",
                "Do not invent tables, columns, relationships, or business values.",
                "Return exactly one read-only MySQL SELECT statement and nothing else.",
            ]
        )

        return "\n".join(lines)


_FORBIDDEN_EXPRESSION_TYPES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
)


def _clean_identifier(
    value: str,
) -> str:
    return str(value or "").strip().strip("`").lower()


def _add_unique(
    values: list[str],
    value: str,
) -> None:
    cleaned = str(value or "").strip()

    if cleaned and cleaned not in values:
        values.append(cleaned)


def _add_issue(
    result: DirectSQLValidationResult,
    category: DirectSQLValidationCategory,
    message: str,
    *,
    warning: bool = False,
) -> None:
    """
    Add one issue once.

    This prevents duplicate error messages when the same invalid column appears
    in SELECT and ORDER BY, which occurred in the Phase-2 test.
    """
    cleaned_message = str(message or "").strip()

    if not cleaned_message:
        return

    issue = DirectSQLValidationIssue(
        category=category,
        message=cleaned_message,
    )

    if issue not in result.issues:
        result.issues.append(issue)

    target = result.warnings if warning else result.errors

    if cleaned_message not in target:
        target.append(cleaned_message)


class DirectSQLValidator:
    """
    Validate candidate direct SQL against SchemaGraph.

    Important:
        * This validator never generates SQL.
        * This validator never repairs SQL.
        * This validator never invents relationships.
        * A caller may use result.repair_feedback() to ask the LLM to reconsider
          a hard schema/safety failure.
        * Semantic result-shape and SQL-construction reasoning are intentionally
          left to the LLM self-review stage.
        * Valid correlated subqueries must not be rejected merely because they
          reference an outer query alias.
    """

    def __init__(
        self,
        schema_graph: SchemaGraph,
    ) -> None:
        self._schema_graph = schema_graph

    def validate(
        self,
        sql: str,
        *,
        question: str = "",
    ) -> DirectSQLValidationResult:
        result = DirectSQLValidationResult(sql=str(sql or "").strip())

        if not result.sql:
            _add_issue(
                result,
                DirectSQLValidationCategory.EMPTY_SQL,
                "SQL is empty.",
            )
            return result

        statements = self._parse_statements(
            result.sql,
            result,
        )

        if not statements:
            return result

        if len(statements) != 1:
            _add_issue(
                result,
                DirectSQLValidationCategory.MULTIPLE_STATEMENTS,
                "Exactly one SQL statement is allowed.",
            )
            return result

        expression = statements[0]

        self._validate_read_only(
            expression,
            result,
        )

        self._validate_tables(
            expression,
            result,
        )

        self._validate_columns(
            expression,
            result,
        )

        self._validate_joins(
            expression,
            result,
        )

        result.is_valid = len(result.errors) == 0

        return result

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_statements(
        sql: str,
        result: DirectSQLValidationResult,
    ) -> list[exp.Expression]:
        try:
            statements = sqlglot.parse(
                sql,
                read="mysql",
            )

            return [statement for statement in statements if statement is not None]

        except ParseError as exc:
            _add_issue(
                result,
                DirectSQLValidationCategory.PARSE_ERROR,
                f"MySQL parse error: {exc}",
            )
            return []

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_read_only(
        expression: exp.Expression,
        result: DirectSQLValidationResult,
    ) -> None:
        """
        Reject non-query statements and modifying operations anywhere in the AST.
        """
        if not isinstance(
            expression,
            (
                exp.Select,
                exp.Union,
                exp.Subquery,
            ),
        ):
            _add_issue(
                result,
                DirectSQLValidationCategory.NON_READ_ONLY,
                "Only read-only SELECT queries are permitted.",
            )

        for forbidden_type in _FORBIDDEN_EXPRESSION_TYPES:
            if expression.find(forbidden_type) is not None:
                _add_issue(
                    result,
                    DirectSQLValidationCategory.MODIFYING_SQL,
                    (
                        "Modifying SQL operation is not permitted: "
                        f"{forbidden_type.__name__}."
                    ),
                )

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def _validate_tables(
        self,
        expression: exp.Expression,
        result: DirectSQLValidationResult,
    ) -> None:
        for table in expression.find_all(exp.Table):
            table_name = _clean_identifier(table.name)

            if not table_name:
                continue

            _add_unique(
                result.tables,
                table_name,
            )

            if not self._schema_graph.has_table(table_name):
                _add_issue(
                    result,
                    DirectSQLValidationCategory.UNDOCUMENTED_TABLE,
                    f"Undocumented table: {table_name}.",
                )

    # ------------------------------------------------------------------
    # Columns
    # ------------------------------------------------------------------

    def _validate_columns(
        self,
        expression: exp.Expression,
        result: DirectSQLValidationResult,
    ) -> None:
        """
        Validate only column references Python can resolve with confidence.

        Qualified reference:
            If alias/table ownership is known, table.column must exist.
            This is a blocking, provable schema fact.

        Unqualified reference:
            Resolve it when exactly one local table owns the column.
            Otherwise emit a warning and let MySQL resolve SQL scope.

        This deliberately avoids blocking valid correlated or nested SQL merely
        because Python/SQLGlot cannot prove the intended scope.
        """
        for scope in traverse_scope(expression):
            scope_expression = scope.expression

            if not isinstance(scope_expression, exp.Select):
                continue

            alias_to_table = self._aliases_for_select(scope_expression)
            scope_tables = list(dict.fromkeys(alias_to_table.values()))

            for column in scope.columns:
                column_name = _clean_identifier(column.name)

                if not column_name:
                    continue

                table_qualifier = _clean_identifier(column.table)

                if table_qualifier:
                    table_name = alias_to_table.get(table_qualifier)

                    if table_name is None:
                        table_name = self._resolve_outer_alias(
                            select=scope_expression,
                            alias=table_qualifier,
                        )

                    if table_name is None:
                        _add_issue(
                            result,
                            DirectSQLValidationCategory.UNRESOLVED_COLUMN_SCOPE,
                            (
                                "Could not confidently resolve qualified column "
                                f"{table_qualifier}.{column_name}; MySQL will "
                                "determine the actual scope."
                            ),
                            warning=True,
                        )
                        continue

                    qualified = f"{table_name}.{column_name}"

                    _add_unique(
                        result.columns,
                        qualified,
                    )

                    if not self._schema_graph.has_column(
                        table_name,
                        column_name,
                    ):
                        # Blocking because ownership is known and the schema
                        # can prove the qualified column does not exist.
                        _add_issue(
                            result,
                            DirectSQLValidationCategory.UNDOCUMENTED_COLUMN,
                            f"Undocumented column: {qualified}.",
                        )

                    continue

                owners = [
                    table_name
                    for table_name in scope_tables
                    if self._schema_graph.has_column(
                        table_name,
                        column_name,
                    )
                ]

                if len(owners) == 1:
                    qualified = f"{owners[0]}.{column_name}"

                    _add_unique(
                        result.columns,
                        qualified,
                    )
                    continue

                if len(owners) > 1:
                    _add_issue(
                        result,
                        DirectSQLValidationCategory.AMBIGUOUS_COLUMN,
                        (
                            f"Unqualified column {column_name} could refer to "
                            "multiple local tables: "
                            + ", ".join(sorted(owners))
                            + ". MySQL will resolve or reject the reference."
                        ),
                        warning=True,
                    )
                    continue

                # No local owner does not prove the SQL is invalid: the column
                # may be a legal correlated reference to an outer scope.
                _add_issue(
                    result,
                    DirectSQLValidationCategory.UNRESOLVED_COLUMN_SCOPE,
                    (
                        f"Unqualified column {column_name} was not uniquely "
                        "resolved in the local scope; MySQL will resolve or "
                        "reject the reference."
                    ),
                    warning=True,
                )

    @classmethod
    def _resolve_outer_alias(
        cls,
        *,
        select: exp.Select,
        alias: str,
    ) -> str | None:
        """Resolve a qualified correlated alias against ancestor SELECT scopes."""
        current: exp.Expression | None = select.parent

        while current is not None:
            if isinstance(current, exp.Select):
                aliases = cls._aliases_for_select(current)
                table_name = aliases.get(alias)

                if table_name:
                    return table_name

            current = current.parent

        return None

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    def _validate_joins(
        self,
        expression: exp.Expression,
        result: DirectSQLValidationResult,
    ) -> None:
        """
        Validate only explicit JOIN edges that can be resolved confidently.

        If both sides of a local column-equality JOIN are resolved, the
        documented SchemaGraph relationship is a hard fact and may block.

        If aliases, scope, nested expressions, or JOIN structure cannot be
        resolved confidently, emit a warning and let MySQL decide.
        """
        for select in expression.find_all(exp.Select):
            alias_to_table = self._aliases_for_select(select)

            for join in (
                select.args.get(
                    "joins",
                    [],
                )
                or []
            ):
                on_expression = join.args.get("on")

                if on_expression is None:
                    # A JOIN without ON may be valid for CROSS/NATURAL syntax.
                    # Do not block merely because this validator cannot model it.
                    _add_issue(
                        result,
                        DirectSQLValidationCategory.JOIN_WITHOUT_ON,
                        (
                            "JOIN has no explicit ON predicate; Python did not "
                            "validate a relationship edge. MySQL will determine "
                            "whether the JOIN syntax is valid."
                        ),
                        warning=True,
                    )
                    continue

                local_equalities = [
                    condition
                    for condition in on_expression.find_all(exp.EQ)
                    if condition.find_ancestor(exp.Select) is select
                ]

                saw_resolved_relationship = False
                saw_uncertain_relationship = False

                for condition in local_equalities:
                    left = condition.this
                    right = condition.expression

                    if not isinstance(left, exp.Column) or not isinstance(
                        right,
                        exp.Column,
                    ):
                        # Literal/date/value predicates are not relationships.
                        continue

                    left_alias = _clean_identifier(left.table)
                    right_alias = _clean_identifier(right.table)

                    if not left_alias or not right_alias:
                        saw_uncertain_relationship = True
                        continue

                    if (
                        left_alias not in alias_to_table
                        or right_alias not in alias_to_table
                    ):
                        saw_uncertain_relationship = True
                        continue

                    saw_resolved_relationship = True

                    self._validate_join_edge(
                        left=left,
                        right=right,
                        alias_to_table=alias_to_table,
                        result=result,
                    )

                if not saw_resolved_relationship:
                    _add_issue(
                        result,
                        DirectSQLValidationCategory.UNSUPPORTED_JOIN,
                        (
                            "Python could not prove a local documented JOIN edge "
                            "for this ON expression; MySQL will resolve or reject "
                            "the JOIN."
                        ),
                        warning=True,
                    )
                elif saw_uncertain_relationship:
                    _add_issue(
                        result,
                        DirectSQLValidationCategory.UNSUPPORTED_JOIN,
                        (
                            "Part of the JOIN predicate could not be resolved "
                            "deterministically; validated relationship edges were "
                            "kept and MySQL will evaluate the remaining predicate."
                        ),
                        warning=True,
                    )

    @staticmethod
    def _aliases_for_select(
        select: exp.Select,
    ) -> dict[str, str]:
        aliases: dict[str, str] = {}

        # Only collect tables owned by this SELECT, excluding nested SELECTs.
        for table in select.find_all(exp.Table):
            nearest_select = table.find_ancestor(exp.Select)

            if nearest_select is not select:
                continue

            table_name = _clean_identifier(table.name)

            alias = _clean_identifier(table.alias_or_name)

            if table_name:
                aliases[table_name] = table_name

            if alias:
                aliases[alias] = table_name

        return aliases

    def _validate_join_edge(
        self,
        *,
        left: exp.Column,
        right: exp.Column,
        alias_to_table: dict[str, str],
        result: DirectSQLValidationResult,
    ) -> None:
        """
        Hard-check a JOIN edge only when both table aliases are locally known.

        This method is called only for a relationship Python can resolve
        deterministically.
        """
        left_alias = _clean_identifier(left.table)
        right_alias = _clean_identifier(right.table)

        left_column = _clean_identifier(left.name)
        right_column = _clean_identifier(right.name)

        left_table = alias_to_table.get(left_alias)
        right_table = alias_to_table.get(right_alias)

        if not left_table or not right_table:
            _add_issue(
                result,
                DirectSQLValidationCategory.UNRESOLVED_JOIN_ALIAS,
                (
                    "Could not confidently resolve JOIN aliases "
                    f"{left_alias}, {right_alias}; MySQL will determine scope."
                ),
                warning=True,
            )
            return

        relationship_text = (
            f"{left_table}.{left_column} -> " f"{right_table}.{right_column}"
        )

        _add_unique(
            result.relationships,
            relationship_text,
        )

        # These are blocking because the table ownership is explicit.
        if not self._schema_graph.has_column(
            left_table,
            left_column,
        ):
            _add_issue(
                result,
                DirectSQLValidationCategory.UNDOCUMENTED_COLUMN,
                f"Undocumented JOIN column: {left_table}.{left_column}.",
            )
            return

        if not self._schema_graph.has_column(
            right_table,
            right_column,
        ):
            _add_issue(
                result,
                DirectSQLValidationCategory.UNDOCUMENTED_COLUMN,
                f"Undocumented JOIN column: {right_table}.{right_column}.",
            )
            return

        relationships = self._schema_graph.relationships_between(
            left_table,
            right_table,
        )

        valid = any(
            (
                relationship.left_table == left_table
                and relationship.left_column == left_column
                and relationship.right_table == right_table
                and relationship.right_column == right_column
            )
            or (
                relationship.left_table == right_table
                and relationship.left_column == right_column
                and relationship.right_table == left_table
                and relationship.right_column == left_column
            )
            for relationship in relationships
        )

        if not valid:
            # Blocking because both sides are explicit, columns exist, and
            # SchemaGraph can prove this particular edge is undocumented.
            _add_issue(
                result,
                DirectSQLValidationCategory.UNDOCUMENTED_RELATIONSHIP,
                (
                    "JOIN relationship is not documented in SchemaGraph: "
                    f"{relationship_text}."
                ),
            )


def _self_test_correlated_scope_shape() -> None:
    """
    Regression-test SQLGlot ownership assumptions used by the validator.

    This verifies that the correlated predicate belongs to the inner SELECT,
    while the actual JOIN predicates belong to the outer SELECT.
    """
    sql = """
    SELECT
        t1.employee_id,
        MAX(t2.clock_in_time) AS most_recent_clock_in_time
    FROM employees AS t1
    LEFT JOIN time_clock_summary AS t2
        ON t1.employee_id = t2.employee_id
       AND t2.work_date = (
            SELECT MAX(work_date)
            FROM time_clock_summary
            WHERE employee_id = t1.employee_id
       )
    GROUP BY t1.employee_id
    """

    expression = sqlglot.parse_one(sql, read="mysql")

    selects = list(expression.find_all(exp.Select))

    assert len(selects) == 2

    outer_select = selects[0]
    inner_select = selects[1]

    outer_joins = outer_select.args.get("joins", []) or []

    assert len(outer_joins) == 1

    join_equalities = [
        condition
        for condition in outer_joins[0].args["on"].find_all(exp.EQ)
        if condition.find_ancestor(exp.Select) is outer_select
    ]

    assert len(join_equalities) == 1

    inner_columns = {
        _clean_identifier(column.name) for column in inner_select.find_all(exp.Column)
    }

    assert "employee_id" in inner_columns

    print("DirectSQLValidator correlated-scope self-test: PASS")


def _self_test_permissive_issue_behavior() -> None:
    """Verify warnings do not make an otherwise clean result invalid."""
    result = DirectSQLValidationResult(
        sql="SELECT 1",
    )

    _add_issue(
        result,
        DirectSQLValidationCategory.AMBIGUOUS_COLUMN,
        "uncertain scope",
        warning=True,
    )

    result.is_valid = len(result.errors) == 0

    assert result.is_valid
    assert result.errors == []
    assert result.warnings == ["uncertain scope"]

    print("DirectSQLValidator permissive-warning self-test: PASS")
