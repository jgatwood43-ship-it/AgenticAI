"""
core/state.py
─────────────
Shared workflow state for the AgenticAI four-agent pipeline.

The state object is passed through:

    Agent 1: LLMDecisionAgent
    Agent 2: RetrieverAgent
    Agent 3: GraderWriterAgent
    Agent 4: AnswerGeneratorAgent

Each user query should create a new WorkflowState instance.

This version includes two generalized contracts:

1. Evidence contract
   Describes what evidence is required and whether interpretation is needed.

2. Presentation contract
   Describes how verified evidence should be rendered after grading.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class Route(str, Enum):
    RETRIEVER = "retriever"
    ANSWER_GENERATOR = "answer_generator"


class RetrievalMode(str, Enum):
    """
    Agent 2 execution strategy after Route.RETRIEVER is selected.

    EVIDENCE_ONLY:
        Schema/policy/document retrieval without a live database query.

    DIRECT_QUERY:
        One bounded live-data lookup or report. The DirectQuery path lets the
        LLM propose read-only SQL, then validates it deterministically.

    INVESTIGATION:
        Broad or multi-step security analysis using InvestigationPlanner and
        the structured investigation pipeline.
    """

    EVIDENCE_ONLY = "evidence_only"
    DIRECT_QUERY = "direct_query"
    INVESTIGATION = "investigation"


class GradeResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"


Grade = GradeResult


class EvidenceType(str, Enum):
    POLICY = "policy"
    SCHEMA = "schema"
    DATABASE = "database"
    INVESTIGATION = "investigation"


class PresentationMode(str, Enum):
    """Rendering strategy selected after evidence grading."""

    AUTO = "auto"
    VERBATIM = "verbatim"
    SUMMARY = "summary"
    INTERPRET = "interpret"
    FAILURE = "failure"


@dataclass
class SQLPlan:
    purpose: str = ""
    database_type: str = "mysql"
    tables: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    sql: str = ""
    is_valid: bool = False
    validation_errors: list[str] = field(default_factory=list)

    def updated(self, **changes: Any) -> "SQLPlan":
        """Return a dataclass copy with selected fields replaced."""
        return replace(self, **changes)


@dataclass
class ToTThought:
    branch_id: str = ""
    reasoning: str = ""
    score: float = 0.0
    selected: bool = False


@dataclass
class TraceEntry:
    component: str
    message: str


@dataclass
class InvestigationEvidenceItem:
    """One semantic or physical evidence component tracked across agents."""

    name: str
    kind: str = "semantic"  # semantic | physical | derived
    status: str = "required"  # required | covered | missing
    source_tasks: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class InvestigationEvidenceLedger:
    """Persistent compact investigation memory shared by all four agents."""

    active: bool = False
    objective: str = ""
    items: dict[str, InvestigationEvidenceItem] = field(default_factory=dict)
    evidence_version: int = 0

    @staticmethod
    def _key(name: str) -> str:
        return " ".join(str(name or "").strip().lower().split())

    def initialize(
        self,
        objective: str,
        *,
        required_components: list[str] | None = None,
    ) -> None:
        """Activate the ledger and seed semantic evidence requirements.

        Required components are semantic investigation facts, not physical
        database identifiers. They begin as ``missing`` because no evidence has
        been retrieved yet. Downstream agents mark them covered only after
        verified evidence supports them.
        """
        self.active = True
        self.objective = str(objective or "").strip()

        for component in required_components or []:
            self.mark_missing(
                component,
                kind="semantic",
                note="Required by Agent 1 investigation contract.",
            )

    def ensure(
        self,
        name: str,
        *,
        kind: str = "semantic",
        status: str = "required",
        note: str = "",
    ) -> InvestigationEvidenceItem | None:
        key = self._key(name)
        if not key:
            return None
        item = self.items.get(key)
        if item is None:
            item = InvestigationEvidenceItem(
                name=str(name).strip(),
                kind=kind,
                status=status,
            )
            self.items[key] = item
        elif item.status != "covered" and status == "missing":
            item.status = "missing"
        if note and note not in item.notes:
            item.notes.append(note)
        return item

    def mark_covered(
        self,
        name: str,
        *,
        kind: str = "semantic",
        task_id: str = "",
        tables: list[str] | None = None,
        fields: list[str] | None = None,
        note: str = "",
    ) -> None:
        item = self.ensure(name, kind=kind, status="covered", note=note)
        if item is None:
            return
        item.status = "covered"
        if task_id and task_id not in item.source_tasks:
            item.source_tasks.append(task_id)
        for table in tables or []:
            value = str(table or "").strip()
            if value and value not in item.tables:
                item.tables.append(value)
        for field_name in fields or []:
            value = str(field_name or "").strip()
            if value and value not in item.fields:
                item.fields.append(value)
        self.evidence_version += 1

    def mark_missing(
        self, name: str, *, kind: str = "semantic", note: str = ""
    ) -> None:
        item = self.ensure(name, kind=kind, status="missing", note=note)
        if item is not None and item.status != "covered":
            item.status = "missing"

    def record_physical_result(
        self,
        *,
        task_id: str,
        tables: list[str],
        fields: list[str],
        row_count: int,
    ) -> None:
        for table in tables:
            self.mark_covered(
                f"table evidence: {table}",
                kind="physical",
                task_id=task_id,
                tables=[table],
                note=f"successful query; rows={row_count}",
            )
        for field_name in fields:
            self.mark_covered(
                f"returned field: {field_name}",
                kind="physical",
                task_id=task_id,
                fields=[field_name],
                note=f"successful query; rows={row_count}",
            )

    def missing_names(self) -> list[str]:
        return [item.name for item in self.items.values() if item.status == "missing"]

    def required_names(self) -> list[str]:
        return [item.name for item in self.items.values() if item.status == "required"]

    def covered_names(self) -> list[str]:
        return [item.name for item in self.items.values() if item.status == "covered"]

    def compact_summary(self, max_items: int = 30) -> str:
        if not self.active:
            return "(investigation ledger inactive)"
        covered = self.covered_names()[:max_items]
        missing = self.missing_names()[:max_items]
        required = [
            item.name for item in self.items.values() if item.status == "required"
        ][:max_items]
        lines = [
            f"Objective: {self.objective or '(not recorded)'}",
            f"Evidence version: {self.evidence_version}",
            "Covered semantic/physical evidence:",
            *([f"- {x}" for x in covered] or ["- (none yet)"]),
            "Still missing:",
            *([f"- {x}" for x in missing] or ["- (none currently recorded)"]),
            "Required but not yet classified:",
            *([f"- {x}" for x in required] or ["- (none)"]),
            "Important: semantic/derived names above are reasoning concepts, not database identifiers.",
        ]
        return "\n".join(lines)

    def fingerprint(
        self,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Return meaningful ledger state for no-progress detection."""
        return (
            tuple(sorted(self.covered_names())),
            tuple(sorted(self.missing_names())),
            tuple(sorted(self.required_names())),
        )


@dataclass
class InvestigationTaskResult:
    task_id: str = ""
    title: str = ""
    question: str = ""
    sql_plan_valid: bool = False
    sql_query: str = ""
    database_query_succeeded: bool = False
    database_error: str | None = None
    database_evidence: str = ""
    database_rows: list[dict[str, Any]] = field(default_factory=list)
    database_columns: list[str] = field(default_factory=list)
    database_row_count: int = 0
    required_tables: list[str] = field(default_factory=list)
    requested_outputs: list[str] = field(default_factory=list)


@dataclass
class WorkflowState:
    query: str = ""
    original_query: str = ""
    resolved_query: str = ""
    query_was_resolved: bool = False

    # Evidence contract
    required_evidence_types: list[str] = field(default_factory=list)
    required_output_fields: list[str] = field(default_factory=list)
    requires_interpretation: bool = False
    evidence_requirement_reason: str = ""

    # Presentation contract
    presentation_mode: PresentationMode = PresentationMode.AUTO
    requires_complete_output: bool = False
    allow_llm_rewrite: bool = True
    allow_summary: bool = True
    allow_inference: bool = False
    presentation_reason: str = ""

    # Agent 1 — routing
    route: Route | None = None
    route_reason: str = ""
    route_confidence: float = 0.0

    # Agent 2 execution strategy chosen by Agent 1.
    retrieval_mode: RetrievalMode = RetrievalMode.EVIDENCE_ONLY
    retrieval_mode_reason: str = ""

    requires_live_data: bool = False
    requires_policy_evidence: bool = False

    # Agent 2 — policy retrieval
    policy_nodes: list[Any] = field(default_factory=list)
    policy_sources: list[str] = field(default_factory=list)
    policy_context: str = ""
    retrieved_nodes: list[Any] = field(default_factory=list)
    retrieved_context: str = ""

    # Agent 2 — schema retrieval
    schema_sources: list[str] = field(default_factory=list)
    schema_context: str = ""
    schema_evidence: str = ""
    schema_query_succeeded: bool = False
    schema_tables: list[str] = field(default_factory=list)
    schema_columns: dict[str, list[str]] = field(default_factory=dict)
    schema_relationships: list[str] = field(default_factory=list)

    # Agent 2 — investigation planning
    investigation_requested: bool = False
    investigation_succeeded: bool = False
    investigation_plan: dict[str, Any] = field(default_factory=dict)
    investigation_results: list[InvestigationTaskResult] = field(default_factory=list)
    investigation_ledger: InvestigationEvidenceLedger = field(
        default_factory=InvestigationEvidenceLedger
    )

    # Agent 2 — SQL planning
    sql_plan: SQLPlan | None = None
    sql_query: str = ""
    sql_parameters: dict[str, Any] = field(default_factory=dict)
    sql_validation_errors: list[str] = field(default_factory=list)

    # Agent 2 — database execution
    database_source: str = ""
    database_query_succeeded: bool = False
    database_error: str | None = None
    database_evidence: str = ""
    database_rows: list[dict[str, Any]] = field(default_factory=list)
    database_columns: list[str] = field(default_factory=list)
    database_row_count: int = 0

    # Agent 3 — evidence diagnostics and grading
    evidence_failure_category: str = ""
    evidence_failure_reason: str = ""
    evidence_failure_repairable: bool = False
    grade: GradeResult | None = None
    refined_context: str = ""
    tot_thoughts: list[ToTThought] = field(default_factory=list)
    tot_best_branch: str | None = None

    # Agent 4 — final response
    answer: str = ""

    # Workflow diagnostics
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    trace: list[TraceEntry] = field(default_factory=list)
    react_trace: list[str] = field(default_factory=list)

    def add_warning(self, warning: str) -> None:
        cleaned = str(warning or "").strip()
        if cleaned and cleaned not in self.warnings:
            self.warnings.append(cleaned)

    def add_trace(self, component: str, message: str) -> None:
        cleaned_component = str(component or "Workflow").strip()
        cleaned_message = str(message or "").strip()
        if not cleaned_message:
            return
        self.trace.append(
            TraceEntry(
                component=cleaned_component,
                message=cleaned_message,
            )
        )
        self.react_trace.append(f"[{cleaned_component}] {cleaned_message}")

    def reset_evidence_failure(self) -> None:
        self.evidence_failure_category = ""
        self.evidence_failure_reason = ""
        self.evidence_failure_repairable = False

    def normalize_evidence_requirements(self) -> None:
        self.required_evidence_types = list(
            dict.fromkeys(
                str(value).strip().lower()
                for value in self.required_evidence_types
                if str(value).strip()
            )
        )
        self.required_output_fields = list(
            dict.fromkeys(
                str(value).strip()
                for value in self.required_output_fields
                if str(value).strip()
            )
        )

    def normalize_presentation_contract(self) -> None:
        """Normalize presentation settings and enforce internal consistency."""
        if not isinstance(self.presentation_mode, PresentationMode):
            try:
                self.presentation_mode = PresentationMode(
                    str(self.presentation_mode).strip().lower()
                )
            except ValueError:
                self.presentation_mode = PresentationMode.AUTO

        if self.presentation_mode == PresentationMode.VERBATIM:
            self.requires_complete_output = True
            self.allow_llm_rewrite = False
            self.allow_summary = False
            self.allow_inference = False

        elif self.presentation_mode == PresentationMode.SUMMARY:
            self.allow_llm_rewrite = True
            self.allow_summary = True
            self.allow_inference = False

        elif self.presentation_mode == PresentationMode.INTERPRET:
            self.allow_llm_rewrite = True
            self.allow_summary = True
            self.allow_inference = True
            self.requires_interpretation = True

        elif self.presentation_mode == PresentationMode.FAILURE:
            self.allow_llm_rewrite = False
            self.allow_summary = False
            self.allow_inference = False

        if self.requires_interpretation:
            self.allow_inference = True

    def set_presentation_contract(
        self,
        *,
        mode: PresentationMode | str,
        reason: str = "",
        requires_complete_output: bool | None = None,
        allow_llm_rewrite: bool | None = None,
        allow_summary: bool | None = None,
        allow_inference: bool | None = None,
    ) -> None:
        """Set the Agent 3-to-Agent 4 presentation contract."""
        try:
            normalized_mode = (
                mode
                if isinstance(mode, PresentationMode)
                else PresentationMode(str(mode).strip().lower())
            )
        except ValueError:
            normalized_mode = PresentationMode.AUTO

        self.presentation_mode = normalized_mode
        self.presentation_reason = str(reason or "").strip()

        if requires_complete_output is not None:
            self.requires_complete_output = bool(requires_complete_output)
        if allow_llm_rewrite is not None:
            self.allow_llm_rewrite = bool(allow_llm_rewrite)
        if allow_summary is not None:
            self.allow_summary = bool(allow_summary)
        if allow_inference is not None:
            self.allow_inference = bool(allow_inference)

        self.normalize_presentation_contract()

    def reset_presentation_contract(self) -> None:
        self.presentation_mode = PresentationMode.AUTO
        self.requires_complete_output = False
        self.allow_llm_rewrite = True
        self.allow_summary = True
        self.allow_inference = False
        self.presentation_reason = ""

    def set_retrieval_mode(
        self,
        mode: RetrievalMode | str,
        *,
        reason: str = "",
    ) -> None:
        """Set and normalize Agent 2's execution strategy."""
        try:
            self.retrieval_mode = (
                mode
                if isinstance(mode, RetrievalMode)
                else RetrievalMode(str(mode).strip().lower())
            )
        except ValueError:
            self.retrieval_mode = RetrievalMode.EVIDENCE_ONLY

        self.retrieval_mode_reason = str(reason or "").strip()

        # Keep the legacy investigation flag synchronized for existing code.
        self.investigation_requested = (
            self.retrieval_mode == RetrievalMode.INVESTIGATION
        )

    @property
    def is_direct_query(self) -> bool:
        return (
            self.route == Route.RETRIEVER
            and self.retrieval_mode == RetrievalMode.DIRECT_QUERY
        )

    @property
    def is_investigation(self) -> bool:
        return (
            self.route == Route.RETRIEVER
            and self.retrieval_mode == RetrievalMode.INVESTIGATION
        )

    @property
    def is_evidence_only(self) -> bool:
        return (
            self.route == Route.RETRIEVER
            and self.retrieval_mode == RetrievalMode.EVIDENCE_ONLY
        )

    def reset_retrieval_state(self) -> None:
        """Clear retrieval, execution, grading, and answer results."""
        self.policy_nodes = []
        self.policy_sources = []
        self.policy_context = ""
        self.retrieved_nodes = []
        self.retrieved_context = ""

        self.schema_sources = []
        self.schema_context = ""
        self.schema_evidence = ""
        self.schema_query_succeeded = False
        self.schema_tables = []
        self.schema_columns = {}
        self.schema_relationships = []

        self.investigation_requested = False
        self.investigation_succeeded = False
        self.investigation_plan = {}
        self.investigation_results = []

        self.sql_plan = None
        self.sql_query = ""
        self.sql_parameters = {}
        self.sql_validation_errors = []

        self.database_source = ""
        self.database_query_succeeded = False
        self.database_error = None
        self.database_evidence = ""
        self.database_rows = []
        self.database_columns = []
        self.database_row_count = 0

        self.reset_evidence_failure()
        self.grade = None
        self.refined_context = ""
        self.tot_thoughts = []
        self.tot_best_branch = None
        self.answer = ""

    def available_evidence_types(self) -> set[str]:
        available: set[str] = set()

        if self.policy_context.strip():
            available.add(EvidenceType.POLICY.value)

        if self.schema_query_succeeded and self.schema_evidence.strip():
            available.add(EvidenceType.SCHEMA.value)

        if self.database_query_succeeded and self.database_evidence.strip():
            available.add(EvidenceType.DATABASE.value)

        if self.investigation_succeeded and self.investigation_results:
            available.add(EvidenceType.INVESTIGATION.value)

        return available

    def build_retrieved_context(self) -> str:
        sections: list[str] = []

        if self.policy_context.strip():
            sections.append(
                "=== POLICY EVIDENCE ===\n" f"{self.policy_context.strip()}"
            )

        if self.schema_evidence.strip():
            sections.append(
                "=== SCHEMA EVIDENCE ===\n" f"{self.schema_evidence.strip()}"
            )
        elif self.schema_context.strip():
            sections.append(
                "=== AUTHORITATIVE SCHEMA ===\n" f"{self.schema_context.strip()}"
            )

        if self.investigation_ledger.active:
            sections.append(
                "=== INVESTIGATION EVIDENCE LEDGER ===\n"
                + self.investigation_ledger.compact_summary()
            )

        if self.investigation_results:
            investigation_sections: list[str] = []

            for index, result in enumerate(
                self.investigation_results,
                start=1,
            ):
                lines = [
                    (
                        f"=== INVESTIGATION TASK {index}: "
                        f"{result.title or result.task_id} ==="
                    ),
                    f"Question: {result.question}",
                    f"Database success: {result.database_query_succeeded}",
                    f"Rows returned: {result.database_row_count}",
                    f"Columns: {result.database_columns}",
                ]

                if result.database_error:
                    lines.append(f"Error: {result.database_error}")

                if result.database_evidence.strip():
                    lines.extend(
                        [
                            "Evidence:",
                            result.database_evidence.strip(),
                        ]
                    )

                investigation_sections.append("\n".join(lines))

            sections.append("\n\n".join(investigation_sections))

        elif self.database_query_succeeded:
            sections.append(
                "=== DATABASE EVIDENCE ===\n"
                f"{self.database_evidence.strip() or '(no rows returned)'}"
            )

        elif self.database_error:
            sections.append(
                "=== DATABASE STATUS ===\n" f"Query failed: {self.database_error}"
            )

        self.retrieved_context = "\n\n".join(
            section for section in sections if section.strip()
        ).strip()

        return self.retrieved_context

    def presentation_contract_summary(self) -> dict[str, Any]:
        """Return a serializable contract summary for traces and Agent 4."""
        self.normalize_presentation_contract()

        return {
            "presentation_mode": self.presentation_mode.value,
            "requires_complete_output": self.requires_complete_output,
            "allow_llm_rewrite": self.allow_llm_rewrite,
            "allow_summary": self.allow_summary,
            "allow_inference": self.allow_inference,
            "presentation_reason": self.presentation_reason,
        }
