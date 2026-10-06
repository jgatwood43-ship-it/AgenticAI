"""
core/workflow.py
────────────────
Multi-agent workflow orchestrator.

Architecture
------------
QueryResolver
    Convert context-dependent follow-ups into standalone queries.

Agent 1: LLMDecisionAgent
    Structured routing only.

Agent 2: RetrieverAgent
    Policy RAG, schema retrieval, logical SQL planning, bounded repair,
    and MCP execution.

Agent 3: GraderWriterAgent
    Evidence grading, failure classification, refined_context, and bounded
    supplemental-evidence requests when investigation coverage is incomplete.

Agent 4: AnswerGeneratorAgent
    Final user-facing writing only.

Supplemental investigation loop
-------------------------------
For investigation requests only, Agent 3 may request bounded targeted
evidence-completion passes only when Agent 3's HOLISTIC review says material
evidence for the original question is still missing. Per-task or ledger checklist
gaps alone do not trigger another pass. Each pass must add new verified evidence
to continue. Workflow stops when holistic evidence is sufficient, when no
progress is made, when the same material request repeats unchanged, or after the
absolute pass budget.

Lifecycle
---------
Expensive shared resources are initialized once:

    * LLM
    * embedding model
    * pgvector policy index
    * schema index
    * schema retriever
    * query resolver

Fresh agent wrappers and WorkflowState are created for every query.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agents.answer_generator_agent import AnswerGeneratorAgent
from agents.grader_writer_agent import GraderWriterAgent
from agents.llm_decision_agent import LLMDecisionAgent
from agents.retriever_agent import RetrieverAgent
from core.llm_factory import (
    build_embed_model,
    build_llm,
    configure_llama_globals,
)
from core.query_resolver import QueryResolver
from core.schema_retriever import SchemaRetriever
from core.state import Route, WorkflowState
from core.vector_store import (
    build_index,
    build_schema_index,
    build_vector_store,
)

console = Console()

# Maximum number of supplemental investigation evidence passes permitted after
# the initial RetrieverAgent → GraderWriterAgent cycle.
MAX_SUPPLEMENTAL_INVESTIGATION_PASSES = 3


class MultiAgentWorkflow:
    """Orchestrate the four-agent workflow."""

    def __init__(self) -> None:
        console.print(
            "\n[bold cyan]"
            "╔══════════════════════════════════════════════╗"
            "[/bold cyan]"
        )
        console.print(
            "[bold cyan]"
            "║   Multi-Agent Workflow — Initialising        ║"
            "[/bold cyan]"
        )
        console.print(
            "[bold cyan]"
            "╚══════════════════════════════════════════════╝"
            "[/bold cyan]\n"
        )

        # ── Shared LLM and embedding model ────────────────────────────────────
        console.print("[cyan]▶ Configuring LLM and embedding model…[/cyan]")

        configure_llama_globals()

        self._llm = build_llm()
        self._embed_model = build_embed_model()

        console.print("[green]  ✔ LLM and embedding model ready.[/green]")

        # ── Shared policy RAG index ───────────────────────────────────────────
        console.print("[cyan]▶ Connecting to pgvector policy index…[/cyan]")

        self._vector_store = build_vector_store()
        self._index = build_index(
            self._vector_store,
            self._embed_model,
        )

        console.print("[green]  ✔ pgvector policy index ready.[/green]")

        # ── Shared schema index and retriever ─────────────────────────────────
        console.print("[cyan]▶ Loading schema retrieval index…[/cyan]")

        self._schema_index = build_schema_index(
            embed_model=self._embed_model,
        )

        self._schema_retriever = SchemaRetriever(
            schema_index=self._schema_index,
            similarity_top_k=5,
        )

        console.print("[green]  ✔ Schema retrieval index ready.[/green]")

        # ── Shared follow-up resolver ─────────────────────────────────────────
        self._query_resolver = QueryResolver(
            llm=self._llm,
        )

        console.print("[green]  ✔ Query resolver ready.[/green]")

        console.print(
            "\n[bold green]" "✔ Shared workflow resources initialised." "[/bold green]"
        )
        console.print(
            "[bold green]"
            "✔ Every query receives fresh state and fresh agents."
            "[/bold green]\n"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Per-query construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_run_agents(
        self,
    ) -> tuple[
        LLMDecisionAgent,
        RetrieverAgent,
        GraderWriterAgent,
        AnswerGeneratorAgent,
    ]:
        """Create fresh agent wrappers for one workflow run."""
        console.print("[cyan]▶ Instantiating fresh agents for this query…[/cyan]")

        console.print(
            "  [dim]Agent 1: LLMDecisionAgent   " "[Structured Routing][/dim]"
        )
        decision_agent = LLMDecisionAgent(
            llm=self._llm,
            schema_retriever=self._schema_retriever,
        )

        console.print(
            "  [dim]Agent 2: RetrieverAgent      "
            "[Policy RAG + SQL Planner + MCP][/dim]"
        )
        retriever_agent = RetrieverAgent(
            llm=self._llm,
            index=self._index,
            schema_retriever=self._schema_retriever,
        )

        console.print(
            "  [dim]Agent 3: GraderWriterAgent   " "[Evidence Gates + ToT][/dim]"
        )
        grader_agent = GraderWriterAgent(
            llm=self._llm,
            schema_retriever=self._schema_retriever,
        )

        console.print(
            "  [dim]Agent 4: AnswerGeneratorAgent " "[Single-Pass Writer][/dim]"
        )
        answer_agent = AnswerGeneratorAgent(
            llm=self._llm,
            schema_retriever=self._schema_retriever,
        )

        console.print("[green]  ✔ Fresh agents ready.[/green]\n")

        return (
            decision_agent,
            retriever_agent,
            grader_agent,
            answer_agent,
        )

    @staticmethod
    def _normalise_query(query: str) -> str:
        """Validate and normalize a user query."""
        cleaned = str(query or "").strip()

        if not cleaned:
            raise ValueError("The query cannot be empty.")

        return cleaned

    @staticmethod
    def _new_state(
        *,
        original_query: str,
        resolved_query: str,
        query_was_resolved: bool,
    ) -> WorkflowState:
        """Create a fresh WorkflowState and preserve query diagnostics."""
        state = WorkflowState(
            query=resolved_query,
        )

        # Optional fields are populated only when present in WorkflowState.
        if hasattr(state, "original_query"):
            state.original_query = original_query

        if hasattr(state, "resolved_query"):
            state.resolved_query = resolved_query

        if hasattr(state, "query_was_resolved"):
            state.query_was_resolved = query_was_resolved

        return state

    @staticmethod
    def _set_error(
        state: WorkflowState,
        component_name: str,
        exc: Exception,
    ) -> WorkflowState:
        """Store one consistent terminal workflow error."""
        state.error = f"{component_name} failed: " f"{type(exc).__name__}: {exc}"

        console.print(f"[bold red]✘ {state.error}[/bold red]")

        return state

    @staticmethod
    def _assert_query_unchanged(
        state: WorkflowState,
        expected_query: str,
        component_name: str,
    ) -> None:
        """Prevent an agent from replacing the active resolved query."""
        actual_query = str(getattr(state, "query", "") or "").strip()

        if actual_query != expected_query:
            raise RuntimeError(
                f"{component_name} changed the active query from "
                f"{expected_query!r} to {actual_query!r}."
            )

    @staticmethod
    def _supplemental_investigation_requested(
        state: WorkflowState,
    ) -> bool:
        """
        Return True only for a repairable HOLISTIC investigation-evidence gap.

        Agent 3 owns the semantic decision that material evidence for the
        ORIGINAL user question is still required. Per-task/ledger checklist
        gaps must not independently trigger this loop. Workflow only orchestrates
        the bounded retry.
        """
        return bool(
            getattr(state, "investigation_requested", False)
            and getattr(state, "evidence_failure_repairable", False)
            and str(
                getattr(
                    state,
                    "evidence_failure_category",
                    "",
                )
                or ""
            )
            .strip()
            .lower()
            == "incomplete_evidence"
            and str(
                getattr(
                    state,
                    "evidence_failure_reason",
                    "",
                )
                or ""
            ).strip()
        )

    @staticmethod
    def _clear_supplemental_request(
        state: WorkflowState,
    ) -> None:
        """
        Clear only Agent 3's supplemental-request markers before regrading.

        Existing verified evidence and investigation results are preserved.
        """
        if hasattr(state, "evidence_failure_category"):
            state.evidence_failure_category = ""

        if hasattr(state, "evidence_failure_reason"):
            state.evidence_failure_reason = ""

        if hasattr(state, "evidence_failure_repairable"):
            state.evidence_failure_repairable = False

    # ─────────────────────────────────────────────────────────────────────────
    # Main execution
    # ─────────────────────────────────────────────────────────────────────────

    async def run(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> WorkflowState:
        """
        Execute one isolated four-agent workflow run.

        Parameters
        ----------
        query:
            Current user message.

        conversation_history:
            Recent user and assistant messages, excluding the current message.
            Used only by QueryResolver.
        """
        original_query = self._normalise_query(query)
        run_id = str(uuid4())

        # Resolve conversational references before routing.
        resolution = self._query_resolver.resolve(
            current_query=original_query,
            history=conversation_history,
        )

        resolved_query = (
            str(resolution.standalone_query or "").strip() or original_query
        )

        state = self._new_state(
            original_query=original_query,
            resolved_query=resolved_query,
            query_was_resolved=resolution.is_follow_up,
        )

        state.add_trace(
            "QueryResolver",
            (
                f"follow_up={resolution.is_follow_up}; "
                f"clarification_used="
                f"{resolution.clarification_used}; "
                f"original={original_query}; "
                f"resolved={resolved_query}; "
                f"reason={resolution.reason}"
            ),
        )

        console.print(
            f"\n[bold white on blue]"
            f"  QUERY: {resolved_query}  "
            f"[/bold white on blue]"
        )
        console.print(f"[dim]  Run ID: {run_id}[/dim]")
        console.print("[dim]  Original query: " f"{original_query}[/dim]")
        console.print(
            "[dim]  Follow-up resolved: " f"{resolution.is_follow_up}[/dim]\n"
        )

        try:
            (
                decision_agent,
                retriever_agent,
                grader_agent,
                answer_agent,
            ) = self._build_run_agents()
        except Exception as exc:
            return self._set_error(
                state,
                "Agent initialization",
                exc,
            )

        # ── Step 1: route ─────────────────────────────────────────────────────
        console.rule(
            "[bold yellow]"
            "Step 1 / 4 — LLMDecisionAgent  [Structured Routing]"
            "[/bold yellow]"
        )

        try:
            state = await decision_agent.run(state)
            self._assert_query_unchanged(
                state,
                resolved_query,
                "LLMDecisionAgent",
            )
        except Exception as exc:
            return self._set_error(
                state,
                "LLMDecisionAgent",
                exc,
            )

        if state.route not in (
            Route.RETRIEVER,
            Route.ANSWER_GENERATOR,
        ):
            console.print(
                "[yellow]⚠ Invalid route. " "Using safe RETRIEVER fallback.[/yellow]"
            )
            state.route = Route.RETRIEVER

        # ── Retrieval branch ──────────────────────────────────────────────────
        if state.route == Route.RETRIEVER:
            console.rule(
                "[bold yellow]"
                "Step 2 / 4 — RetrieverAgent  "
                "[Policy RAG + SQL Planner + MCP]"
                "[/bold yellow]"
            )

            try:
                state = await retriever_agent.run(state)
                self._assert_query_unchanged(
                    state,
                    resolved_query,
                    "RetrieverAgent",
                )
            except Exception as exc:
                return self._set_error(
                    state,
                    "RetrieverAgent",
                    exc,
                )

            console.rule(
                "[bold yellow]"
                "Step 3 / 4 — GraderWriterAgent  "
                "[Evidence Gates + ToT]"
                "[/bold yellow]"
            )

            try:
                state = await grader_agent.run(state)
                self._assert_query_unchanged(
                    state,
                    resolved_query,
                    "GraderWriterAgent",
                )
            except Exception as exc:
                return self._set_error(
                    state,
                    "GraderWriterAgent",
                    exc,
                )

            # ── Adaptive bounded supplemental investigation loop ─────────────
            supplemental_passes = 0
            previous_request = ""
            previous_ledger_fingerprint = state.investigation_ledger.fingerprint()
            stop_reason = ""

            while (
                supplemental_passes < MAX_SUPPLEMENTAL_INVESTIGATION_PASSES
                and self._supplemental_investigation_requested(state)
            ):
                supplemental_reason = str(
                    getattr(
                        state,
                        "evidence_failure_reason",
                        "",
                    )
                    or ""
                ).strip()

                normalized_request = " ".join(supplemental_reason.lower().split())

                if (
                    supplemental_passes > 0
                    and normalized_request
                    and normalized_request == previous_request
                    and state.investigation_ledger.fingerprint()
                    == previous_ledger_fingerprint
                ):
                    stop_reason = (
                        "Agent 3 repeated the same supplemental evidence request "
                        "after the prior pass; stopping for no progress."
                    )
                    state.add_warning(stop_reason)
                    state.add_trace(
                        "MultiAgentWorkflow",
                        stop_reason,
                    )
                    break

                previous_request = normalized_request
                previous_ledger_fingerprint = state.investigation_ledger.fingerprint()
                supplemental_passes += 1

                console.rule(
                    "[bold magenta]"
                    "Supplemental Investigation Evidence Pass "
                    f"{supplemental_passes} / "
                    f"{MAX_SUPPLEMENTAL_INVESTIGATION_PASSES}"
                    "[/bold magenta]"
                )

                console.print(
                    "[magenta]"
                    "Agent 3's holistic review identified material missing "
                    "evidence for the original question. Continuing targeted "
                    "evidence reasoning."
                    "[/magenta]"
                )

                console.print(
                    "[dim]" f"Supplemental request: {supplemental_reason}" "[/dim]"
                )

                state.add_trace(
                    "MultiAgentWorkflow",
                    (
                        "Starting adaptive supplemental investigation pass "
                        f"{supplemental_passes}; "
                        f"reason={supplemental_reason}"
                    ),
                )

                try:
                    (
                        state,
                        made_progress,
                        progress_reason,
                    ) = await retriever_agent.run_supplemental_investigation(
                        state,
                        supplemental_request=supplemental_reason,
                    )

                    self._assert_query_unchanged(
                        state,
                        resolved_query,
                        "RetrieverAgent supplemental investigation",
                    )

                except Exception as exc:
                    stop_reason = (
                        "Supplemental investigation retrieval failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    state.add_warning(stop_reason)
                    state.add_trace(
                        "MultiAgentWorkflow",
                        stop_reason,
                    )
                    break

                if not made_progress:
                    stop_reason = (
                        "Supplemental reasoning stopped because the pass made "
                        f"no new evidence progress. {progress_reason}"
                    )
                    state.add_warning(stop_reason)
                    state.add_trace(
                        "MultiAgentWorkflow",
                        stop_reason,
                    )
                    break

                self._clear_supplemental_request(state)

                console.rule(
                    "[bold yellow]"
                    "Step 3 / 4 — GraderWriterAgent  "
                    "[Regrade After Supplemental Evidence]"
                    "[/bold yellow]"
                )

                try:
                    state = await grader_agent.run(state)

                    self._assert_query_unchanged(
                        state,
                        resolved_query,
                        "GraderWriterAgent supplemental regrade",
                    )

                except Exception as exc:
                    return self._set_error(
                        state,
                        "GraderWriterAgent supplemental regrade",
                        exc,
                    )

                state.add_trace(
                    "MultiAgentWorkflow",
                    (
                        "Completed supplemental investigation pass "
                        f"{supplemental_passes}; new evidence was added and "
                        "Agent 3 regraded the expanded evidence set."
                    ),
                )

                if not self._supplemental_investigation_requested(state):
                    stop_reason = (
                        "Supplemental reasoning completed because Agent 3 "
                        "reported that the successful evidence is holistically "
                        "sufficient for the original question."
                    )
                    state.add_trace(
                        "MultiAgentWorkflow",
                        stop_reason,
                    )
                    break

            if (
                supplemental_passes >= MAX_SUPPLEMENTAL_INVESTIGATION_PASSES
                and self._supplemental_investigation_requested(state)
            ):
                stop_reason = (
                    "Investigation evidence remains incomplete after the "
                    f"maximum {MAX_SUPPLEMENTAL_INVESTIGATION_PASSES} "
                    "supplemental evidence passes."
                )
                state.add_warning(stop_reason)
                state.add_trace(
                    "MultiAgentWorkflow",
                    stop_reason,
                )

        else:
            console.print(
                "[yellow]⚡ Direct route selected — "
                "RetrieverAgent and GraderWriterAgent skipped.[/yellow]"
            )

        # ── Step 4: final answer ───────────────────────────────────────────────
        console.rule(
            "[bold yellow]"
            "Step 4 / 4 — AnswerGeneratorAgent  "
            "[Single-Pass Writer]"
            "[/bold yellow]"
        )

        # Only Agent 4 may produce the final answer.
        state.answer = ""

        try:
            state = await answer_agent.run(state)
            self._assert_query_unchanged(
                state,
                resolved_query,
                "AnswerGeneratorAgent",
            )
        except Exception as exc:
            return self._set_error(
                state,
                "AnswerGeneratorAgent",
                exc,
            )

        if not str(getattr(state, "answer", "") or "").strip():
            state.error = (
                "AnswerGeneratorAgent completed without producing " "a final answer."
            )
            console.print(f"[bold red]✘ {state.error}[/bold red]")
            return state

        self._print_summary(state)
        return state

    # ─────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _print_summary(
        state: WorkflowState,
    ) -> None:
        """Print a concise terminal summary."""
        table = Table(
            title="[bold]Pipeline Run Summary[/bold]",
            show_lines=True,
            expand=False,
        )

        table.add_column(
            "Field",
            style="cyan bold",
            no_wrap=True,
        )
        table.add_column(
            "Value",
            style="white",
        )
        table.add_column(
            "Component",
            style="dim yellow",
        )

        table.add_row(
            "Query",
            str(state.query),
            "QueryResolver",
        )
        table.add_row(
            "Route",
            (state.route.value.upper() if state.route else "—"),
            "Agent 1",
        )
        table.add_row(
            "Policy passages",
            str(len(getattr(state, "policy_sources", []) or [])),
            "Agent 2",
        )
        table.add_row(
            "Database success",
            str(
                bool(
                    getattr(
                        state,
                        "database_query_succeeded",
                        False,
                    )
                )
            ),
            "Agent 2",
        )
        table.add_row(
            "Database rows",
            str(
                getattr(
                    state,
                    "database_row_count",
                    0,
                )
            ),
            "Agent 2",
        )
        table.add_row(
            "Grade",
            (state.grade.value.upper() if state.grade else "—"),
            "Agent 3",
        )
        table.add_row(
            "Supplemental repairable",
            str(
                bool(
                    getattr(
                        state,
                        "evidence_failure_repairable",
                        False,
                    )
                )
            ),
            "Agent 3 / Workflow",
        )

        table.add_row(
            "Ledger version",
            str(state.investigation_ledger.evidence_version),
            "Shared state",
        )
        table.add_row(
            "Ledger missing",
            str(state.investigation_ledger.missing_names()),
            "Shared state",
        )

        table.add_row(
            "Failure category",
            str(
                getattr(
                    state,
                    "evidence_failure_category",
                    "",
                )
                or "—"
            ),
            "Agent 3",
        )
        table.add_row(
            "ToT branches",
            str(len(getattr(state, "tot_thoughts", []) or [])),
            "Agent 3",
        )
        table.add_row(
            "Refined context",
            f"{len(getattr(state, 'refined_context', '') or '')} chars",
            "Agent 3",
        )
        table.add_row(
            "Answer",
            f"{len(getattr(state, 'answer', '') or '')} chars",
            "Agent 4",
        )

        console.print(table)

        if getattr(state, "tot_thoughts", None):
            details = Text()

            for thought in state.tot_thoughts:
                marker = "  ◀ SELECTED" if thought.selected else ""

                details.append(
                    f"{thought.branch_id} "
                    f"score={thought.score:.0f}/100"
                    f"{marker}\n",
                    style="bold cyan",
                )
                details.append(
                    thought.reasoning[:250].replace(
                        "\n",
                        " ",
                    )
                    + "\n\n",
                    style="dim",
                )

            console.print(
                Panel(
                    details,
                    title="[bold yellow]ToT Details[/bold yellow]",
                    expand=False,
                )
            )

        console.print(
            Panel(
                state.answer,
                title="[bold green]✔ Final Answer[/bold green]",
                expand=False,
            )
        )
