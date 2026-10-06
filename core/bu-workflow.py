"""
core/workflow.py
─────────────────
Multi-Agent Workflow Orchestrator
═══════════════════════════════════════════════════════════════════════════════

FULL PIPELINE GRAPH WITH FRAMEWORKS
─────────────────────────────────────────────────────────────────────────────

    ┌──────────────────────────────────────────┐
    │  Agent 1: LLMDecisionAgent               │
    │  Frameworks: ReAct + MCP                 │
    │  → Inspects query, optionally peeks at   │
    │    DB schemas via MCP, emits ROUTE token  │
    └──────────────────┬───────────────────────┘
                       │
           ┌───────────┴────────────┐
           │                        │
     route=retriever         route=answer_generator
           │                        │
           ▼                        │
    ┌──────────────────────────────────┐
    │  Agent 2: RetrieverAgent         │
    │  Frameworks: RAG + ReAct + MCP   │
    │  → Semantic pgvector ANN search  │
    │    (bge-large-en-v1.5, HNSW)     │
    │  → And  MCP DB lookups           │
    └──────────────────┬───────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────┐
    │  Agent 3: GraderWriterAgent              │
    │  Frameworks: ToT (PRIMARY) + ReAct + MCP │
    │  → Pre-ToT MCP fact-checking (ReAct)     │
    │  → ToT Phase 1: Generate 3 branches      │
    │      branch_1: strict grader             │
    │      branch_2: lenient grader            │
    │      branch_3: balanced grader           │
    │  → ToT Phase 2: LLM evaluator scores all │
    │  → ToT Phase 3: Select highest scorer    │
    └──────────────────┬───────────────────────┘
                       │
                       └──────────────┐
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  Agent 4: AnswerGeneratorAgent        │
                    │  Frameworks: ReAct + MCP              │
                    │  → RAG mode  : cites refined_context  │
                    │  → Direct mode: parametric knowledge  │
                    │  → And MCP DB enrichment             │
                    └──────────────────────────────────────┘

FRAMEWORK MAPPING SUMMARY
─────────────────────────
  ReAct : All 4 agents.  Core Thought→Action→Observe loop.
  MCP   : All 4 agents.  mysql_query + postgres_query via Database-MCP server.
  RAG   : Agent 2 (primary) + Agent 4 (consumes refined context).
  ToT   : Agent 3 only.  3-branch generate→evaluate→select pattern.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core.llm_factory import build_llm, build_embed_model, configure_llama_globals
from core.state import Route, WorkflowState
from core.vector_store import build_vector_store, build_index, build_schema_index
from core.schema_retriever import SchemaRetriever
from agents.llm_decision_agent import LLMDecisionAgent
from agents.retriever_agent import RetrieverAgent
from agents.grader_writer_agent import GraderWriterAgent
from agents.answer_generator_agent import AnswerGeneratorAgent

from core.schema_map import get_schema_map

console = Console()


class MultiAgentWorkflow:
    """
    Orchestrates the four-agent pipeline.

    Usage
    -----
        workflow = MultiAgentWorkflow()
        result   = workflow.run("What are user access security risks?")
        print(result.answer)
    """

    def __init__(self) -> None:
        console.print(
            "\n[bold cyan]╔══════════════════════════════════════════════╗[/bold cyan]"
        )
        console.print(
            "[bold cyan]║   Multi-Agent Workflow — Initialising        ║[/bold cyan]"
        )
        console.print(
            "[bold cyan]╚══════════════════════════════════════════════╝[/bold cyan]\n"
        )

        # ── Shared LLM + embeddings ──────────────────────────────────────────
        # Configures LlamaIndex global Settings so all components use Ollama
        # llama3.2 and bge-large-en-v1.5 automatically.
        console.print(
            "[cyan]▶ Configuring LLM (Ollama llama3.2) and embeddings (bge-large-en-v1.5)…[/cyan]"
        )
        configure_llama_globals()
        llm = build_llm()
        embed_model = build_embed_model()
        console.print("[green]  ✔ LLM and embedding model ready.[/green]")

        # ── Vector store / index (RAG infrastructure) ─────────────────────────
        # Connects to pgvector and builds a VectorStoreIndex.
        # This index is the backbone of the RAG retrieval in Agent 2.
        console.print("[cyan]▶ Connecting to pgvector (RAG infrastructure)…[/cyan]")
        vector_store = build_vector_store()
        index = build_index(vector_store, embed_model)
        console.print("[green]  ✔ pgvector index ready.[/green]")

        console.print("[cyan]▶ Loading schema retrieval index...[/cyan]")

        schema_index = build_schema_index(
            embed_model=embed_model,
        )

        self._schema_retriever = SchemaRetriever(
            schema_index=schema_index,
            similarity_top_k=5,
        )

        console.print("[green]  ✔ Schema retrieval index ready.[/green]")

        # ── Instantiate all four agent nodes ──────────────────────────────────
        console.print("[cyan]▶ Instantiating agent nodes…[/cyan]")

        # Agent 1: LLMDecisionAgent  (ReAct + MCP)
        console.print("  [dim]Agent 1: LLMDecisionAgent   [ReAct + MCP][/dim]")
        self._decision_agent = LLMDecisionAgent(
            llm=llm, schema_retriever=self._schema_retriever
        )

        # Agent 2: RetrieverAgent    (RAG + ReAct + MCP)
        console.print("  [dim]Agent 2: RetrieverAgent      [RAG + ReAct + MCP][/dim]")
        self._retriever_agent = RetrieverAgent(
            llm=llm, index=index, schema_retriever=self._schema_retriever
        )

        # Agent 3: GraderWriterAgent (ToT + ReAct + MCP)  ← ToT lives here
        console.print(
            "  [dim]Agent 3: GraderWriterAgent   [ToT + ReAct + MCP]  ← Tree of Thought[/dim]"
        )
        self._grader_agent = GraderWriterAgent(
            llm=llm, schema_retriever=self._schema_retriever
        )

        # Agent 4: AnswerGeneratorAgent (ReAct + MCP)
        console.print("  [dim]Agent 4: AnswerGeneratorAgent [ReAct + MCP][/dim]")
        self._answer_agent = AnswerGeneratorAgent(
            llm=llm, schema_retriever=self._schema_retriever
        )

        console.print(
            "\n[bold green]✔ All agents ready.  Workflow initialised.[/bold green]\n"
        )

    # ─────────────────────────────────────────────────────────────────────────

    async def run(self, query: str) -> WorkflowState:
        """
        Execute the full multi-agent pipeline for a single user query.

        Parameters
        ----------
        query : The user's natural-language question.

        Returns
        -------
        WorkflowState with .answer populated and full diagnostic fields:
          .route, .grade, .tot_thoughts, .tot_best_branch,
          .retrieved_nodes, .refined_context, .react_trace, .error
        """
        state = WorkflowState(query=query)

        console.print(
            f"\n[bold white on blue]  QUERY: {query}  [/bold white on blue]\n"
        )

        # ─────────────────────────────────────────────────────────────────────
        # STEP 1 — LLMDecisionAgent  (ReAct + MCP)
        # ─────────────────────────────────────────────────────────────────────
        console.rule(
            "[bold yellow]Step 1 / 4 — LLMDecisionAgent  [ReAct + MCP][/bold yellow]"
        )

        try:
            state = await self._decision_agent.run(state)
        except Exception as exc:
            state.error = f"LLMDecisionAgent failed: {exc}"
            console.print(f"[bold red]✘ {state.error}[/bold red]")
            return state

        # ─────────────────────────────────────────────────────────────────────
        # ROUTING BRANCH
        # ─────────────────────────────────────────────────────────────────────
        if state.route == Route.RETRIEVER:

            # ─────────────────────────────────────────────────────────────────
            # STEP 2 — RetrieverAgent  (RAG + ReAct + MCP)
            # ─────────────────────────────────────────────────────────────────
            console.rule(
                "[bold yellow]Step 2 / 4 — RetrieverAgent  [RAG + ReAct + MCP][/bold yellow]"
            )
            try:
                state = await self._retriever_agent.run(state)
            except Exception as exc:
                state.error = f"RetrieverAgent failed: {exc}"
                console.print(f"[bold red]✘ {state.error}[/bold red]")
                return state

            # ─────────────────────────────────────────────────────────────────
            # STEP 3 — GraderWriterAgent  (ToT + ReAct + MCP)
            # ─────────────────────────────────────────────────────────────────
            console.rule(
                "[bold yellow]Step 3 / 4 — GraderWriterAgent  [ToT + ReAct + MCP][/bold yellow]"
            )
            try:
                state = await self._grader_agent.run(state)
            except Exception as exc:
                state.error = f"GraderWriterAgent failed: {exc}"
                console.print(f"[bold red]✘ {state.error}[/bold red]")
                return state

        else:
            # Direct route: LLMDecisionAgent chose to skip retrieval
            console.print(
                "[yellow]⚡ Direct route selected — skipping Steps 2 & 3 "
                "(RetrieverAgent + GraderWriterAgent).[/yellow]"
            )

        # ─────────────────────────────────────────────────────────────────────
        # STEP 4 — AnswerGeneratorAgent  (ReAct + MCP)
        # ─────────────────────────────────────────────────────────────────────
        console.rule(
            "[bold yellow]Step 4 / 4 — AnswerGeneratorAgent  [ReAct + MCP][/bold yellow]"
        )
        try:
            state = await self._answer_agent.run(state)
        except Exception as exc:
            state.error = f"AnswerGeneratorAgent failed: {exc}"
            console.print(f"[bold red]✘ {state.error}[/bold red]")
            return state

        self._print_summary(state)
        return state

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _print_summary(state: WorkflowState) -> None:
        """Print a rich terminal summary of the completed workflow run."""

        # ── Pipeline summary table ────────────────────────────────────────────
        tbl = Table(
            title="[bold]Pipeline Run Summary[/bold]", show_lines=True, expand=False
        )
        tbl.add_column("Field", style="cyan bold", no_wrap=True)
        tbl.add_column("Value", style="white")
        tbl.add_column("Framework", style="dim yellow")

        tbl.add_row("Query", state.query, "—")
        tbl.add_row(
            "Route decision",
            state.route.value.upper() if state.route else "—",
            "ReAct + MCP  (Agent 1)",
        )
        tbl.add_row(
            "Nodes retrieved",
            str(len(state.retrieved_nodes)),
            "RAG / pgvector  (Agent 2)",
        )
        tbl.add_row(
            "Context length (pre-grade)",
            f"{len(state.retrieved_context)} chars",
            "RAG  (Agent 2 → Agent 3)",
        )

        # ToT details
        tot_winner = state.tot_best_branch or "—"
        tot_scores = ""
        if state.tot_thoughts:
            tot_scores = "  |  ".join(
                f"{t.branch_id}: {t.score:.0f}/100{'  ✔' if t.selected else ''}"
                for t in state.tot_thoughts
            )
        tbl.add_row(
            "ToT branches", f"{len(state.tot_thoughts)} generated", "ToT  (Agent 3)"
        )
        tbl.add_row("ToT scores", tot_scores or "—", "ToT evaluator  (Agent 3)")
        tbl.add_row("ToT winner", tot_winner, "ToT  (Agent 3)")
        tbl.add_row(
            "Grade",
            state.grade.value.upper() if state.grade else "—",
            "ToT + ReAct  (Agent 3)",
        )
        tbl.add_row(
            "Refined context", f"{len(state.refined_context)} chars", "ToT  (Agent 3)"
        )
        tbl.add_row(
            "Answer length", f"{len(state.answer)} chars", "ReAct + MCP  (Agent 4)"
        )
        tbl.add_row("ReAct trace entries", str(len(state.react_trace)), "All agents")

        console.print(tbl)

        # ── ToT branch detail panel ───────────────────────────────────────────
        if state.tot_thoughts:
            tot_detail = Text()
            for t in state.tot_thoughts:
                marker = "  ◀ SELECTED" if t.selected else ""
                tot_detail.append(f"  {t.branch_id}", style="bold cyan")
                tot_detail.append(f"  score={t.score:.0f}/100{marker}\n", style="white")
                snippet = t.reasoning[:200].replace("\n", " ")
                tot_detail.append(f"    {snippet}…\n\n", style="dim")
            console.print(
                Panel(
                    tot_detail,
                    title="[bold yellow]Tree of Thought Branch Details[/bold yellow]",
                    expand=False,
                )
            )

        # ── Final answer panel ────────────────────────────────────────────────
        console.print(
            Panel(
                state.answer,
                title="[bold green]✔  Final Answer[/bold green]",
                expand=False,
            )
        )
