"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  agents/answer_generator_agent.py                                           ║
║  Agent 4 – Answer Generator                                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE
───────
Synthesise the final, user-facing answer.  This is the terminal node in the
pipeline — every execution path ends here.

The agent operates in two transparent modes:
  A. RAG mode    — state.refined_context is non-empty (context was retrieved
                   AND passed the GraderWriterAgent's ToT quality gate).
                   The answer is grounded in the curated document passages.
  B. Direct mode — state.refined_context is empty (either the LLMDecisionAgent
                   chose the direct route, or the ToT grader issued a FAIL).
                   The answer is generated from the LLM's parametric knowledge.

FRAMEWORKS USED IN THIS AGENT
──────────────────────────────
┌──────────┬──────────────────────────────────────────────────────────────────┐
│ ReAct    │ PRIMARY framework for this agent.  The LLM reasons about the     │
│          │ query and refined context, optionally calling database tools for  │
│          │ any last-mile fact enrichment before composing the answer.        │
│          │                                                                   │
│          │ ReAct loop iterations:                                            │
│          │   Thought  → "I need the latest order count from MySQL."          │
│          │   Action   → mysql_query("SELECT COUNT(*) FROM orders …")         │
│          │   Observe  → {"count": 1523}                                      │
│          │   Thought  → "Good.  I can now write a complete answer."          │
│          │   Answer   → [final synthesised response]                         │
├──────────┼──────────────────────────────────────────────────────────────────┤
│ MCP      │ mysql_query and postgres_query tools available for enriching the  │
│          │ answer with live structured data (counts, dates, status values,  │
│          │ etc.) that may not be in the vector store.                        │
├──────────┼──────────────────────────────────────────────────────────────────┤
│ RAG      │ INDIRECT — the agent consumes state.refined_context which was     │
│          │ produced by the full RAG pipeline (RetrieverAgent retrieval +     │
│          │ GraderWriterAgent ToT evaluation).  It does not call the vector  │
│          │ store directly, but its answer is grounded in RAG output when    │
│          │ refined_context is available.                                     │
├──────────┼──────────────────────────────────────────────────────────────────┤
│ ToT      │ NOT USED in this agent.  Answer generation benefits from a        │
│          │ single, coherent, well-structured response rather than parallel   │
│          │ branches.  ToT was already applied upstream in GraderWriterAgent. │
└──────────┴──────────────────────────────────────────────────────────────────┘

REACT LOOP DETAIL
─────────────────
  Iteration 1  (always):
    Thought  : "The query asks about X.  The refined context says Y.
                I should verify fact Z against the database before answering."
    Action   : mysql_query / postgres_query  [MCP]   (optional)
    Observe  : [structured data]

  Final iteration:
    Thought  : "I have enough information to write a comprehensive answer."
    Answer   : [well-structured response grounded in context + live DB data]

ANSWER MODES
────────────
  RAG mode   : "Based on the retrieved documents, …"
               Cites specific facts from refined_context.
  Direct mode: Answers from LLM parametric knowledge.
               Clearly distinguishes from DB-backed answers.

TERMINAL OUTPUT
───────────────
  Prints mode detection (RAG vs Direct), each ReAct iteration's
  Thought/Action/Observation (via verbose=True), and a final summary
  with answer length and a preview of the first 200 chars.
"""
from __future__ import annotations

from llama_index.core.agent.workflow import ReActAgent
# from llama_index.core.agent import ReActAgent
from llama_index.core.llms import LLM

from core.state import WorkflowState
from tools.mcp_tools import ALL_MCP_TOOLS
from config.settings import settings

# ─────────────────────────────────────────────────────────────────────────────
# System prompt for the answer generation ReAct agent.
# Two-mode design is reflected in the prompt so the LLM understands both paths.
# ─────────────────────────────────────────────────────────────────────────────
_ANSWER_SYSTEM_PROMPT = """\
You are the ANSWER GENERATOR agent — the final step of a multi-agent RAG pipeline.

Your job is to produce a clear, accurate, well-structured answer to the user's
question.

You will receive:
  QUERY   : the original user question.
  CONTEXT : curated document passages (may be empty if retrieval was skipped
            or failed the quality gate).

Answering rules
───────────────
  • If CONTEXT is non-empty  → ground your answer in it.  Cite specific facts.
    Explicitly note when you are drawing from the provided context vs. your
    own knowledge.
  • If CONTEXT is empty      → answer from your own knowledge.  Be honest about
    the source and any uncertainty.
  • Never hallucinate facts or invent citations.
  • If the answer requires live data (counts, dates, statuses), use the
    available database tools to look it up.

Available tools
───────────────
  mysql_query     : Query the MySQL database (MCP) for live structured data.
  postgres_query  : Query the Postgres database (MCP) for live data.

Output format
─────────────
  Write a clear, concise answer in natural language.
  Use markdown headings or bullet points only if the question clearly calls for
  a structured list or comparison.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent class
# ─────────────────────────────────────────────────────────────────────────────

class AnswerGeneratorAgent:
    """
    Agent 4 – Answer Generator.

    Terminal node of the multi-agent pipeline.  Uses ReAct to compose a final
    answer, optionally enriched with live database data via MCP tools.

    Frameworks
    ----------
    ReAct : Core reasoning loop for answer composition + optional MCP calls.
    MCP   : mysql_query / postgres_query for last-mile data enrichment.
    RAG   : Indirect — consumes refined_context from the RAG pipeline.
    """

    def __init__(self, llm: LLM) -> None:
        # ── [ReAct] Initialise the ReActAgent with MCP tools ──────────────────
        # Like the other agents, verbose=True causes every Thought / Action /
        # Observation in the answer-composition loop to be printed to the terminal.
        print("\n[AnswerGeneratorAgent] ⚙  Initialising ReActAgent (ReAct + MCP)…")
        print("[AnswerGeneratorAgent]    Tools: mysql_query (MCP), postgres_query (MCP)")
        print("[AnswerGeneratorAgent]    ToT  : ✗  Not used — single coherent answer preferred")
        print("[AnswerGeneratorAgent]    RAG  : ✔  Consumes refined_context from GraderWriterAgent")

        self._agent = ReActAgent(
    tools=ALL_MCP_TOOLS,
    llm=llm,
    max_iterations=settings.react_max_iterations,
    verbose=True,
    # system_prompt=_DECISION_SYSTEM_PROMPT,
    system_prompt=_ANSWER_SYSTEM_PROMPT,
        )
        # self._agent = ReActAgent.from_tools(
          #  tools=ALL_MCP_TOOLS,
          #  llm=llm,
          #  max_iterations=settings.react_max_iterations,
          #  verbose=True,          # prints all Thought/Action/Observe to terminal
          #  system_prompt=_ANSWER_SYSTEM_PROMPT,
        #)

    # ─────────────────────────────────────────────────────────────────────────

    async def run(self, state: WorkflowState) -> WorkflowState:
        """
        Generate the final answer using ReAct with optional MCP tool calls.

        Mode detection
        --------------
        Checks state.refined_context to determine whether to operate in
        RAG mode (context-grounded) or Direct mode (parametric knowledge).

        ReAct flow
        ----------
        1. LLM Thought: review the query and context; decide if a DB lookup is needed.
        2. LLM Action : mysql_query / postgres_query (optional) [MCP]
        3. LLM Observe: live data returned.
        4. Steps 2-3 repeat up to max_iterations.
        5. LLM generates the final answer [RAG-grounded if context present].

        Consumes : state.query, state.refined_context, state.route, state.grade
        Produces : state.answer, state.react_trace (appended)
        """
        print("\n" + "═" * 70)
        print("[AnswerGeneratorAgent] ▶  STARTING — Agent 4: Answer Generator")
        print(f"[AnswerGeneratorAgent]    Framework : ReAct + MCP")
        print(f"[AnswerGeneratorAgent]    ToT       : ✗  Not applied here")

        # ── Detect operating mode ─────────────────────────────────────────────
        # RAG mode  : refined_context was produced by the full retrieval +
        #             ToT grading pipeline and is available for grounding.
        # Direct mode: context is empty — either the routing decision skipped
        #              retrieval, or the GraderWriterAgent's ToT issued a FAIL.
        has_rag_context = bool(state.refined_context.strip())
        mode_label      = "RAG (context-grounded)" if has_rag_context else "DIRECT (parametric knowledge)"

        print(f"[AnswerGeneratorAgent]    Mode      : {mode_label}")

        if has_rag_context:
            # [RAG] Inform user that the answer will cite the curated context
            print(f"[AnswerGeneratorAgent] [RAG] refined_context available ({len(state.refined_context)} chars)")
            print("[AnswerGeneratorAgent] [RAG] Answer will be grounded in retrieved + ToT-graded passages.")
        else:
            if state.grade is not None:
                print("[AnswerGeneratorAgent] [RAG] GraderWriterAgent issued FAIL — context discarded.")
            else:
                print("[AnswerGeneratorAgent]       Direct route taken — retrieval was skipped.")
            print("[AnswerGeneratorAgent]       Answer will be generated from LLM parametric knowledge.")

        print(f"[AnswerGeneratorAgent]    Query     : {state.query}")
        print("─" * 70)

        # ── Build the context block for the prompt ────────────────────────────
        # When context is available (RAG mode), it is injected into the prompt
        # so the ReAct agent can draw from it when composing the answer.
        if has_rag_context:
            context_block = (
                "CONTEXT (retrieved from document store, quality-verified by ToT grader):\n"
                f"{state.refined_context}"
            )
        else:
            context_block = (
                "CONTEXT: (none)\n"
                "No relevant context was retrieved. Answer from your own knowledge.\n"
                "Be transparent about the source of your answer."
            )

        # ── [ReAct] Start Thought → Action → Observe loop ─────────────────────
        # The agent will:
        #   1. Reason about whether a DB lookup is needed.
        #   2. Optionally call mysql_query / postgres_query [MCP].
        #   3. Compose a final grounded answer.
        # All Thought/Action/Observation steps are printed by verbose=True.
        print("[AnswerGeneratorAgent] [ReAct] Starting Thought → Action → Observe loop…")
        print("[AnswerGeneratorAgent] [MCP]   Tools armed: mysql_query, postgres_query")
        print("[AnswerGeneratorAgent]         (agent will call these if live data is needed)")

        prompt = (
            f"QUERY: {state.query}\n\n"
            f"{context_block}\n\n"
            "Please generate a comprehensive, accurate answer to the query.\n"
            "If you need live data from a database to complete or verify your answer, "
            "use the available tools."
        )

        # ── [ReAct + RAG + MCP] Execute answer generation ─────────────────────
        # Inside agent.chat():
        #   • The LLM reasons using the prompt (which contains RAG context if available).
        #   • It may call MCP tools for live enrichment.
        #   • It produces the final answer text.
        # response      = self._agent.chat(prompt)
        response = await self._agent.run(user_msg=prompt)
        state.answer  = str(response)

        if hasattr(response, "response"):
            state.answer = str(response.response)
        elif hasattr(response, "message"):
            state.answer = str(response.message)
        else:
            state.answer = str(response)

            if state.react_trace is None:
               state.react_trace = []

        # Append trace entry with mode, route, and grade for full observability
        state.react_trace.append(
            f"[AnswerGeneratorAgent][ReAct]\n"
            f"  Mode       : {mode_label}\n"
            f"  Route      : {state.route.value if state.route else 'N/A'}\n"
            f"  Grade      : {state.grade.value if state.grade else 'N/A'}\n"
            f"  Answer len : {len(state.answer)} chars\n"
            f"  Preview    : {state.answer[:200]}"
        )

        # ── Terminal summary ──────────────────────────────────────────────────
        print("─" * 70)
        print(f"[AnswerGeneratorAgent] ✔  COMPLETE")
        print(f"[AnswerGeneratorAgent]    Mode         : {mode_label}")
        print(f"[AnswerGeneratorAgent]    Answer length: {len(state.answer)} chars")
        print(f"[AnswerGeneratorAgent]    Preview      : {state.answer[:120].replace(chr(10), ' ')}…")
        print("═" * 70 + "\n")

        return state
