"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  agents/grader_writer_agent.py                                              ║
║  Agent 3 – Grader / Writer                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE
───────
Evaluate the quality of the retrieved context and produce a clean, refined
version of it for the AnswerGeneratorAgent.  This is the most cognitively
complex step in the pipeline — it must balance relevance, completeness, and
accuracy — which makes it the ideal home for Tree of Thought (ToT).

FRAMEWORKS USED IN THIS AGENT
──────────────────────────────
┌──────────┬──────────────────────────────────────────────────────────────────┐
│ ToT      │ PRIMARY framework for grading and rewriting.                     │
│ (Tree of │                                                                  │
│ Thought) │ The agent generates NUM_BRANCHES (default 3) independent thought │
│          │ branches, each taking a different strategy for:                  │
│          │   • How strictly to grade the context.                           │
│          │   • How aggressively to rewrite/filter it.                       │
│          │ Each branch is scored on relevance, completeness, and fidelity.  │
│          │ The highest-scoring branch is selected and its refined_context   │
│          │ is passed downstream.                                            │
│          │                                                                  │
│          │ WHY ToT HERE?                                                    │
│          │ Grading is inherently ambiguous: context that is partially        │
│          │ relevant, incorrectly phrased, or missing key details is hard to │
│          │ evaluate with a single linear chain of thought.  Exploring        │
│          │ multiple grading strategies in parallel and picking the best one  │
│          │ is exactly the problem ToT was designed for.                     │
├──────────┼──────────────────────────────────────────────────────────────────┤
│ ReAct    │ WRAPS the ToT loop.  A single ReActAgent manages the overall     │
│          │ agent turn.  Before entering the ToT branching phase, it can     │
│          │ call MCP tools to fact-check context against live database data. │
│          │ After ToT selects a winner, the ReAct agent finalises the output.│
├──────────┼──────────────────────────────────────────────────────────────────┤
│ MCP      │ mysql_query, mysql_describe_table, and mysql_list_tables tools let the agent cross-check   │
│          │ facts in the retrieved context against authoritative DB records  │
│          │ before grading.  E.g. if context mentions a user ID, the agent   │
│          │ can verify it exists in MySQL.                                   │
├──────────┼──────────────────────────────────────────────────────────────────┤
│ RAG      │ NOT USED directly.  This agent receives the already-retrieved    │
│          │ context from RetrieverAgent via the shared state.                │
└──────────┴──────────────────────────────────────────────────────────────────┘

TREE OF THOUGHT ALGORITHM (as implemented here)
────────────────────────────────────────────────
  Phase 1 – GENERATE BRANCHES
  ─────────────────────────────
  For each branch i in [1 .. NUM_BRANCHES]:
    Prompt the LLM with a different grading strategy persona:
      Branch 1 → "Strict grader: accept only highly relevant chunks."
      Branch 2 → "Lenient grader: accept any tangentially related chunks."
      Branch 3 → "Balanced grader: accept relevant chunks, rewrite gaps."
    The LLM produces:
      • A GRADE (pass/fail) for the context.
      • A REFINED_CONTEXT (rewritten passage if grade=pass).
      • A CONFIDENCE score (0.0 – 1.0) for its own assessment.

  Phase 2 – EVALUATE BRANCHES
  ─────────────────────────────
  A separate LLM call acts as the ToT evaluator.
  It receives all branch outputs and scores each on:
    • Relevance to the query          (0-40 points)
    • Completeness of refined context (0-30 points)
    • Factual fidelity to source      (0-30 points)
  Returns a JSON score card.

  Phase 3 – SELECT BEST BRANCH
  ─────────────────────────────
  The branch with the highest total score is selected.
  Its grade and refined_context propagate into WorkflowState.
  All branches (including losers) are stored in state.tot_thoughts
  for full observability.

TERMINAL OUTPUT
───────────────
  Prints a header, each ToT branch generation, branch scores,
  the winning branch selection, the final grade, and a footer summary.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from llama_index.core.agent.workflow import ReActAgent

# from llama_index.core.agent import ReActAgent
from llama_index.core.llms import LLM

from core import schema_map, state
from core.state import GradeResult, ToTThought, WorkflowState
from tools.mcp_tools import MYSQL_TOOLS
from config.settings import settings as db_config
from core.schema_retriever import SchemaRetriever

from core.system_prompts import GRADER_SYSTEM_PROMPT

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Number of independent thought branches the ToT phase will generate.
# More branches = higher quality but more LLM calls.
NUM_BRANCHES: int = 3

# ─────────────────────────────────────────────────────────────────────────────
# Branch strategy definitions
# ─────────────────────────────────────────────────────────────────────────────

# Each branch uses a different grading persona / strategy.
# This diversity is the key insight of Tree of Thought: exploring the problem
# from multiple angles before committing to a single answer.
BRANCH_STRATEGIES = [
    {
        "id": "branch_1",
        "persona": "STRICT GRADER",
        "strategy": (
            "You are a strict quality assessor. Only grade as PASS if the context "
            "directly and completely answers the query with verifiable evidence. "
            "When rewriting, keep only sentences with clear factual relevance. "
            "Err on the side of FAIL if uncertain."
        ),
    },
    {
        "id": "branch_2",
        "persona": "LENIENT GRADER",
        "strategy": (
            "You are a broad-coverage assessor. Grade as PASS if the context "
            "contains ANY information related to the query topic, even tangentially. "
            "When rewriting, preserve all potentially useful sentences. "
            "Err on the side of PASS unless context is completely off-topic."
        ),
    },
    {
        "id": "branch_3",
        "persona": "BALANCED GRADER",
        "strategy": (
            "You are a balanced quality assessor. Grade PASS if at least 50% of "
            "the context is relevant. When rewriting, filter out irrelevant sentences "
            "but supplement gaps with clear notes about missing information. "
            "Aim for the most useful context for the downstream answer generator."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_BRANCH_PROMPT_TEMPLATE = """\
You are acting as: {persona}
Strategy: {strategy}

──────────────────────────────────────────────────────────────────────────────
QUERY:
{query}

RETRIEVED CONTEXT:
{context}
──────────────────────────────────────────────────────────────────────────────

Apply your grading strategy and output ALL of the following tokens:

GRADE: <pass|fail>
CONFIDENCE: <0.0 to 1.0>
REFINED_CONTEXT: <your rewritten/filtered context, or "(none)" if grade=fail>

Rules:
- GRADE      : "pass" or "fail" only.
- CONFIDENCE : your self-assessed certainty in this grade (1.0 = very certain).
- REFINED_CONTEXT : rewritten text if pass; exactly "(none)" if fail.
                    Preserve key facts verbatim where possible.
Do not output anything after REFINED_CONTEXT.
"""

_EVALUATOR_PROMPT_TEMPLATE = """\
You are a TREE OF THOUGHT evaluator.  You will receive {n} candidate grading
branches and must score each one.

QUERY: {query}

ORIGINAL CONTEXT:
{original_context}

CANDIDATE BRANCHES:
{branches_text}

For each branch, assign points (integers) as follows:
  relevance_score    : 0-40  (how relevant is the refined context to the query?)
  completeness_score : 0-30  (does it preserve enough information to answer the query?)
  fidelity_score     : 0-30  (does it accurately represent the original context?)
  total              : sum of the three scores

Output ONLY valid JSON in this exact format — no other text:
{{
  "scores": [
    {{"branch_id": "branch_1", "relevance": 0, "completeness": 0, "fidelity": 0, "total": 0}},
    {{"branch_id": "branch_2", "relevance": 0, "completeness": 0, "fidelity": 0, "total": 0}},
    {{"branch_id": "branch_3", "relevance": 0, "completeness": 0, "fidelity": 0, "total": 0}}
  ],
  "best_branch_id": "branch_X",
  "rationale": "one sentence explaining why this branch wins"
}}
"""

_GRADER_SYSTEM_PROMPT = """\
You are the USER ACCESS CYBERSECURITY GRADER/WRITER COORDINATOR agent in a multi-agent RAG pipeline.

Your job is to manage the Tree of Thought (ToT) grading process:
  1. For each branch strategy provided, evaluate the retrieved context and
     produce a grade + refined context.
  2. You may use MCP database tools to cross-check facts before grading.
  3. After all branches are generated, select the best one based on evaluation scores.

You have access to MySQL tools only: mysql_query and mysql_list_tables and mysql_describe_table.
Use MySQL only to verify deterministic structured data.
Do not query Postgres directly.
Document evidence from Postgres must come only from the retrieved RAG context.

"STRUCTURED DATA RULES:\n"
"- For employee, title, department, role, or access-list questions, use MySQL tools.\n"
"- Never guess column names.\n"
"- Before writing a SELECT query, first call mysql_list_tables if table names are unknown.\n"
"- Then call mysql_describe_table for the most relevant table.\n"
"- If a table contains an ID field such as job_title_id, role_id, department_id, or user_id, look for a related lookup table before answering.\n"
"- Do not stop after finding an ID field. Resolve the ID to the human-readable name when possible.\n"
"- If a query fails because of an unknown column, do not repeat the same query. Inspect the schema and correct the query.\n\n"


After one successful verification tool call, immediately return a final answer.
Do not call another tool unless the result is empty, malformed, or directly contradicts the context.

If the question is conversational, asks about the assistant's purpose, or does not contain factual claims requiring database verification, return:
NO_VERIFICATION_NEEDED
"""

# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

_GRADE_RE = re.compile(r"GRADE:\s*(pass|fail)", re.IGNORECASE)
_CONF_RE = re.compile(r"CONFIDENCE:\s*([0-9.]+)", re.IGNORECASE)
_REFINED_RE = re.compile(r"REFINED_CONTEXT:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _parse_branch_output(text: str) -> Tuple[GradeResult, float, str]:
    """
    Parse the three structured tokens from a ToT branch LLM response.

    Returns (grade, confidence, refined_context).
    """
    grade_match = _GRADE_RE.search(text)
    conf_match = _CONF_RE.search(text)
    refined_match = _REFINED_RE.search(text)

    grade = (
        GradeResult.PASS
        if grade_match and grade_match.group(1).lower() == "pass"
        else GradeResult.FAIL
    )
    confidence = float(conf_match.group(1)) if conf_match else 0.5

    if refined_match:
        rc = refined_match.group(1).strip()
        refined_context = "" if rc.lower().startswith("(none)") else rc
    else:
        refined_context = ""

    return grade, confidence, refined_context


def _parse_evaluator_scores(json_text: str, num_branches: int) -> dict:
    """
    Parse the JSON score card produced by the ToT evaluator LLM call.
    Falls back to sequential scoring if JSON is malformed.
    """
    # Strip any accidental markdown fences
    clean = re.sub(r"```json|```", "", json_text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Fallback: assign descending scores so we always pick branch_1
        return {
            "scores": [
                {
                    "branch_id": f"branch_{i+1}",
                    "relevance": 30 - i * 5,
                    "completeness": 25 - i * 5,
                    "fidelity": 20 - i * 5,
                    "total": 75 - i * 15,
                }
                for i in range(num_branches)
            ],
            "best_branch_id": "branch_1",
            "rationale": "Fallback scoring due to JSON parse error.",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Agent class
# ─────────────────────────────────────────────────────────────────────────────


class GraderWriterAgent:
    """
    Agent 3 – Grader / Writer.

    Implements Tree of Thought (ToT) grading with NUM_BRANCHES parallel
    reasoning paths, evaluated and ranked by a separate LLM judge call.
    The winning branch's grade and refined context propagate downstream.

    Frameworks
    ----------
    ToT   : Multi-branch generation + evaluation + selection (PRIMARY).
    ReAct : Outer agent loop allowing MCP fact-checking before ToT.
    MCP   : mysql_query / postgres_query for cross-referencing facts.
    """

    def __init__(self, llm: LLM, schema_retriever: SchemaRetriever) -> None:
        # Store the LLM directly for custom ToT prompting calls
        self._llm = llm

        # ── [ReAct] Build a ReActAgent for MCP-backed fact-checking ───────────
        # This ReActAgent is used in the pre-ToT phase to let the agent
        # optionally verify facts in the retrieved context against live DBs.
        print("\n[GraderWriterAgent] ⚙  Initialising ReActAgent for MCP fact-checking…")
        print("[GraderWriterAgent]    MCP tools: mysql_query, postgres_query")
        print(f"[GraderWriterAgent]    ToT branches to generate: {NUM_BRANCHES}")

        # Add this initialization line so the pre-ToT phase has its engine
        self._react_agent = ReActAgent(
            tools=MYSQL_TOOLS,
            llm=llm,
            max_iterations=db_config.react_max_iterations,
            verbose=True,
            schema_map=schema_map,
            schema_retriever=schema_retriever,
            # Give it a short, targeted system prompt for fact verification
            system_prompt=GRADER_SYSTEM_PROMPT,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ToT Phase 1: Generate branches
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_tot_branches(
        self,
        query: str,
        context: str,
    ) -> List[ToTThought]:
        """
        [ToT] Phase 1 — Generate NUM_BRANCHES independent thought branches.

        Each branch:
          • Receives the same query + context.
          • Uses a different grading persona / strategy.
          • Produces a GRADE, CONFIDENCE, and REFINED_CONTEXT.

        This deliberate diversity (strict / lenient / balanced) surfaces
        different aspects of the grading problem, maximising the chance
        that at least one branch produces an optimal result.
        """
        branches: List[ToTThought] = []

        for strat in BRANCH_STRATEGIES:
            bid = strat["id"]
            print(f"\n[GraderWriterAgent] [ToT] Generating {bid} ({strat['persona']})…")

            # ── [ToT] Call LLM with branch-specific strategy prompt ────────────
            branch_prompt = _BRANCH_PROMPT_TEMPLATE.format(
                persona=strat["persona"],
                strategy=strat["strategy"],
                query=query,
                context=context or "(empty — context retrieval returned nothing)",
            )

            # Direct LLM call (not via ReAct) for clean, parseable branch output
            branch_response = self._llm.complete(branch_prompt)
            branch_text = str(branch_response)

            # Parse the structured tokens from the branch output
            grade, confidence, refined = _parse_branch_output(branch_text)

            # Build a ToTThought record for this branch
            thought = ToTThought(
                branch_id=bid,
                reasoning=branch_text,
                score=confidence,  # preliminary score from self-assessment
                selected=False,
            )
            branches.append(thought)

            # Terminal output for this branch
            print(f"[GraderWriterAgent] [ToT] {bid} result:")
            print(f"[GraderWriterAgent]        Grade      : {grade.value.upper()}")
            print(f"[GraderWriterAgent]        Confidence : {confidence:.2f}")
            print(f"[GraderWriterAgent]        Refined ctx: {len(refined)} chars")

        return branches

    # ─────────────────────────────────────────────────────────────────────────
    # ToT Phase 2: Evaluate and score branches
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_tot_branches(
        self,
        query: str,
        original_context: str,
        branches: List[ToTThought],
    ) -> dict:
        """
        [ToT] Phase 2 — Score all branches with a dedicated evaluator LLM call.

        The evaluator receives all branch reasoning texts side by side and
        produces a JSON score card covering:
          • relevance_score    (0-40)
          • completeness_score (0-30)
          • fidelity_score     (0-30)
          • total              (0-100)

        Returns the parsed score dict including best_branch_id.
        """
        print(
            f"\n[GraderWriterAgent] [ToT] Phase 2: Evaluating {len(branches)} branches…"
        )

        # Format all branch reasoning texts for the evaluator
        branches_text = "\n\n".join(
            f"=== {b.branch_id} ===\n{b.reasoning[:1000]}" for b in branches
        )

        evaluator_prompt = _EVALUATOR_PROMPT_TEMPLATE.format(
            n=len(branches),
            query=query,
            original_context=original_context[:800],
            branches_text=branches_text,
        )

        # ── [ToT] Evaluator LLM call ───────────────────────────────────────────
        # This is a separate LLM inference pass acting as the "judge" in the
        # ToT framework.  It is NOT wrapped in ReAct (we want clean JSON output,
        # not a reasoning loop).
        print("[GraderWriterAgent] [ToT] Calling evaluator LLM for branch scoring…")
        eval_response = self._llm.complete(evaluator_prompt)
        scores = _parse_evaluator_scores(str(eval_response), len(branches))

        # Print score table to terminal
        print("[GraderWriterAgent] [ToT] Branch scores:")
        print(
            f"[GraderWriterAgent]       {'Branch':<12} {'Relevance':>10} {'Complete':>10} {'Fidelity':>10} {'Total':>8}"
        )
        print("[GraderWriterAgent]       " + "─" * 52)
        for entry in scores.get("scores", []):
            marker = (
                " ◀ WINNER"
                if entry["branch_id"] == scores.get("best_branch_id")
                else ""
            )
            print(
                f"[GraderWriterAgent]       {entry['branch_id']:<12}"
                f" {entry.get('relevance', 0):>10}"
                f" {entry.get('completeness', 0):>10}"
                f" {entry.get('fidelity', 0):>10}"
                f" {entry.get('total', 0):>8}{marker}"
            )
        print(f"[GraderWriterAgent] [ToT] Best branch: {scores.get('best_branch_id')}")
        print(
            f"[GraderWriterAgent] [ToT] Rationale  : {scores.get('rationale', 'N/A')}"
        )

        return scores

    # ─────────────────────────────────────────────────────────────────────────
    # ToT Phase 3: Select the winning branch
    # ─────────────────────────────────────────────────────────────────────────

    def _select_best_branch(
        self,
        branches: List[ToTThought],
        scores: dict,
    ) -> Tuple[ToTThought, int]:
        """
        [ToT] Phase 3 — Select the highest-scoring branch.

        Updates the selected=True flag on the winner and propagates the
        evaluator's total score back into each ToTThought for the state.

        Returns (winning_thought, winning_total_score).
        """
        best_id = scores.get("best_branch_id", "branch_1")
        score_map = {
            entry["branch_id"]: entry.get("total", 0)
            for entry in scores.get("scores", [])
        }

        best_thought: Optional[ToTThought] = None
        best_score: int = -1

        for thought in branches:
            # Update the thought's score with the evaluator's total
            thought.score = score_map.get(thought.branch_id, 0)
            if thought.branch_id == best_id:
                thought.selected = True
                best_thought = thought
                best_score = int(thought.score)

        # Fallback: if best_id not found, pick branch with highest score
        if best_thought is None:
            best_thought = max(branches, key=lambda b: b.score)
            best_thought.selected = True
            best_score = int(best_thought.score)

        print(
            f"\n[GraderWriterAgent] [ToT] Phase 3: Selected winner → {best_thought.branch_id}"
        )
        print(f"[GraderWriterAgent]            Total score : {best_score}/100")

        return best_thought, best_score

    # ─────────────────────────────────────────────────────────────────────────
    # Main run method
    # ─────────────────────────────────────────────────────────────────────────

    async def run(self, state: WorkflowState) -> WorkflowState:
        """
        Execute the ToT grading + ReAct MCP fact-checking pipeline.

        Full execution sequence
        -----------------------
        [ReAct] Pre-ToT: optionally fact-check context against live DBs via MCP.
        [ToT]   Phase 1: Generate NUM_BRANCHES independent grading branches.
        [ToT]   Phase 2: Evaluate all branches with a dedicated LLM judge call.
        [ToT]   Phase 3: Select the highest-scoring branch.
        [State] Write grade, refined_context, and ToT artifacts into WorkflowState.

        Consumes : state.query, state.retrieved_context
        Produces : state.grade, state.refined_context,
                   state.tot_thoughts, state.tot_best_branch,
                   state.react_trace (appended)
        """
        query = str(state.query or "").strip()
        retrieved_context = str(state.retrieved_context or "").strip()

        # Print immediately so an early return can never make the grader appear
        # as though it was not called.
        print("\n" + "═" * 70)
        print("[GraderWriterAgent] ▶  STARTING — Agent 3: Grader / Writer")
        print("[GraderWriterAgent]    Frameworks : ToT (PRIMARY) + ReAct + MCP")
        print(
            "[GraderWriterAgent]    RAG        : ✗  Receives context "
            "from RetrieverAgent"
        )
        print(
            f"[GraderWriterAgent]    ToT        : ✔  {NUM_BRANCHES} "
            "branches → evaluate → select best"
        )
        print("[GraderWriterAgent]    ReAct      : ✔  " "Pre-ToT MCP fact-checking")
        print(
            "[GraderWriterAgent]    MCP        : ✔  "
            "mysql_query, mysql_describe_table, mysql_list_tables"
        )
        print(f"[GraderWriterAgent]    Query      : {query}")
        print(f"[GraderWriterAgent]    Context len: " f"{len(retrieved_context)} chars")
        print("─" * 70)

        # A grader cannot produce grounded refined context when retrieval
        # produced nothing. Mark this deterministically as FAIL rather than
        # asking the LLM to grade an empty string.
        if not retrieved_context:
            state.grade = GradeResult.FAIL
            state.refined_context = ""
            state.tot_thoughts = []
            state.tot_best_branch = None
            state.react_trace.append(
                "[GraderWriterAgent][GUARD] "
                "FAIL: RetrieverAgent produced no context."
            )

            print("[GraderWriterAgent] ⚠️ No retrieved context was supplied.")
            print("[GraderWriterAgent]    Grade set deterministically to FAIL.")
            print("═" * 70 + "\n")
            return state

        # ── [ReAct + MCP] Pre-ToT: optional fact-checking against live DBs ────
        # The prior prompt did not include either the query or retrieved context,
        # so the verification agent did not know what claim it was checking.
        print("[GraderWriterAgent] [ReAct] " "Pre-ToT MCP fact-checking phase…")
        print(
            "[GraderWriterAgent] [MCP]   "
            "Agent may call MySQL tools to verify a necessary fact."
        )

        pre_tot_prompt = f"""Fact-check the retrieved evidence only when a
live database check is necessary before grading.

USER QUERY
----------
{query}

RETRIEVED CONTEXT
-----------------
{retrieved_context[:6000]}

RULES
-----
1. Do not answer the user's question.
2. Do not invent table names, column names, IDs, dates, or filter values.
3. If the retrieved context already contains sufficient exact database rows,
   do not call a tool.
4. If verification is necessary, call at most ONE database tool.
5. Use only table and column names explicitly present in the retrieved context.
6. After one tool result, stop.
7. End the response with exactly:
   PRE_TOT_CHECK_COMPLETE
"""

        fact_check_result = "NO_VERIFICATION_NEEDED"

        try:
            pre_tot_response = await self._react_agent.run(
                user_msg=pre_tot_prompt,
                max_iterations=3,
                early_stopping_method="generate",
            )

            response_candidate = getattr(
                pre_tot_response,
                "response",
                pre_tot_response,
            )
            fact_check_result = str(response_candidate).strip()

            if "PRE_TOT_CHECK_COMPLETE" in fact_check_result:
                print(
                    "[GraderWriterAgent] [ReAct+MCP] "
                    "Pre-ToT check completed successfully."
                )
            else:
                print(
                    "[GraderWriterAgent] [ReAct+MCP] "
                    "Pre-ToT check ended without completion marker; "
                    "continuing with retrieved evidence."
                )

        except Exception as exc:
            # Fact verification is optional. A failure here should not discard
            # the RetrieverAgent evidence or skip Tree of Thought grading.
            fact_check_result = f"PRE_TOT_CHECK_FAILED: {type(exc).__name__}: {exc}"
            print(
                "[GraderWriterAgent] ⚠️ Optional pre-ToT verification failed: "
                f"{type(exc).__name__}: {exc}"
            )
            print("[GraderWriterAgent]    Continuing with retrieved context.")

        print("[GraderWriterAgent] [ReAct+MCP] " "Pre-ToT check complete.")

        # Include a successful verification result as supplemental evidence,
        # but never replace the original RetrieverAgent context.
        grading_context = retrieved_context

        if (
            fact_check_result
            and fact_check_result != "NO_VERIFICATION_NEEDED"
            and not fact_check_result.startswith("PRE_TOT_CHECK_FAILED:")
        ):
            grading_context += "\n\n=== PRE-TOT VERIFICATION ===\n" + fact_check_result

        # ── [ToT] Phase 1: Generate all branches ──────────────────────────────
        print(
            "\n[GraderWriterAgent] [ToT] "
            "═══ PHASE 1: Generating thought branches ═══"
        )

        branches = self._generate_tot_branches(
            query=query,
            context=grading_context,
        )

        if not branches:
            state.grade = GradeResult.FAIL
            state.refined_context = ""
            state.tot_thoughts = []
            state.tot_best_branch = None
            state.react_trace.append(
                "[GraderWriterAgent][ToT] " "FAIL: No grading branches were generated."
            )
            print("[GraderWriterAgent] ✘ No ToT branches were generated.")
            print("═" * 70 + "\n")
            return state

        # ── [ToT] Phase 2: Evaluate branches ──────────────────────────────────
        print("\n[GraderWriterAgent] [ToT] " "═══ PHASE 2: Evaluating branches ═══")

        scores = self._evaluate_tot_branches(
            query=query,
            original_context=grading_context,
            branches=branches,
        )

        # ── [ToT] Phase 3: Select the winner ──────────────────────────────────
        print("\n[GraderWriterAgent] [ToT] " "═══ PHASE 3: Selecting best branch ═══")

        best_thought, best_score = self._select_best_branch(
            branches,
            scores,
        )

        # ── Parse grade and refined context from the winning branch ───────────
        grade, _, refined_context = _parse_branch_output(best_thought.reasoning)
        refined_context = str(refined_context or "").strip()

        # Deterministic grading invariant:
        # PASS is invalid unless it contains usable refined evidence.
        if grade == GradeResult.PASS and not refined_context:
            print(
                "[GraderWriterAgent] ⚠️ Winning branch returned PASS "
                "with empty refined context; overriding to FAIL."
            )
            grade = GradeResult.FAIL

        # Prevent a lenient branch from treating an error-only context as a PASS.
        error_markers = (
            "database error:",
            "mcpservererror",
            "unknown column",
            "doesn't exist",
            "sql parse error",
            "no usable evidence",
        )
        lower_refined = refined_context.lower()

        if (
            grade == GradeResult.PASS
            and any(marker in lower_refined for marker in error_markers)
            and len(refined_context) < 300
        ):
            print(
                "[GraderWriterAgent] ⚠️ Refined context consists primarily "
                "of retrieval errors; overriding PASS to FAIL."
            )
            grade = GradeResult.FAIL
            refined_context = ""

        if grade == GradeResult.FAIL:
            refined_context = ""

        # ── Write all results into shared state ───────────────────────────────
        state.tot_thoughts = branches
        state.tot_best_branch = best_thought.branch_id
        state.grade = grade
        state.refined_context = refined_context

        # Append a comprehensive trace entry.
        state.react_trace.append(
            "[GraderWriterAgent][ToT+ReAct]\n"
            f"  Pre-ToT MCP check : {fact_check_result[:200]}\n"
            f"  Branches generated: {[b.branch_id for b in branches]}\n"
            f"  Best branch       : "
            f"{best_thought.branch_id} (score={best_score}/100)\n"
            f"  Grade             : {grade.value}\n"
            f"  Refined ctx chars : {len(refined_context)}"
        )

        # ── Terminal summary ──────────────────────────────────────────────────
        print("─" * 70)
        print("[GraderWriterAgent] ✔  COMPLETE")
        print(
            "[GraderWriterAgent]    ToT winner     : "
            f"{best_thought.branch_id} ({best_score}/100)"
        )
        print("[GraderWriterAgent]    Final grade    : " f"{grade.value.upper()}")
        print(
            "[GraderWriterAgent]    Refined ctx    : " f"{len(refined_context)} chars"
        )
        print(
            "[GraderWriterAgent]    Next agent     : "
            "AnswerGeneratorAgent (ReAct + MCP)"
        )
        print("═" * 70 + "\n")

        return state
