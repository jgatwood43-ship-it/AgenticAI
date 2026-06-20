"""
core/state.py
─────────────
Shared workflow state passed between every agent node.
Typed with Pydantic for safety and easy serialisation.

Extended to carry Tree of Thought (ToT) artifacts produced by GraderWriterAgent.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class Route(str, Enum):
    """Routing decisions produced by the LLM Decision Step agent."""
    RETRIEVER        = "retriever"
    ANSWER_GENERATOR = "answer_generator"


class GradeResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"


# ─────────────────────────────────────────────────────────────────────────────
# Tree of Thought thought node
# ─────────────────────────────────────────────────────────────────────────────

class ToTThought(BaseModel):
    """
    Represents a single 'thought' (candidate reasoning branch) in a
    Tree of Thought exploration.

    Fields
    ------
    branch_id   : Unique identifier, e.g. "branch_1".
    reasoning   : The full reasoning text produced for this branch.
    score       : Evaluation score (0.0 – 1.0) assigned by the ToT evaluator.
    selected    : True if this branch was chosen as the best path forward.
    """
    branch_id : str   = Field(..., description="Unique branch label.")
    reasoning : str   = Field(..., description="Reasoning text for this thought branch.")
    score     : float = Field(0.0,  description="Evaluation score 0.0–1.0.")
    selected  : bool  = Field(False, description="Was this branch selected as best?")


# ─────────────────────────────────────────────────────────────────────────────
# Main shared state
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowState(BaseModel):
    """
    Centralised state object threaded through every agent in the pipeline.

    Lifecycle
    ---------
        query ──▶ LLMDecisionAgent (ReAct)
                     │
                     ├─(retriever)──▶ RetrieverAgent (ReAct + RAG/pgvector + MCP)
                     │                    │
                     │                    ▼
                     │               GraderWriterAgent (ReAct + ToT + MCP)
                     │                    │
                     └─(direct)───────────┤
                                          ▼
                                  AnswerGeneratorAgent (ReAct + MCP)

    ToT is applied inside GraderWriterAgent: it generates N candidate
    grade/rewrite branches, scores each, and picks the best one before
    writing to refined_context.
    """

    # ── Input ────────────────────────────────────────────────────────────────
    query: str = Field(..., description="The original user question.")

    # ── Routing ──────────────────────────────────────────────────────────────
    route: Optional[Route] = Field(
        None, description="Routing decision made by LLMDecisionAgent."
    )

    # ── Retrieval (RAG) ───────────────────────────────────────────────────────
    retrieved_nodes: List[Any] = Field(
        default_factory=list,
        description="Raw LlamaIndex NodeWithScore objects from pgvector retrieval.",
    )
    retrieved_context: str = Field(
        default="",
        description="Flattened text of retrieved nodes passed to GraderWriterAgent.",
    )

    # ── Tree of Thought artifacts (produced by GraderWriterAgent) ─────────────
    tot_thoughts: List[ToTThought] = Field(
        default_factory=list,
        description=(
            "All candidate thought branches explored by the ToT process. "
            "Each branch represents a different grading / rewriting strategy."
        ),
    )
    tot_best_branch: Optional[str] = Field(
        None, description="branch_id of the ToT branch selected as best."
    )

    # ── Grading / Writing ────────────────────────────────────────────────────
    grade: Optional[GradeResult] = Field(
        None, description="Quality grade assigned by GraderWriterAgent."
    )
    refined_context: str = Field(
        default="",
        description="Rewritten / filtered context produced by the winning ToT branch.",
    )

    # ── Final Answer ──────────────────────────────────────────────────────────
    answer: str = Field(
        default="", description="Final answer produced by AnswerGeneratorAgent."
    )

    # ── ReAct trace ───────────────────────────────────────────────────────────
    react_trace: List[str] = Field(
        default_factory=list,
        description="Step-by-step ReAct Thought/Action/Observation trace.",
    )

    # ── Error ─────────────────────────────────────────────────────────────────
    error: Optional[str] = Field(
        None, description="Propagated error message if any agent fails."
    )

    class Config:
        arbitrary_types_allowed = True
