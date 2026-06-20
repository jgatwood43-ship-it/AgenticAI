"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  agents/llm_decision_agent.py                                               ║
║  Agent 1 – LLM Decision Step                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE
───────
Inspect the incoming user query and emit a ROUTING decision that determines
which branch of the workflow executes next:

    • Route.RETRIEVER        → send the query through the full RAG pipeline
                               (RetrieverAgent → GraderWriterAgent → AnswerGeneratorAgent)
    • Route.ANSWER_GENERATOR → skip retrieval and answer directly from the LLM's
                               parametric knowledge

FRAMEWORKS USED IN THIS AGENT
──────────────────────────────
┌──────────┬──────────────────────────────────────────────────────────────────┐
│ ReAct    │ The agent runs a Reason → Act → Observe loop (LlamaIndex          │
│          │ ReActAgent).  It can call MCP tools to inspect database schemas   │
│          │ before making its routing decision, then emits a ROUTE token.     │
├──────────┼──────────────────────────────────────────────────────────────────┤
│ MCP      │ Two FunctionTools (mysql_query, postgres_query) give the agent    │
│          │ optional read access to the live databases via the Database-MCP   │
│          │ server (npx database-mcp).  Useful for checking whether a table   │
│          │ relevant to the query exists before committing to a route.        │
├──────────┼──────────────────────────────────────────────────────────────────┤
│ ToT      │ NOT USED in this agent.  Routing is a binary decision that does   │
│          │ not benefit from multi-branch exploration; ReAct is sufficient.   │
├──────────┼──────────────────────────────────────────────────────────────────┤
│ RAG      │ NOT USED in this agent.  This agent DECIDES whether to use RAG;  │
│          │ it does not perform retrieval itself.                              │
└──────────┴──────────────────────────────────────────────────────────────────┘

REACT LOOP DETAIL
─────────────────
  Thought  : "The question asks about X. Does the vector DB likely have relevant
              documents?  Let me check the Postgres schema."
  Action   : postgres_query("SELECT table_name FROM information_schema.tables …")
  Observe  : [table list returned]
  Thought  : "Table 'document_embeddings' exists and the query is domain-specific.
              I should route to retriever."
  Answer   : ROUTE: retriever

TERMINAL OUTPUT
───────────────
  Every ReAct Thought / Action / Observation is printed as it happens (verbose=True).
  A summary line is printed at the end showing the chosen route.
"""
from __future__ import annotations

import re
import asyncio

# from llama_index.core.agent import ReActAgent
# from llama_index.core.agent.react.base import ReActAgent
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.llms import LLM

from core.state import Route, WorkflowState
from tools.mcp_tools import ALL_MCP_TOOLS

# FIX: Import with a safe, descriptive alias
from config.settings import settings as db_config

# ─────────────────────────────────────────────────────────────────────────────
# System prompt injected into the ReAct agent at construction time.
# The prompt constrains the LLM to output exactly one routing token after
# its reasoning so the parser below has a reliable signal to extract.
# ─────────────────────────────────────────────────────────────────────────────
_DECISION_SYSTEM_PROMPT = """\
You are a ROUTING agent in a multi-agent RAG pipeline.

Your ONLY job is to decide how to handle the user's question:

  "retriever"        → the question needs context from documents stored in the
                       vector database.  Use this route for specific, factual,
                       or domain-specific questions that require grounded evidence.
  "answer_generator" → the question is general enough to be answered from your
                       own parametric knowledge without document retrieval.

You have access to database tools (mysql_query, postgres_query).
Use them ONLY if you need to confirm whether relevant data exists in a schema.

──────────────────────────────────────────────────────────────────────────────
REACT INSTRUCTIONS
──────────────────
1. Think step by step about the nature of the query.
2. Optionally call a database tool to inspect schemas.
3. Reason about whether the vector store is likely to contain relevant chunks.
4. Output EXACTLY ONE of these two lines (nothing after it):

     ROUTE: retriever
     ROUTE: answer_generator
──────────────────────────────────────────────────────────────────────────────
CRITICAL TOOL FAILURE RULES:
If you call a database tool and it returns an error, connection failure, 
or tells you that a table/schema cannot be found, DO NOT try again. 
Immediately output your final ROUTE decision line. Fall back to "retriever" 
if you think document chunks might help, or "answer_generator" if it is a general question.
"""


def _parse_route(text: str) -> Route:
    """
    Extract the routing decision token from the ReAct agent's final response.

    Searches for the pattern  ROUTE: <token>  (case-insensitive).
    Falls back to Route.RETRIEVER if no clear token is found — safer to
    over-retrieve than to miss domain-specific information.
    """
    match = re.search(r"ROUTE:\s*(retriever|answer_generator)", text, re.IGNORECASE)
    if match:
        token = match.group(1).lower()
        return Route.RETRIEVER if token == "retriever" else Route.ANSWER_GENERATOR
    # Conservative fallback: prefer retrieval over direct generation
    return Route.RETRIEVER


# ─────────────────────────────────────────────────────────────────────────────
# Agent class
# ─────────────────────────────────────────────────────────────────────────────

class LLMDecisionAgent:
    """
    Agent 1 – LLM Decision Step.

    Wraps a LlamaIndex ReActAgent configured with MCP database tools.
    Produces a single routing decision (Route enum) stored in WorkflowState.
    """

    def __init__(self, llm: LLM) -> None:
        print("\n[LLMDecisionAgent] ⚙  Initialising ReActAgent (ReAct framework)…")
        print("[LLMDecisionAgent]    Tools available: mysql_query (MCP), postgres_query (MCP)")

        self._agent = ReActAgent(
            tools=ALL_MCP_TOOLS,
            llm=llm,
            max_iterations=3,
            verbose=True,
            system_prompt=_DECISION_SYSTEM_PROMPT,
        )

    # ─────────────────────────────────────────────────────────────────────────

    async def run(self, state: WorkflowState) -> WorkflowState:
        """
        Execute the routing decision for the incoming query.
        """
        print("\n" + "═" * 70)
        print("[LLMDecisionAgent] ▶  STARTING — Agent 1: LLM Decision Step")
        print(f"[LLMDecisionAgent]    Framework : ReAct")
        print(f"[LLMDecisionAgent]    Tools     : MCP (mysql_query, postgres_query)")
        print(f"[LLMDecisionAgent]    ToT       : ✗  Not applicable for binary routing")
        print(f"[LLMDecisionAgent]    RAG       : ✗  Decides whether RAG is needed; does not retrieve")
        print(f"[LLMDecisionAgent]    Query     : {state.query}")
        print("─" * 70)
        print("[LLMDecisionAgent] [ReAct] Starting Thought → Action → Observe loop…")

        prompt = (
            f"User question: {state.query}\n\n"
            "Decide whether to route to 'retriever' or 'answer_generator'.\n"
            "Think step by step, optionally query a database schema, then output "
            "your ROUTE decision."
        )

        print("[LLMDecisionAgent] [MCP] MCP tools armed: mysql_query, postgres_query")
        print("[LLMDecisionAgent]       (agent will call these if it needs schema info)")

        # Initialize your fallback tracking text
        response_text = ""

        # 1. Wrap the workflow run inside a try/except block
        try:
            response = await self._agent.run(user_msg=prompt, max_iterations=3)
            
            # Extract text safely if it finished cleanly
            if hasattr(response, "response"):
                response_text = str(response.response)
            elif hasattr(response, "message"):
                response_text = str(response.message)
            else:
                response_text = str(response)
                
        except Exception as exc:
            # 2. Catch the max iterations ceiling exception gracefully
            print(f"\n[LLMDecisionAgent] ⚠️ Max iterations or error hit: {exc}")
            print("[LLMDecisionAgent]    Triggering safe fallback strategy...")
            # Leave response_text empty so the parser triggers the default route

        # 3. Parse whatever text remains (will correctly return Route.RETRIEVER if empty)
        route = _parse_route(response_text)
        state.route = route
        
        if state.react_trace is None:
            state.react_trace = []
            
        state.react_trace.append(
            f"[LLMDecisionAgent][ReAct] route={route.value}\n{response_text or 'Turn limit reached. Fallback triggered.'}"
        )   

        return state