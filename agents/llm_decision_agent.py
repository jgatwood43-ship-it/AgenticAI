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
│ MCP      │ FunctionTools (mysql_query, mysql_list_tables, mysql_describe_table |
|.         | give the agent                        │
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
              documents?  Let me check the schema."
  Action   : postgres_query ("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;)
  Observe  : [table list returned]
  Thought  : "Table 'data_document_embeddings' exists and the query is domain-specific.
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
from tools.mcp_tools import MYSQL_TOOLS

# FIX: Import with a safe, descriptive alias
from config.settings import settings as db_config

from core.guardrails import (
    validate_user_query,
    validate_route_output,
    record_guardrail_event,
    record_runtime_event,
    start_timer,
    elapsed_ms,
)

# ─────────────────────────────────────────────────────────────────────────────
# System prompt injected into the ReAct agent at construction time.
# The prompt constrains the LLM to output exactly one routing token after
# its reasoning so the parser below has a reliable signal to extract.
# ─────────────────────────────────────────────────────────────────────────────
_DECISION_SYSTEM_PROMPT = """\
You are a USER ACCESS CYBERSECURITY ROUTING agent in a multi-agent RAG pipeline.

Your ONLY job is to decide how to handle the user's question:

  "retriever"        → the question needs context from documents stored in the
                       Postgres vector database.  Use this route for security rules, 
                       or domain-related questions that require grounded evidence.
                       the questions need deterministic structured data from the MySQL database, you may call the
                       MySQL tools to inspect the schema and determine if a relevant table exists.
  "answer_generator" → the question is general enough to be answered from your
                       own parametric knowledge without document retrieval.

You have access to MySQL tools only: mysql_query, mysql_list_tables, and mysql_describe_table.
Use these to find relationships between tables only if the user is asking for deterministic structured data.
Do not query Postgres directly. Postgres is reserved for pgvector document retrieval by RetrieverAgent.

"STRUCTURED DATA RULES:\n"
"- For employee, title, department, role, or access-list questions, use MySQL tools.\n"
"- Never guess column names.\n"
"- Before writing a SELECT query, first call mysql_list_tables if table names are unknown.\n"
"- Then call mysql_describe_table for the most relevant table.\n"
"- If a table contains an ID field such as job_title_id, role_id, department_id, or user_id, look for a related lookup table before answering.\n"
"- Do not stop after finding an ID field. Resolve the ID to the human-readable name when possible.\n"
"- If a query fails because of an unknown column, do not repeat the same query. Inspect the schema and correct the query.\n\n"

──────────────────────────────────────────────────────────────────────────────
REACT INSTRUCTIONS
──────────────────
1. Think step by step about the nature of the query.
2. Optionally call a database tool to inspect schemas.
3. Reason about whether the vector store is likely to contain relevant chunks.
4. Output EXACTLY ONE of these two lines (nothing after it):

     ROUTE: retriever
     ROUTE: answer_generator
     After any tool use, your final response MUST consist of exactly one line:

──────────────────────────────────────────────────────────────────────────────
CRITICAL TOOL FAILURE RULES:
If you call a database tool and it returns an error, connection failure, 
or tells you that a table/schema cannot be found, DO NOT try again. 
Immediately output your final ROUTE decision line. Fall back to "retriever" 
if you think document chunks might help, or "answer_generator" if it is a general question.

You must return exactly one route:
ROUTE: RETRIEVER
or
ROUTE: ANSWER_GENERATOR

Do not call tools or the RETRIEVER agent for greetings, identity questions, purpose questions, or general conversation.
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
    # Conservative fallback: prefer ANSWER_GENERATOR to catch converstional items that don't need retrieval, but this can be tuned based on your use case
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
        print(
            "[LLMDecisionAgent]    Tools available: mysql_query (MCP), mysq_list_tables (MCP), mysql_describe_table (MCP)"
        )

        self._agent = ReActAgent(
            tools=MYSQL_TOOLS,
            llm=llm,
            max_iterations=5,
            verbose=True,
            system_prompt=_DECISION_SYSTEM_PROMPT,
        )

    # ─────────────────────────────────────────────────────────────────────────

    async def run(self, state: WorkflowState) -> WorkflowState:
        """
        Execute the routing decision for the incoming query.
        """
        timer = start_timer()

        query_check = validate_user_query(state.query)
        record_guardrail_event(
            state,
            "LLMDecisionAgent",
            "validate_user_query",
            query_check,
        )

        if not query_check.allowed:
            state.route = Route.ANSWER_GENERATOR
            state.error = query_check.reason

            if state.react_trace is None:
                state.react_trace = []

            state.react_trace.append(
                f"[LLMDecisionAgent][GUARD] Query blocked: {query_check.reason}"
            )

            record_runtime_event(
                state,
                "LLMDecisionAgent",
                "blocked_query",
                {"reason": query_check.reason},
            )

            return state

        state.query = query_check.sanitized_value
        record_runtime_event(
            state,
            "LLMDecisionAgent",
            "started",
            {"query": state.query},
        )

        print("\n" + "═" * 70)
        print("[LLMDecisionAgent] ▶  STARTING — Agent 1: LLM Decision Step")
        print(f"[LLMDecisionAgent]    Framework : ReAct")
        print(
            f"[LLMDecisionAgent]    Tools     : MCP (mysql_query, mysql_list_tables, mysql_describe_table)"
        )
        print(f"[LLMDecisionAgent]    ToT       : ✗  Not applicable for binary routing")
        print(
            f"[LLMDecisionAgent]    RAG       : ✗  Decides whether RAG is needed; does not retrieve"
        )
        print(f"[LLMDecisionAgent]    Query     : {state.query}")
        print("─" * 70)
        print("[LLMDecisionAgent] [ReAct] Starting Thought → Action → Observe loop…")

        prompt = (
            f"User question: {state.query}\n\n"
            "Decide whether to route to 'retriever' or 'answer_generator'.\n"
            "Think step by step, optionally query a database schema, then output "
            "your ROUTE decision."
        )

        print(
            "[LLMDecisionAgent] [MCP] MCP tools armed: mysql_query, mysql_list_tables, mysql_describe_table"
        )
        print(
            "[LLMDecisionAgent]       (agent will call these if it needs schema info)"
        )

        # 1. Wrap the workflow run inside a try/except block
        # Initialize your fallback tracking text
        response_text = ""

        try:
            response = await self._agent.run(
                user_msg=prompt,
                max_iterations=5,
                early_stopping_method="generate",
            )

            if hasattr(response, "response"):
                response_text = str(response.response)
            elif hasattr(response, "message"):
                response_text = str(response.message)
            else:
                response_text = str(response)

        except Exception as exc:
            print(f"\n[LLMDecisionAgent] ⚠️ Max iterations or error hit: {exc}")
            print("[LLMDecisionAgent]    Triggering safe fallback strategy...")
            response_text = ""

        # Always validate route output, whether ReAct succeeded or failed
        route_check = validate_route_output(response_text)
        record_guardrail_event(
            state,
            "LLMDecisionAgent",
            "validate_route_output",
            route_check,
            elapsed_ms(timer),
        )

        safe_route_text = (
            route_check.sanitized_value
            if route_check.allowed
            else route_check.fallback_value
        )

        route = _parse_route(safe_route_text)
        state.route = route

        if state.react_trace is None:
            state.react_trace = []

        state.react_trace.append(
            f"[LLMDecisionAgent][ReAct] route={route.value}\n"
            f"{response_text or 'Turn limit reached. Fallback triggered.'}"
        )

        record_runtime_event(
            state,
            "LLMDecisionAgent",
            "completed",
            {
                "route": route.value,
                "elapsed_ms": elapsed_ms(timer),
            },
        )

        return state
