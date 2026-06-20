"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  agents/retriever_agent.py                                                  ║
║  Agent 2 – Retriever Node                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE
───────
Perform semantic RAG retrieval from the Postgres pgvector database, surfacing
the most relevant document chunks for the user's query.

The agent also has access to the MySQL and Postgres MCP tools so it can
pull supplementary structured data (e.g. lookup tables, metadata) that
complements the semantic search results.

FRAMEWORKS USED IN THIS AGENT
──────────────────────────────
┌──────────┬──────────────────────────────────────────────────────────────────┐
│ RAG      │ CORE responsibility of this agent.  The LlamaIndex               │
│          │ VectorStoreIndex (backed by pgvector + bge-large-en-v1.5) is      │
│          │ wrapped as a QueryEngineTool.  The agent calls it to perform      │
│          │ Approximate Nearest-Neighbour (ANN) search over document          │
│          │ embeddings and return the top-k most semantically similar chunks. │
├──────────┼──────────────────────────────────────────────────────────────────┤
│ ReAct    │ The agent wraps RAG retrieval in a ReAct loop, allowing it to:    │
│          │   • Decompose complex queries into sub-queries.                   │
│          │   • Iteratively refine the search if the first pass is weak.      │
│          │   • Combine vector results with structured DB lookups (via MCP).  │
├──────────┼──────────────────────────────────────────────────────────────────┤
│ MCP      │ mysql_query and postgres_query tools are available for pulling    │
│          │ structured metadata, foreign-key lookups, or cross-referencing    │
│          │ relational data alongside the vector search results.              │
├──────────┼──────────────────────────────────────────────────────────────────┤
│ ToT      │ NOT USED in this agent.  Retrieval is a deterministic ANN         │
│          │ search; branching is better applied at the evaluation step        │
│          │ (GraderWriterAgent).                                              │
└──────────┴──────────────────────────────────────────────────────────────────┘

RAG ARCHITECTURE DETAIL
───────────────────────
  Embedding model : BAAI/bge-large-en-v1.5 (1024-dim, Apple MPS-accelerated)
  Chunking        : SemanticSplitterNodeParser (applied at ingest time)
  Vector store    : Postgres pgvector @ 192.168.86.48:5432
  ANN index       : HNSW (m=16, ef_construction=64, ef_search=40)
  Top-k           : 5 nodes per query (configurable)

REACT LOOP DETAIL
─────────────────
  Thought  : "The query asks about X.  I should search for 'X concept' and also
              'related Y' to get broad coverage."
  Action   : vector_search("X concept")          ← RAG call into pgvector
  Observe  : [top-5 document chunks returned]
  Thought  : "Chunks look relevant.  Let me also check the MySQL DB for metadata."
  Action   : mysql_query("SELECT …")             ← MCP call
  Observe  : [structured rows returned]
  Answer   : [combined context returned to orchestrator]

TERMINAL OUTPUT
───────────────
  Prints a header banner, RAG search announcement, each MCP tool call,
  the number of nodes retrieved, and a footer summary.
"""
from __future__ import annotations

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core import VectorStoreIndex
# from llama_index.core.agent import ReActAgent
from llama_index.core.llms import LLM
from llama_index.core.tools import QueryEngineTool

from core.state import WorkflowState
from tools.mcp_tools import ALL_MCP_TOOLS
from config.settings import settings as db_config

import asyncio

# ─────────────────────────────────────────────────────────────────────────────
# System prompt for the retrieval ReAct agent.
# Instructs the LLM NOT to answer the question — only to retrieve context.
# Distinguishes this agent's role clearly from AnswerGeneratorAgent.
# ─────────────────────────────────────────────────────────────────────────────
_RETRIEVER_SYSTEM_PROMPT = """\
You are a RETRIEVAL SPECIALIST agent in a multi-agent RAG pipeline.

Your SOLE purpose is to find the most relevant document chunks from the vector
database and, optionally, supplementary structured data from the relational
databases.  You must NOT answer the user's question — that is another agent's job.

Available tools
───────────────
  vector_search   : Semantic ANN search over the pgvector document store.
                    Use this as your PRIMARY tool.
  mysql_query     : SQL query against the MySQL database (MCP).
                    Use for structured metadata or relational lookups.
  postgres_query  : SQL query against Postgres (MCP, non-vector tables).
                    Use for cross-referencing data alongside vector results.

REACT INSTRUCTIONS
──────────────────
1. Analyse the question — identify 1-3 distinct search angles / sub-queries.
2. Call vector_search for each angle to maximise recall.
3. If a relational lookup would add useful structured facts, call mysql_query
   or postgres_query.
4. Aggregate ALL retrieved passages and return them verbatim.
5. Do NOT summarise, filter, or answer the question.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper: wrap the VectorStoreIndex as a LlamaIndex QueryEngineTool
# ─────────────────────────────────────────────────────────────────────────────

def _build_vector_search_tool(
    index: VectorStoreIndex,
    similarity_top_k: int = 5,
) -> QueryEngineTool:
    """
    [RAG] Wrap the LlamaIndex VectorStoreIndex as a QueryEngineTool.

    The QueryEngine performs:
      1. Embed the query text using bge-large-en-v1.5.
      2. ANN search against pgvector (HNSW index).
      3. Return the top-k most similar document chunks with scores.

    This tool is then handed to the ReActAgent so it can call
    'vector_search' as an Action inside its reasoning loop.
    """
    query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)

    return QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="vector_search",
        description=(
            "[RAG] Semantic similarity search over the pgvector document store. "
            "Embeds the query with bge-large-en-v1.5 and retrieves the top-5 "
            "most relevant document chunks via HNSW ANN search. "
            "Input: a natural-language search query string. "
            "Returns: relevant document passages with similarity scores."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent class
# ─────────────────────────────────────────────────────────────────────────────

class RetrieverAgent:
    """
    Agent 2 – Retriever Node.

    Uses ReAct to orchestrate one or more RAG (pgvector) searches and optional
    MCP-backed structured database lookups.  Returns all retrieved context
    into the shared WorkflowState for the GraderWriterAgent to evaluate.

    Frameworks
    ----------
    RAG   : Primary retrieval via LlamaIndex VectorStoreIndex + pgvector.
    ReAct : Reasoning loop enabling multi-angle querying and tool composition.
    MCP   : Supplementary MySQL / Postgres access via Database-MCP server.
    """

    def __init__(self, llm: LLM, index: VectorStoreIndex) -> None:
        # ── [RAG] Build the vector search tool from the pgvector index ─────────
        print("\n[RetrieverAgent] ⚙  Building RAG vector_search tool from pgvector index…")
        print(f"[RetrieverAgent]    Embedding model : {db_config.embed_model_name}")
        print(f"[RetrieverAgent]    Vector DB       : {db_config.pg_host}:{db_config.pg_port}/{db_config.pg_database}")
        print(f"[RetrieverAgent]    Table           : {db_config.pg_table}")
        print(f"[RetrieverAgent]    ANN index       : HNSW (similarity_top_k=5)")

        vector_tool = _build_vector_search_tool(index, similarity_top_k=5)

        # Keep a direct reference to the index so we can also pull raw
        # NodeWithScore objects (with similarity scores) into the state.
        self._index = index

        # ── [ReAct] Build the ReActAgent with RAG + MCP tools ─────────────────
        # The tool list order matters: vector_search is listed first to signal
        # its primacy; MCP tools follow as supplementary options.
        print("[RetrieverAgent] ⚙  Initialising ReActAgent (ReAct + RAG + MCP)…")
        print("[RetrieverAgent]    Tools: vector_search (RAG), mysql_query (MCP), postgres_query (MCP)")

        self._agent = ReActAgent(
    tools=ALL_MCP_TOOLS,
    llm=llm,
    max_iterations=db_config.react_max_iterations,
    verbose=True,
    # system_prompt=_DECISION_SYSTEM_PROMPT,
    system_prompt=_RETRIEVER_SYSTEM_PROMPT,
        )
        # self._agent = ReActAgent.from_tools(
           # tools=[vector_tool, *ALL_MCP_TOOLS],   # RAG tool first, then MCP tools
           # llm=llm,
           # max_iterations=settings.react_max_iterations,
           # verbose=True,                           # prints Thought/Action/Observe
           # system_prompt=_RETRIEVER_SYSTEM_PROMPT,
        #)

    # ─────────────────────────────────────────────────────────────────────────

    async def run(self, state: WorkflowState) -> WorkflowState:
        """
        Execute RAG retrieval for the incoming query via the ReAct loop.

        ReAct + RAG flow
        ----------------
        1. LLM Thought: decompose the query into search angles.
        2. LLM Action : vector_search(<angle>)  → pgvector ANN search [RAG]
        3. LLM Observe: top-k document chunks returned.
        4. LLM Thought: assess coverage; decide if another angle or MCP call needed.
        5. LLM Action : mysql_query / postgres_query (optional) [MCP]
        6. LLM Observe: structured data returned.
        7. Steps 2-6 repeat up to max_iterations.
        8. LLM emits aggregated context as its final answer.

        Additionally, raw NodeWithScore objects are pulled directly from the
        index (outside the ReAct loop) and stored in state.retrieved_nodes
        for downstream score-based filtering.

        Consumes : state.query
        Produces : state.retrieved_nodes, state.retrieved_context,
                   state.react_trace (appended)
        """
        print("\n" + "═" * 70)
        print("[RetrieverAgent] ▶  STARTING — Agent 2: Retriever Node")
        print(f"[RetrieverAgent]    Frameworks : ReAct + RAG + MCP")
        print(f"[RetrieverAgent]    ToT        : ✗  Not used in retrieval")
        print(f"[RetrieverAgent]    Query      : {state.query}")
        print("─" * 70)

        # ── [ReAct] Start the Thought → Action → Observe loop ─────────────────
        # The agent will print its reasoning to the terminal in real time.
        # Each call to vector_search is a RAG retrieval; each call to
        # mysql_query / postgres_query is an MCP call.
        print("[RetrieverAgent] [ReAct] Starting Thought → Action → Observe loop…")
        print("[RetrieverAgent] [RAG]   Primary tool: vector_search (pgvector ANN)")
        print("[RetrieverAgent] [MCP]   Supplementary: mysql_query, postgres_query")

        prompt = (
            f"Retrieve all relevant context for the following question:\n\n"
            f"QUERY: {state.query}\n\n"
            "Instructions:\n"
            "1. Identify 1-3 distinct search angles for this query.\n"
            "2. Call vector_search for each angle to maximise document recall.\n"
            "3. If useful structured data might exist in a relational DB, call "
            "mysql_query or postgres_query.\n"
            "4. Return ALL retrieved passages verbatim — do NOT filter or answer."
        )

        # ── [RAG + ReAct] Execute retrieval ───────────────────────────────────
        # vector_search calls inside agent.chat() trigger:
        #   embed(query) → pgvector ANN search → top-k chunks returned
        # MCP calls inside agent.chat() trigger:
        #   npx database-mcp → SQL executed → rows returned as JSON
        # response = self._agent.chat(prompt)
        response = await self._agent.run(user_msg=prompt)
        response_text = str(response)

        # ── [RAG] Pull raw NodeWithScore objects directly from the index ───────
        # This gives us similarity scores and raw node metadata that the
        # GraderWriterAgent can use for relevance filtering.
        print("[RetrieverAgent] [RAG] Pulling raw NodeWithScore objects from index…")
        retriever = self._index.as_retriever(similarity_top_k=5)
        state.retrieved_nodes = retriever.retrieve(state.query)

        # Log similarity scores for each node to the terminal
        print(f"[RetrieverAgent] [RAG] Retrieved {len(state.retrieved_nodes)} nodes:")
        for i, node in enumerate(state.retrieved_nodes):
            score = getattr(node, "score", "N/A")
            snippet = node.node.get_content()[:80].replace("\n", " ")
            print(f"[RetrieverAgent]       Node {i+1}: score={score:.4f}  text='{snippet}…'")

        # Store the full aggregated context text for the grader
        state.retrieved_context = response_text

        # Append a condensed trace entry
        state.react_trace.append(
            f"[RetrieverAgent][ReAct+RAG] {len(state.retrieved_nodes)} nodes retrieved\n"
            + response_text[:500]
        )

        # ── Terminal summary ──────────────────────────────────────────────────
        print("─" * 70)
        print(f"[RetrieverAgent] ✔  COMPLETE")
        print(f"[RetrieverAgent]    Nodes retrieved  : {len(state.retrieved_nodes)}")
        print(f"[RetrieverAgent]    Context length   : {len(state.retrieved_context)} chars")
        print(f"[RetrieverAgent]    Next agent       : GraderWriterAgent (ReAct + ToT + MCP)")
        print("═" * 70 + "\n")

        return state
