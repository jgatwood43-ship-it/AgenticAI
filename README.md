# Multi-Agent RAG Workflow

A four-agent LLM pipeline built with **LlamaIndex**, **Ollama (llama3.1)**,
**ReAct**, **Tree of Thought (ToT)**, and **pgvector RAG**, orchestrated via
a shared Pydantic state object.

---

## Project Location

```
/Users/jeffgatwood/multi_agent_workflow/
```

---

## Architecture

```
User Query
    │
    ▼
┌───────────────────────────────────────────┐
│  Agent 1: LLMDecisionAgent                │
│  Frameworks: ReAct + MCP                  │
│  → Inspects query; optionally peeks at    │
│    DB schemas via MCP; emits ROUTE token  │
└─────────────────────┬─────────────────────┘
                      │
          ┌───────────┴────────────┐
          │                        │
    route=retriever         route=answer_generator
          │                        │
          ▼                        │
┌────────────────────────────────────┐
│  Agent 2: RetrieverAgent           │
│  Frameworks: RAG + ReAct + MCP     │
│  → pgvector ANN search             │
│    (bge-large-en-v1.5, HNSW)       │
│  → Optional MCP DB lookups         │
└──────────────────┬─────────────────┘
                   │
                   ▼
┌───────────────────────────────────────────┐
│  Agent 3: GraderWriterAgent               │
│  Frameworks: ToT (PRIMARY) + ReAct + MCP  │
│  → Pre-ToT MCP fact-checking (ReAct)      │
│  → ToT Phase 1: Generate 3 branches       │
│      branch_1 : strict grader             │
│      branch_2 : lenient grader            │
│      branch_3 : balanced grader           │
│  → ToT Phase 2: LLM evaluator scores all  │
│  → ToT Phase 3: Select highest scorer     │
└──────────────────┬────────────────────────┘
                   │
                   └──────────────┐
                                  ▼
                ┌──────────────────────────────────────┐
                │  Agent 4: AnswerGeneratorAgent        │
                │  Frameworks: ReAct + MCP              │
                │  → RAG mode  : cites refined_context  │
                │  → Direct mode: parametric knowledge  │
                │  → Optional MCP DB enrichment         │
                └──────────────────────────────────────┘
```

---

## Framework Matrix

| Agent | ReAct | RAG | MCP | ToT |
|---|---|---|---|---|
| 1 LLMDecisionAgent | ✔ primary | ✗ | ✔ schema peek | ✗ |
| 2 RetrieverAgent | ✔ multi-angle | ✔ primary | ✔ supplementary | ✗ |
| 3 GraderWriterAgent | ✔ pre-ToT | ✗ consumes result | ✔ fact-checking | ✔ **PRIMARY** |
| 4 AnswerGeneratorAgent | ✔ primary | ✔ indirect | ✔ enrichment | ✗ |

---

## Stack

| Component | Technology |
|---|---|
| LLM | Ollama `llama3.1` |
| Agent framework | LlamaIndex `ReActAgent` |
| Embeddings | `BAAI/bge-large-en-v1.5` (HuggingFace, MPS) |
| Chunking | `SemanticSplitterNodeParser` |
| Vector DB | Postgres + pgvector @ `192.168.86.48:5432` |
| Relational DB | MySQL @ `192.168.1.205:3306` |
| DB tooling | Database-MCP (via `npx`) |
| State | Pydantic `WorkflowState` |
| Hardware | Apple M4 Pro, 48 GB RAM, macOS Tahoe 26.2 |
| Python | 3.13.13 |

---

## Project Layout

```
/Users/jeffgatwood/multi_agent_workflow/
├── main.py                            ← CLI entry point
├── requirements.txt
├── .env                               ← credentials (never commit this)
├── docs/                              ← place source documents here for ingestion
├── .vscode/
│   ├── launch.json                    ← 3 debug run configurations
│   └── settings.json                  ← Python path + interpreter
├── config/
│   ├── __init__.py
│   └── settings.py                    ← typed env config (reads .env)
├── core/
│   ├── __init__.py
│   ├── state.py                       ← Pydantic WorkflowState + ToTThought
│   ├── llm_factory.py                 ← Ollama LLM + bge-large-en-v1.5
│   ├── vector_store.py                ← pgvector index + SemanticSplitter ingest
│   └── workflow.py                    ← pipeline orchestrator
├── agents/
│   ├── __init__.py
│   ├── llm_decision_agent.py          ← Agent 1 : ReAct + MCP
│   ├── retriever_agent.py             ← Agent 2 : RAG + ReAct + MCP
│   ├── grader_writer_agent.py         ← Agent 3 : ToT + ReAct + MCP  ← ToT lives here
│   └── answer_generator_agent.py      ← Agent 4 : ReAct + MCP
└── tools/
    ├── __init__.py
    └── mcp_tools.py                   ← MySQL + Postgres FunctionTools via Database-MCP
```

---

## Setup

### 1. Open the project in VS Code

```
File → Open Folder → /Users/jeffgatwood/multi_agent_workflow
```

### 2. Install Python dependencies

Open the VS Code integrated terminal (`⌃ `` ` ```) and run:

```bash
cd /Users/jeffgatwood/multi_agent_workflow
pip3 install -r requirements.txt
```

### 3. Configure credentials in `.env`

Edit `/Users/jeffgatwood/multi_agent_workflow/.env` and fill in:

```
MYSQL_USER=your_user
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=your_database

POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_DATABASE=your_vector_db
```

### 4. Ensure Ollama is running with llama3.1

```bash
ollama serve           # start server (or it may already be running)
ollama pull llama3.1   # download model if not already present
ollama list            # confirm llama3.1 appears
```

### 5. Enable pgvector on your Postgres instance (one-time)

Connect to `192.168.86.48:5432` and run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 6. Confirm Database-MCP is available

```bash
npx database-mcp --version
```

---

## Running

### Option A — VS Code Run & Debug panel

Open the Run & Debug sidebar (`⇧ ⌘ D`) and choose one of:

| Config name | What it does |
|---|---|
| **Run Workflow (Interactive)** | Start the REPL; type queries at the prompt |
| **Run Workflow (Single Query)** | Run one hard-coded query and exit |
| **Ingest Documents** | Chunk + embed + upsert all files in `docs/` |

### Option B — Terminal

```bash
cd /Users/jeffgatwood/multi_agent_workflow

# Interactive mode
python3 main.py

# Single query
python3 main.py "What are the retention policies for customer orders?"

# Ingest documents
python3 main.py --ingest /Users/jeffgatwood/multi_agent_workflow/docs
```

---

## Ingesting Documents

Place your source files (PDF, TXT, DOCX, Markdown, etc.) in:

```
/Users/jeffgatwood/multi_agent_workflow/docs/
```

Then run the **Ingest Documents** launch config or:

```bash
python3 main.py --ingest /Users/jeffgatwood/multi_agent_workflow/docs
```

LlamaIndex's `SimpleDirectoryReader` handles most formats.
`SemanticSplitterNodeParser` chunks each document on semantic boundaries
(not fixed token windows), and `bge-large-en-v1.5` produces 1024-dim
embeddings stored in pgvector.

---

## Tree of Thought — How It Works

ToT is applied in **Agent 3 (GraderWriterAgent)** across three phases:

**Phase 1 — Generate branches**
Three independent LLM calls each use a different grading persona:
- `branch_1` Strict — accept only directly relevant, verifiable chunks
- `branch_2` Lenient — accept anything tangentially related
- `branch_3` Balanced — filter noise, note gaps, optimise for downstream use

**Phase 2 — Evaluate**
A dedicated LLM evaluator call scores all three branches on:
- Relevance (0–40) · Completeness (0–30) · Fidelity (0–30)

**Phase 3 — Select**
The highest-scoring branch's `grade` and `refined_context` propagate into
`WorkflowState` for the AnswerGeneratorAgent.

All branch reasoning and scores are stored in `state.tot_thoughts` for
full observability in the terminal summary panel.

---

## Terminal Output Tags

Every framework activation prints a tagged prefix:

| Tag | Meaning |
|---|---|
| `[ReAct]` | Thought → Action → Observe loop starting |
| `[RAG]` | pgvector ANN search firing; node scores printed |
| `[MCP]` | MySQL or Postgres tool call via Database-MCP |
| `[ToT]` | Branch generation, score table, or winner selection |

---

## Key Design Notes

- **`SemanticSplitterNodeParser`** chunks on semantic boundaries rather than
  fixed token counts, reducing context fragmentation at retrieval time.
- **`bge-large-en-v1.5`** (1024-dim) runs on Apple MPS automatically via
  HuggingFaceEmbedding `device="mps"`.
- **Pydantic `WorkflowState`** is the single source of truth threaded through
  all four agents; each agent reads only its inputs and writes to clearly
  defined output fields.
- **MCP tools** are `FunctionTool` wrappers so ReAct agents can call them
  natively within their reasoning loop without any special glue code.
- **ToT** lives in the GraderWriterAgent because grading retrieved context
  is inherently ambiguous — multiple evaluation strategies explored in
  parallel surface better results than a single linear chain of thought.
