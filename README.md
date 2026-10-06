# AgenticAI

## Multi-Agent RAG System for User Access Security Analysis

**Author:** Jeff Gatwood
**Capstone Project**
**Franklin County Board of Elections**
**2026**

---

# Overview

AgenticAI is a modular, autonomous multi-agent application designed to analyze user access security questions using Retrieval-Augmented Generation (RAG), ReAct reasoning, Tree of Thought (ToT), and the Model Context Protocol (MCP). The system combines semantic document retrieval, structured database access, and multiple cooperating AI agents to produce accurate, explainable, and evidence-based responses.

Unlike traditional single-agent chatbots, AgenticAI divides responsibilities among specialized agents. Each agent performs a single well-defined task, allowing the overall workflow to be more reliable, easier to maintain, and more transparent.

The application was developed as a graduate capstone project demonstrating modern agentic AI architecture while emphasizing cybersecurity, explainability, modular software engineering, and secure database access.

---

# Project Objectives

The primary objectives of AgenticAI are to:

* Demonstrate autonomous multi-agent workflows.
* Integrate Retrieval-Augmented Generation (RAG) using PostgreSQL pgvector.
* Implement ReAct reasoning across multiple agents.
* Use Tree of Thought reasoning for evaluating retrieved evidence.
* Securely access relational databases through the Model Context Protocol (MCP).
* Prevent unsafe behavior through centralized guardrails.
* Produce transparent, explainable decision making.

---

# Key Features

* Four independent autonomous agents
* Retrieval-Augmented Generation (RAG)
* Tree of Thought (ToT) reasoning
* ReAct reasoning framework
* Model Context Protocol (MCP) integration
* PostgreSQL pgvector vector database
* MySQL relational database integration
* Local LLM execution using Ollama
* HuggingFace embedding model
* Shared workflow state
* Comprehensive guardrails
* Runtime monitoring
* Rich terminal visualization

---

# System Architecture

```
                     User Question
                           │
                           ▼
        ┌─────────────────────────────────────┐
        │ Agent 1                             │
        │ LLMDecisionAgent                    │
        │ ReAct + MCP                         │
        └─────────────────────────────────────┘
                     │
         ┌───────────┴────────────┐
         │                        │
         ▼                        ▼
ROUTE: Retriever        ROUTE: Answer Generator
         │                        │
         ▼                        │
┌─────────────────────────────────────┐
│ Agent 2                             │
│ RetrieverAgent                      │
│ RAG + ReAct + MCP                   │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Agent 3                             │
│ GraderWriterAgent                   │
│ Tree of Thought + ReAct + MCP       │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Agent 4                             │
│ AnswerGeneratorAgent                │
│ ReAct + MCP                         │
└─────────────────────────────────────┘
         │
         ▼
      Final Answer
```

---

# Framework Architecture

| Framework                      | Purpose                                  |
| ------------------------------ | ---------------------------------------- |
| ReAct                          | Agent reasoning and tool usage           |
| Retrieval-Augmented Generation | Semantic retrieval from pgvector         |
| Tree of Thought                | Multi-path reasoning and evaluation      |
| MCP                            | Secure access to PostgreSQL and MySQL    |
| Guardrails                     | Validation, monitoring, safety           |
| LlamaIndex                     | Agent orchestration and vector retrieval |

---

# Technology Stack

## Artificial Intelligence

* Ollama
* Llama 3.2
* HuggingFace Embeddings
* BAAI/bge-large-en-v1.5

## Agent Framework

* LlamaIndex

## Databases

* PostgreSQL
* pgvector
* MySQL

## Programming Language

* Python 3.13+

## Supporting Libraries

* SQLAlchemy
* psycopg2
* PyMySQL
* Pydantic
* Rich
* python-dotenv

---

# Multi-Agent Workflow

Each agent performs one specialized responsibility.

## Agent 1 – LLMDecisionAgent

### Responsibilities

* Inspect user query
* Determine whether retrieval is required
* Optionally inspect database schema
* Route workflow

### Frameworks

* ReAct
* MCP

### Inputs

* User query

### Outputs

```
ROUTE: retriever
```

or

```
ROUTE: answer_generator
```

---

## Agent 2 – RetrieverAgent

### Responsibilities

* Semantic document retrieval
* Vector similarity search
* Structured database lookup
* Aggregate retrieved evidence

### Frameworks

* Retrieval-Augmented Generation
* ReAct
* MCP

### Technologies

* PostgreSQL pgvector
* HNSW ANN Index
* SemanticSplitterNodeParser
* QueryEngineTool

---

## Agent 3 – GraderWriterAgent

### Responsibilities

* Evaluate retrieved evidence
* Generate multiple reasoning paths
* Select best reasoning branch
* Rewrite context for downstream use

### Frameworks

* Tree of Thought
* ReAct
* MCP

### Tree of Thought Process

Three independent reasoning branches are generated.

```
Branch 1
Strict Grader

Branch 2
Lenient Grader

Branch 3
Balanced Grader
```

Each branch is evaluated using:

* Relevance
* Completeness
* Fidelity

The highest scoring branch becomes the refined context passed to the Answer Generator.

---

## Agent 4 – AnswerGeneratorAgent

### Responsibilities

* Produce final user response
* Optionally enrich response using database tools
* Distinguish between RAG-grounded and direct responses

### Frameworks

* ReAct
* MCP

### Operating Modes

#### RAG Mode

Uses the refined context produced by the retrieval pipeline.

#### Direct Mode

Answers directly from the LLM when retrieval is unnecessary.

---

# Retrieval Architecture

```
Source Documents
        │
        ▼
SemanticSplitterNodeParser
        │
        ▼
Semantic Chunks
        │
        ▼
BAAI/bge-large-en-v1.5
        │
        ▼
1024-Dimensional Embeddings
        │
        ▼
PostgreSQL pgvector
        │
        ▼
HNSW ANN Search
        │
        ▼
RetrieverAgent
```

---

# ReAct Workflow

Every agent follows the ReAct reasoning pattern.

```
Thought

↓

Action

↓

Observation

↓

Repeat

↓

Final Answer
```

This allows each agent to:

* Reason
* Use tools
* Evaluate observations
* Continue reasoning
* Produce a grounded answer

---

# Model Context Protocol (MCP)

Database access is performed through persistent MCP servers.

Supported databases include:

* PostgreSQL
* MySQL

Available MCP tools include:

* mysql_query
* postgres_query
* mysql_list_tables
* postgres_list_tables

Persistent MCP sessions improve performance by eliminating repeated server startup overhead while maintaining secure database access.

---

# Guardrails

A centralized `guardrails.py` module protects every stage of the workflow.

## Input Validation

* Empty query detection
* Maximum query length
* Prompt injection detection
* Query sanitization

## Tool Validation

* Approved tool allow-list
* Read-only SQL enforcement
* SQL length limits
* Unsafe SQL detection

## Source Verification

Every retrieved source is validated before use.

Required metadata includes:

* source_id
* source_type
* created_at

Unknown source types are rejected before reaching downstream agents.

## Output Constraints

Outputs are validated for:

* Route correctness
* Unsafe conclusions
* Unsupported claims
* Required supporting evidence

## Runtime Monitoring

Each agent records:

* Timestamp
* Execution time
* Warnings
* Validation failures
* Runtime events

These records provide complete workflow observability.

---

# Shared Workflow State

All agents communicate through a shared `WorkflowState` object.

The state contains:

* Original query
* Routing decision
* Retrieved nodes
* Retrieved context
* Tree of Thought branches
* Winning branch
* Grade
* Refined context
* Final answer
* ReAct trace
* Guardrail trace
* Runtime trace
* Error information

This shared state enables loose coupling while preserving complete execution history.

---

# Project Structure

```
AgenticAI/
│
├── agents/
│   ├── llm_decision_agent.py
│   ├── retriever_agent.py
│   ├── grader_writer_agent.py
│   └── answer_generator_agent.py
│
├── core/
│   ├── workflow.py
│   ├── vector_store.py
│   ├── llm_factory.py
│   └── state.py
│
├── tools/
│   └── mcp_tools.py
│
├── config/
│   └── settings.py
│
├── docs/
│
├── guardrails.py
│
├── ingest_documents.py
│
├── main.py
│
├── requirements.txt
│
└── README.md
```

# Prerequisites

Before running AgenticAI, install the following software.

| Software               | Version |
| ---------------------- | ------- |
| Python                 | 3.13+   |
| Ollama                 | Latest  |
| PostgreSQL             | 16+     |
| pgvector Extension     | Latest  |
| MySQL                  | 8.x     |
| Git                    | Latest  |
| Database MCP (`dbmcp`) | Latest  |

Verify Python

```bash
python --version
```

Verify Ollama

```bash
ollama --version
```

Verify PostgreSQL

```bash
psql --version
```

Verify MySQL

```bash
mysql --version
```

---

# Installation

Clone the repository.

```bash
git clone https://github.com/jgatwood43-ship-it/AgenticAI.git
```

Change to the project directory.

```bash
cd AgenticAI
```

Create a virtual environment.

macOS/Linux

```bash
python3 -m venv .venv
```

Windows

```cmd
python -m venv .venv
```

Activate the environment.

macOS/Linux

```bash
source .venv/bin/activate
```

Windows

```cmd
.venv\Scripts\activate
```

Install all Python dependencies.

```bash
pip install -r requirements.txt
```

---

# Python Dependencies

The project relies on the following Python packages.

| Package                            | Purpose                                         |
| ---------------------------------- | ----------------------------------------------- |
| llama-index-core                   | Core agent framework, workflows, vector indexes |
| llama-index-llms-ollama            | Local Ollama integration                        |
| llama-index-embeddings-huggingface | Embedding generation                            |
| llama-index-vector-stores-postgres | PostgreSQL pgvector integration                 |
| psycopg2-binary                    | PostgreSQL driver                               |
| pymysql                            | MySQL driver                                    |
| sqlalchemy                         | Database abstraction                            |
| mcp                                | Model Context Protocol                          |
| python-dotenv                      | Environment variable loading                    |
| pydantic                           | Typed models and workflow state                 |
| rich                               | Rich terminal interface                         |
| asyncio-mqtt                       | Async messaging support                         |

---

# requirements.txt

```text
# Core LLM & Agent Framework
llama-index-core>=0.11.0
llama-index-llms-ollama>=0.4.0
llama-index-embeddings-huggingface>=0.4.0
llama-index-vector-stores-postgres>=0.3.0

# Database Drivers
psycopg2-binary>=2.9.9
pymysql>=1.1.1
sqlalchemy>=2.0.0

# MCP
mcp>=1.0.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
rich>=13.0.0
asyncio-mqtt>=0.16.0
```

---

# Environment Variables

Create a `.env` file in the project root.

Example

```text
#####################################################
# Ollama
#####################################################

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

#####################################################
# PostgreSQL
#####################################################

PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=AgenticAIvectorDB
PG_USER=postgres
PG_PASSWORD=your_password
PG_VECTOR_TABLE=data_document_embeddings_v3

#####################################################
# MySQL
#####################################################

MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=AgenticAI
MYSQL_USER=root
MYSQL_PASSWORD=your_password

#####################################################
# Embedding Model
#####################################################

EMBED_MODEL_NAME=BAAI/bge-large-en-v1.5

#####################################################
# Semantic Splitter
#####################################################

SEMANTIC_BREAKPOINT_THRESHOLD=95

#####################################################
# ReAct
#####################################################

REACT_MAX_ITERATIONS=10
```

---

# PostgreSQL Setup

Enable the pgvector extension.

```sql
CREATE EXTENSION vector;
```

Create the vector database.

```sql
CREATE DATABASE AgenticAIvectorDB;
```

The application automatically creates or connects to the configured vector table through `PGVectorStore`.

The vector index uses:

* 1024-dimensional embeddings
* HNSW Approximate Nearest Neighbor indexing
* Semantic chunk retrieval

---

# MySQL Setup

Create the application database.

```sql
CREATE DATABASE AgenticAI;
```

Grant the application account read-only access.

The MCP tools are designed to execute **read-only** SQL statements. Guardrails reject destructive SQL commands such as `DROP`, `DELETE`, `UPDATE`, `ALTER`, and `TRUNCATE`.

---

# Ollama Setup

Download the required language model.

```bash
ollama pull llama3.2
```

Start the Ollama server.

```bash
ollama serve
```

Verify the model.

```bash
ollama list
```

---

# Embedding Model

AgenticAI uses the HuggingFace embedding model:

```
BAAI/bge-large-en-v1.5
```

Features include:

* 1024-dimensional embeddings
* Semantic similarity search
* Apple Silicon (MPS) acceleration
* CPU fallback support

---

# Model Context Protocol (MCP)

Database communication is handled using persistent MCP sessions.

Supported databases:

* PostgreSQL
* MySQL

Available tools:

* `mysql_query`
* `postgres_query`
* `mysql_list_tables`
* `postgres_list_tables`

Persistent sessions reduce startup latency and improve workflow performance.

---

# Building the Vector Database

Before asking questions, documents must be ingested into the vector database.

Run:

```bash
python ingest_documents.py
```

The ingestion process performs:

1. Load documents.
2. Perform semantic chunking.
3. Generate embeddings.
4. Insert vectors into PostgreSQL pgvector.
5. Build the HNSW index.

---

# Running the Application

Start the application.

```bash
python main.py
```

Example question:

```
What are the most common user access security risks?
```

---

# Example Workflow Execution

```text
User Query
      │
      ▼
LLMDecisionAgent
      │
      ▼
RetrieverAgent
      │
      ▼
GraderWriterAgent
      │
      ▼
AnswerGeneratorAgent
      │
      ▼
Final Answer
```

If retrieval is unnecessary:

```text
User Query
      │
      ▼
LLMDecisionAgent
      │
      ▼
AnswerGeneratorAgent
      │
      ▼
Final Answer
```

---

# Runtime Logging

Each workflow execution records:

* ReAct reasoning traces
* Tree of Thought branches
* Guardrail events
* Runtime monitoring events
* Errors
* Final answer

This information supports debugging, auditing, and explainability.

---

# Observability

The application provides rich terminal output showing:

* Workflow initialization
* Agent execution
* Routing decisions
* Vector retrieval statistics
* Tree of Thought branch scores
* Final answer
* Pipeline summary

The Rich library is used to produce formatted tables and panels.

---

# Troubleshooting

## Ollama Not Running

```text
Connection refused
```

Start the server:

```bash
ollama serve
```

---

## PostgreSQL Connection Error

Verify:

* PostgreSQL service is running
* pgvector extension is installed
* `.env` values are correct

---

## MySQL Connection Error

Verify:

* MySQL service is running
* User credentials
* Database permissions

---

## Empty Retrieval Results

Verify:

* Documents were ingested successfully
* pgvector table exists
* Embedding model matches the stored vectors

---

## MCP Errors

Verify:

* `dbmcp` is installed
* MCP server is reachable
* Database credentials are valid

---

## HuggingFace Download Errors

Ensure internet connectivity for the initial model download. Once downloaded, the embedding model is cached locally.

---

# Security Considerations

AgenticAI incorporates multiple defensive mechanisms.

## Input Security

* Prompt injection detection
* Input validation
* Length limits

## Tool Security

* Read-only SQL enforcement
* Tool allow-list
* SQL validation

## Source Validation

* Metadata verification
* Source-type validation
* Evidence requirements

## Output Validation

* Unsupported conclusion detection
* Route validation
* Output sanitization

---

# Future Enhancements

Planned improvements include:

* LangGraph orchestration
* Multi-agent collaboration
* Long-term memory
* Human-in-the-loop review
* Knowledge graph integration
* Hybrid retrieval
* OpenTelemetry metrics
* Agent benchmarking
* Automated evaluation suite
* Web interface
* Docker deployment
* Kubernetes support
* Multi-modal document retrieval

---

# Contributing

Future contributions should follow these guidelines:

* Maintain modular agent design.
* Keep business logic separated from orchestration.
* Add unit tests for new features.
* Update documentation for architectural changes.
* Follow Python style guidelines (PEP 8).

---

# Copyright Notice

Copyright © 2026 Jeff Gatwood

All Rights Reserved.

This repository is made publicly available solely for viewing and evaluation as part of an academic capstone project and professional portfolio.

No permission is granted to copy, modify, distribute, sublicense, publish, merge, sell, or otherwise use any portion of this software or its documentation without the prior written permission of the copyright holder.

Viewing the source code through GitHub's interface is permitted for evaluation purposes only.

For permission requests, please contact:

Jeff Gatwood

GitHub: https://github.com/jgatwood43-ship-it

---

# Acknowledgments

This project builds upon the work of the following open-source communities:

* LlamaIndex
* Ollama
* HuggingFace
* PostgreSQL
* pgvector
* Rich
* Pydantic
* SQLAlchemy

---

# References

* LlamaIndex Documentation
* Ollama Documentation
* HuggingFace Documentation
* PostgreSQL Documentation
* pgvector Documentation
* Model Context Protocol Specification

---

# Author

**Jeff Gatwood**

Capstone Project
Agentic AI for User Access Security Analysis

2026

---

## Repository Summary

AgenticAI demonstrates how multiple autonomous AI agents can collaborate using ReAct reasoning, Retrieval-Augmented Generation, Tree of Thought evaluation, secure Model Context Protocol database access, and centralized guardrails to answer user access security questions accurately, transparently, and safely.

The project emphasizes modular design, explainable AI, secure data access, and production-oriented software engineering practices suitable for both academic research and real-world enterprise applications.
