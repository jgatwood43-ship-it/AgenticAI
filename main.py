"""
main.py
────────
Entry point for the Multi-Agent Workflow.

Run interactively:
    python main.py

Or pass a single query:
    python main.py "What are the order retention policies?"

To ingest documents first:
    python main.py --ingest path/to/docs/
"""
from __future__ import annotations

import argparse
import sys
import asyncio
from pathlib import Path

from rich.console import Console

console = Console()


def ingest_mode(docs_path: str) -> None:
    """Load documents from a directory and upsert them into pgvector."""
    from llama_index.core import SimpleDirectoryReader
    from core.llm_factory import build_embed_model
    from core.vector_store import build_vector_store, ingest_documents

    console.print(f"[cyan]Loading documents from: {docs_path}[/cyan]")
    documents = SimpleDirectoryReader(docs_path).load_data()
    console.print(f"[cyan]Loaded {len(documents)} documents.[/cyan]")

    embed_model = build_embed_model()
    vector_store = build_vector_store()
    ingest_documents(documents, embed_model, vector_store)
    console.print("[green]✓ Ingestion complete.[/green]")


# 1. Turn your query loop into an asynchronous function
async def async_query_loop(workflow):
    print("Multi-Agent Workflow CLI Active. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\nQuery: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
                
            if not user_input:
                continue
                
            #  Runs on the same, continuously open event loop
            state = await workflow.run(user_input)
            print(f"\nFinal Answer: {state.answer}")
            
        except Exception as e:
            print(f"Workflow error: {e}")

# 2. Keep your synchronous main setup thin
def main():
    # ... your existing workflow and LLM initialization code ...
    from core.workflow import MultiAgentWorkflow
    workflow = MultiAgentWorkflow() 
    
    # Kicks off the loop exactly ONCE natively
    asyncio.run(async_query_loop(workflow))

if __name__ == "__main__":
    main()

def query_mode(query: str | None) -> None:
   """Run the multi-agent workflow for one query (interactive if None)."""
   from core.workflow import MultiAgentWorkflow

   workflow = MultiAgentWorkflow()

   if query:
       workflow.run(query)
   else:
       console.print("[bold]Multi-Agent Workflow – Interactive Mode[/bold]")
       console.print("Type [bold]exit[/bold] or [bold]quit[/bold] to stop.\n")
       while True:
           try:
               user_input = console.input("[bold green]Query:[/bold green] ").strip()
           except (KeyboardInterrupt, EOFError):
               console.print("\n[yellow]Bye![/yellow]")
               break

           if not user_input:
               continue
           if user_input.lower() in {"exit", "quit"}:
               break

           asyncio.run(workflow.run(user_input))
           workflow.run(user_input)
           console.print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent RAG Workflow (Ollama llama3.1 + pgvector + MCP)"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Query to run (omit for interactive mode).",
    )
    parser.add_argument(
        "--ingest",
        metavar="DOCS_DIR",
        help="Ingest documents from a directory into the vector store, then exit.",
    )
    args = parser.parse_args()

    if args.ingest:
        ingest_mode(args.ingest)
    else:
        query_mode(args.query)


if __name__ == "__main__":
    main()
