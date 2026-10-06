"""
Schema retrieval support for SQL generation.

Retrieves the database schema documents most relevant to a user's question.
"""

from __future__ import annotations

from typing import Any


class SchemaRetriever:
    """Retrieve relevant schema documents from the schema vector index."""

    def __init__(
        self,
        schema_index: Any,
        similarity_top_k: int = 5,
    ) -> None:
        if schema_index is None:
            raise ValueError("schema_index cannot be None.")

        self._retriever = schema_index.as_retriever(
            similarity_top_k=similarity_top_k,
        )

    def retrieve(self, question: str) -> list[Any]:
        """Retrieve schema nodes relevant to a natural-language question."""

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError("Question cannot be empty.")

        return self._retriever.retrieve(cleaned_question)

    def retrieve_context(self, question: str) -> str:
        """Return retrieved schema documents as formatted text."""

        retrieved_nodes = self.retrieve(question)
        schema_sections: list[str] = []

        for retrieved_node in retrieved_nodes:
            node = getattr(retrieved_node, "node", retrieved_node)

            text = getattr(node, "text", None)

            if text is None and hasattr(node, "get_content"):
                text = node.get_content()

            if text:
                schema_sections.append(str(text).strip())

        return "\n\n---\n\n".join(schema_sections)
