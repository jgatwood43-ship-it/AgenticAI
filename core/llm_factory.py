"""
core/llm_factory.py
────────────────────
Centralised factory for the Ollama LLM and the bge-large-en-v1.5 embedding model.
Both are module-level singletons so they are only initialised once.
"""

from __future__ import annotations

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings as LlamaSettings

from config.settings import settings


def build_llm() -> Ollama:
    """Return a configured Ollama LLM instance (llama3.2)."""
    return Ollama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        request_timeout=240.0,
        temperature=0.0,
        context_window=16384,
        additional_kwargs={"num_ctx": 16384},
    )


def build_embed_model() -> HuggingFaceEmbedding:
    """Return bge-large-en-v1.5 embedding model via HuggingFace."""
    return HuggingFaceEmbedding(
        model_name=settings.embed_model_name,
        # Apple Silicon: use MPS if available, fall back to CPU
        device="mps",
    )


def configure_llama_globals() -> None:
    """
    Inject the LLM and embedding model into LlamaIndex's global Settings
    so every component picks them up automatically.
    """
    LlamaSettings.llm = build_llm()
    LlamaSettings.embed_model = build_embed_model()
