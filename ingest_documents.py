"""
ingest_documents.py
───────────────────
Clean ingestion pipeline for policy/RAG documents.

Key behaviors:
- PDF files are explicitly extracted with PyMuPDFReader.
- HTML/HTM files are parsed with BeautifulSoup.
- TXT/MD files are loaded as UTF-8 text.
- Every discovered file is actually processed.
- Raw PDF bytes and corrupted text are rejected.
- Low-value chunks are skipped instead of aborting ingestion.
- A post-ingestion retrieval test verifies readable RAG content.
- Default logical pgvector table: document_embeddings_v3

PGVectorStore typically creates the physical PostgreSQL table:
    data_document_embeddings_v3
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.postgres import PGVectorStore

load_dotenv()

PG_HOST = os.getenv("PG_HOST", "192.168.86.46")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DATABASE = os.getenv("PG_DATABASE", "AgenticAIvectorDB")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_INGEST_TABLE = os.getenv("PG_INGEST_TABLE", "document_embeddings_v3")

DOCUMENT_DIR = Path(
    os.getenv(
        "RAG_DOCUMENT_DIR",
        "/Users/jeffgatwood/multi_agent_workflow/docs/RAG/",
    )
).expanduser()

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-large-en-v1.5")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))

RETRIEVAL_TEST_QUERY = os.getenv(
    "RAG_TEST_QUERY",
    "What do NIST and CIS say about access control and least privilege?",
)

SUPPORTED_TEXT_SUFFIXES = {".txt", ".md", ".html", ".htm"}

MIN_DOCUMENT_CHARS = 50
MIN_NODE_CHARS = 20
MIN_PRINTABLE_RATIO = 0.90
MIN_ALPHANUMERIC_RATIO = 0.20

RAW_PDF_MARKERS = ("%PDF-", "startxref", "endstream")
BINARY_CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean_text(text: str) -> str:
    cleaned = str(text or "").replace("\x00", "")
    cleaned = BINARY_CONTROL_PATTERN.sub("", cleaned)
    return cleaned.strip()


def _text_quality_metrics(text: str) -> tuple[int, float, float]:
    cleaned = str(text or "")
    if not cleaned:
        return 0, 0.0, 0.0

    printable_count = sum(
        character.isprintable() or character in "\n\r\t" for character in cleaned
    )
    alphanumeric_count = sum(character.isalnum() for character in cleaned)
    length = len(cleaned)

    return (
        length,
        printable_count / length,
        alphanumeric_count / length,
    )


def _contains_raw_pdf_data(text: str) -> bool:
    sample = str(text or "")[:10000]

    if sample.lstrip().startswith("%PDF-"):
        return True

    marker_count = sum(marker in sample for marker in RAW_PDF_MARKERS)
    return marker_count >= 2


def validate_extracted_text(
    text: str,
    *,
    source: str,
    minimum_chars: int,
) -> str:
    cleaned = _clean_text(text)
    length, printable_ratio, alphanumeric_ratio = _text_quality_metrics(cleaned)

    if length < minimum_chars:
        raise ValueError(f"{source}: extracted text is too short ({length} chars).")

    if _contains_raw_pdf_data(cleaned):
        raise ValueError(
            f"{source}: raw PDF data was loaded instead of extracted text."
        )

    if printable_ratio < MIN_PRINTABLE_RATIO:
        raise ValueError(
            f"{source}: text appears corrupted. "
            f"Printable ratio={printable_ratio:.3f}; "
            f"minimum={MIN_PRINTABLE_RATIO:.3f}."
        )

    if alphanumeric_ratio < MIN_ALPHANUMERIC_RATIO:
        raise ValueError(
            f"{source}: text contains too little readable language. "
            f"Alphanumeric ratio={alphanumeric_ratio:.3f}; "
            f"minimum={MIN_ALPHANUMERIC_RATIO:.3f}."
        )

    return cleaned


def _load_pdf(pdf_path: Path) -> list[Document]:
    print(f"  PDF: {pdf_path.name}")

    reader = PyMuPDFReader()
    page_documents = reader.load(file_path=pdf_path)
    validated_documents: list[Document] = []

    for page_number, document in enumerate(page_documents, start=1):
        raw_text = document.get_content()
        stripped = _clean_text(raw_text)

        if len(stripped) < MIN_DOCUMENT_CHARS:
            print(f"    Skipping page {page_number}: {len(stripped)} chars")
            continue

        cleaned_text = validate_extracted_text(
            raw_text,
            source=f"{pdf_path.name} page {page_number}",
            minimum_chars=MIN_DOCUMENT_CHARS,
        )

        metadata = dict(getattr(document, "metadata", {}) or {})
        metadata.update(
            {
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "file_type": "application/pdf",
                "page_number": page_number,
            }
        )

        validated_documents.append(Document(text=cleaned_text, metadata=metadata))

    if not validated_documents:
        raise ValueError(f"{pdf_path.name}: no readable pages were extracted.")

    print(f"    Readable pages: {len(validated_documents)}")
    return validated_documents


def _load_html_file(file_path: Path) -> Document:
    print(f"  HTML: {file_path.name}")

    raw_html = file_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw_html, "html.parser")

    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    extracted_text = soup.get_text(separator="\n", strip=True)

    cleaned_text = validate_extracted_text(
        extracted_text,
        source=file_path.name,
        minimum_chars=MIN_DOCUMENT_CHARS,
    )

    return Document(
        text=cleaned_text,
        metadata={
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_type": "text/html",
        },
    )


def _load_text_file(file_path: Path) -> Document:
    print(f"  Text: {file_path.name}")

    raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    cleaned_text = validate_extracted_text(
        raw_text,
        source=file_path.name,
        minimum_chars=MIN_DOCUMENT_CHARS,
    )

    return Document(
        text=cleaned_text,
        metadata={
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_type": (
                "text/markdown" if file_path.suffix.lower() == ".md" else "text/plain"
            ),
        },
    )


def load_documents(document_dir: Path) -> list[Document]:
    if not document_dir.exists():
        raise FileNotFoundError(f"Document directory does not exist: {document_dir}")

    if not document_dir.is_dir():
        raise NotADirectoryError(
            f"RAG document path is not a directory: {document_dir}"
        )

    supported_files = sorted(
        path
        for path in document_dir.iterdir()
        if (
            path.is_file()
            and (
                path.suffix.lower() == ".pdf"
                or path.suffix.lower() in SUPPORTED_TEXT_SUFFIXES
            )
        )
    )

    if not supported_files:
        raise FileNotFoundError(
            "No supported PDF, HTML, TXT, or Markdown files found in " f"{document_dir}"
        )

    print("\nSupported source files discovered:")
    for file_path in supported_files:
        print(f"  - {file_path.name} ({file_path.suffix.lower()})")

    documents: list[Document] = []

    print("\nLoading source documents:")

    for file_path in supported_files:
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            documents.extend(_load_pdf(file_path))

        elif suffix in {".html", ".htm"}:
            try:
                documents.append(_load_html_file(file_path))
            except ValueError as exc:
                print("  WARNING: Skipping HTML file " f"{file_path.name}: {exc}")

        elif suffix in {".txt", ".md"}:
            documents.append(_load_text_file(file_path))

    if not documents:
        raise ValueError("No readable source documents were loaded.")

    print("\nSource loading complete.")
    return documents


def validate_nodes(nodes: Iterable) -> list:
    validated_nodes = []
    skipped_nodes = 0

    for index, node in enumerate(nodes, start=1):
        raw_text = node.get_content()
        cleaned_text = _clean_text(raw_text)
        text_length, printable_ratio, alphanumeric_ratio = _text_quality_metrics(
            cleaned_text
        )

        if _contains_raw_pdf_data(cleaned_text):
            raise ValueError(f"chunk {index}: raw PDF data detected.")

        if printable_ratio < MIN_PRINTABLE_RATIO:
            raise ValueError(
                f"chunk {index}: content appears corrupted. "
                f"Printable ratio={printable_ratio:.3f}."
            )

        if text_length < MIN_NODE_CHARS:
            skipped_nodes += 1
            print(f"Skipping chunk {index}: only {text_length} chars.")
            continue

        if alphanumeric_ratio < MIN_ALPHANUMERIC_RATIO:
            skipped_nodes += 1
            print(
                f"Skipping chunk {index}: "
                f"alphanumeric ratio={alphanumeric_ratio:.3f}."
            )
            continue

        if hasattr(node, "text"):
            node.text = cleaned_text

        metadata = getattr(node, "metadata", {}) or {}
        for key, value in list(metadata.items()):
            if isinstance(value, str):
                metadata[key] = _clean_text(value)

        validated_nodes.append(node)

    print(
        f"Chunk validation complete: "
        f"{len(validated_nodes)} retained, "
        f"{skipped_nodes} skipped."
    )

    if not validated_nodes:
        raise ValueError("No readable chunks remained after validation.")

    return validated_nodes


def print_source_summary(documents: list[Document]) -> None:
    source_counts = Counter(
        document.metadata.get("file_name", "unknown") for document in documents
    )

    print("\nSource documents successfully loaded:")
    for file_name, count in sorted(source_counts.items()):
        print(f"  {file_name}: {count} page/document objects")


def print_samples(items: list, label: str) -> None:
    print(f"\n{label} samples:")

    for index, item in enumerate(items[:3], start=1):
        text = item.get_content().strip()
        length, printable, alphanumeric = _text_quality_metrics(text)

        print("\n" + "-" * 72)
        print(
            f"{label} {index}: "
            f"{length} chars, "
            f"printable={printable:.3f}, "
            f"alphanumeric={alphanumeric:.3f}"
        )
        print(repr(text[:500]))

    print("-" * 72 + "\n")


def validate_retrieval(index: VectorStoreIndex) -> None:
    print("\nValidating vector retrieval...")
    print(f"Test query: {RETRIEVAL_TEST_QUERY}")

    retriever = index.as_retriever(similarity_top_k=5)
    results = retriever.retrieve(RETRIEVAL_TEST_QUERY)

    if not results:
        raise RuntimeError("Post-ingestion retrieval returned no results.")

    sources_seen: set[str] = set()

    for result_number, result in enumerate(results, start=1):
        content = validate_extracted_text(
            result.node.get_content(),
            source=f"retrieval result {result_number}",
            minimum_chars=MIN_NODE_CHARS,
        )

        score = getattr(result, "score", None)
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"

        metadata = getattr(result.node, "metadata", {}) or {}
        source_name = metadata.get("file_name", "unknown")
        sources_seen.add(str(source_name))

        print("\n" + "-" * 72)
        print(
            f"Result {result_number}: "
            f"score={score_text}, "
            f"source={source_name}, "
            f"chars={len(content)}"
        )
        print(repr(content[:700]))

    print("-" * 72)
    print(f"Readable retrieval results: {len(results)}/{len(results)}")
    print("Sources represented in retrieval:")

    for source_name in sorted(sources_seen):
        print(f"  - {source_name}")


def main() -> None:
    if not PG_PASSWORD:
        raise ValueError("PG_PASSWORD is not set.")

    logical_table_name = (
        PG_INGEST_TABLE[len("data_") :]
        if PG_INGEST_TABLE.startswith("data_")
        else PG_INGEST_TABLE
    )

    if logical_table_name != PG_INGEST_TABLE:
        print(
            "WARNING: PG_INGEST_TABLE started with 'data_'. "
            f"Using logical name '{logical_table_name}' instead."
        )

    print("=" * 72)
    print("AgenticAI Policy RAG Ingestion")
    print("=" * 72)
    print(f"Source directory : {DOCUMENT_DIR}")
    print(f"Postgres         : {PG_HOST}:{PG_PORT}/{PG_DATABASE}")
    print(f"Logical table    : {logical_table_name}")
    print(f"Physical table   : data_{logical_table_name}")
    print(f"Embedding model  : {EMBED_MODEL_NAME}")
    print(f"Embedding dim    : {EMBED_DIM}")

    print("\nLoading embedding model...")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    print(f"\nExtracting readable documents from: {DOCUMENT_DIR}")
    documents = load_documents(DOCUMENT_DIR)

    print_source_summary(documents)

    print(f"\nLoaded {len(documents)} readable page/document objects.")
    print_samples(documents, "Document")

    print("Splitting documents into semantic chunks...")
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} chunks.")

    print("Validating chunks before embedding...")
    nodes = validate_nodes(nodes)
    print(f"Validated {len(nodes)} readable chunks.")
    print_samples(nodes, "Chunk")

    print("Connecting to Postgres pgvector...")
    vector_store = PGVectorStore.from_params(
        database=PG_DATABASE,
        host=PG_HOST,
        password=PG_PASSWORD,
        port=PG_PORT,
        user=PG_USER,
        table_name=logical_table_name,
        embed_dim=EMBED_DIM,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Embedding and inserting chunks into Postgres...")
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    validate_retrieval(index)

    print("\n" + "=" * 72)
    print("INGESTION COMPLETE AND VALIDATED")
    print("=" * 72)
    print(f"Documents/pages : {len(documents)}")
    print(f"Chunks          : {len(nodes)}")
    print(f"Logical table   : {logical_table_name}")
    print(f"Physical table  : data_{logical_table_name}")
    print("\nAfter verifying source coverage in PostgreSQL, set:")
    print(f"    PG_TABLE={logical_table_name}")


if __name__ == "__main__":
    main()
