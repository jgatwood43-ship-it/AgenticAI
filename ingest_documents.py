from dotenv import load_dotenv
import os

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

load_dotenv()

PG_HOST = os.getenv("PG_HOST", "192.168.86.43")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DATABASE = os.getenv("PG_DATABASE", "AgenticAIvectorDB")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_TABLE = os.getenv("PG_TABLE", "data_document_embeddings")

DOCUMENT_DIR = "/Users/jeffgatwood/multi_agent_workflow/docs/RAG/"

print("Loading embedding model...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

print(f"Loading documents from: {DOCUMENT_DIR}")
documents = SimpleDirectoryReader(DOCUMENT_DIR).load_data()
print(f"Loaded {len(documents)} documents.")

print("Splitting documents into chunks...")
splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
)
nodes = splitter.get_nodes_from_documents(documents)
print(f"Created {len(nodes)} chunks.")

print("Cleaning chunks...")

for node in nodes:
    if hasattr(node, "text") and node.text:
        node.text = node.text.replace("\x00", "")

    if hasattr(node, "metadata") and node.metadata:
        for key, value in list(node.metadata.items()):
            if isinstance(value, str):
                node.metadata[key] = value.replace("\x00", "")

print("Chunks cleaned.")

print("Connecting to Postgres pgvector...")
vector_store = PGVectorStore.from_params(
    database=PG_DATABASE,
    host=PG_HOST,
    password=PG_PASSWORD,
    port=PG_PORT,
    user=PG_USER,
    table_name=PG_TABLE,
    embed_dim=1024,
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

print("Ingest complete.")
print(f"Ingested {len(nodes)} chunks into Postgres table: {PG_TABLE}")
