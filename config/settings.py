"""
config/settings.py
──────────────────
Loads all environment variables and exposes typed settings used across the workflow.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Ollama
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")

    # Postgres
    pg_host: str = os.getenv("PG_HOST", "localhost")
    pg_port: str = os.getenv("PG_PORT", "5432")
    pg_name: str = os.getenv("PG_DATABASE", "postgres")
    pg_database: str = os.getenv(
        "PG_DATABASE", "postgres"
    )  # Add this line for the vector store
    pg_user: str = os.getenv("PG_USER", "postgres")
    pg_password: str = os.getenv("PG_PASSWORD", "")
    pg_vector_table: str = os.getenv("PG_VECTOR_TABLE", "document_ebeddings")
    pg_table: str = os.getenv(
        "PG__VECTOR_TABLE", "document_embeddings"
    )  # 💡 Add this line for the vector store

    # MySQL
    mysql_host: str = os.getenv("MYSQL_HOST", "localhost")
    mysql_port: str = os.getenv("MYSQL_PORT", "3306")
    mysql_name: str = os.getenv("MYSQL_DATABASE", "mysql")
    mysql_user: str = os.getenv("MYSQL_USER", "root")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "")

    @property
    def pg_connection_string(self) -> str:
        return (
            f"postgresql+psycopg2://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    @property
    def mysql_connection_string(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    # Embedding
    embed_model_name: str = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-large-en-v1.5")

    # Semantic Splitter
    semantic_breakpoint_threshold: int = int(
        os.getenv("SEMANTIC_BREAKPOINT_THRESHOLD", "95")
    )

    # ReAct
    react_max_iterations: int = int(os.getenv("REACT_MAX_ITERATIONS", "10"))


settings = Settings()
