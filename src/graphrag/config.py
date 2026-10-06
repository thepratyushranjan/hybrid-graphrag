from functools import lru_cache
from typing import Literal

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_name: str = "graphrag"
    app_env: str = "dev"
    api_port: int = 8000

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_name: str = "knowledge_vectors"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = Field(min_length=8)

    # LLM
    llm_provider: Literal["groq", "openai", "ollama"] = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_api_key: str | None = None

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Retrieval
    top_k: int = Field(default=4, ge=1)
    graph_hops: int = Field(default=1, ge=1, le=2)


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as exc:
        missing = ", ".join(str(err["loc"][0]).upper() for err in exc.errors())
        raise RuntimeError(
            f"Invalid or missing settings: {missing}. Copy .env.example to .env and fill them in."
        ) from exc
