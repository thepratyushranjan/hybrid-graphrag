from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Known embedding sizes, used to catch an EMBEDDING_DIM that would not match the Qdrant collection.
KNOWN_EMBEDDING_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


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

    # Provider credentials
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    # LLM & generation
    llm_provider: Literal["openai", "gemini", "ollama"] = "openai"
    llm_model: str = "gpt-4o-mini"

    # Embeddings
    embedding_provider: Literal["huggingface", "openai"] = "huggingface"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Retrieval
    top_k: int = Field(default=4, ge=1)
    graph_hops: int = Field(default=1, ge=1, le=2)

    @model_validator(mode="after")
    def check_providers(self) -> Self:
        uses_openai = self.llm_provider == "openai" or self.embedding_provider == "openai"
        if uses_openai and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER or EMBEDDING_PROVIDER is 'openai'")
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER is 'gemini'")

        expected_dim = KNOWN_EMBEDDING_DIMS.get(self.embedding_model)
        if expected_dim is not None and expected_dim != self.embedding_dim:
            raise ValueError(
                f"EMBEDDING_DIM={self.embedding_dim} does not match {self.embedding_model} ({expected_dim})"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as exc:
        problems = "; ".join(
            f"{str(err['loc'][0]).upper()}: {err['msg']}" if err["loc"] else err["msg"] for err in exc.errors()
        )
        raise RuntimeError(f"Invalid settings in .env -> {problems}") from exc
