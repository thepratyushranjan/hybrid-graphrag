from functools import lru_cache
from typing import Literal

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# Known embedding sizes, used to catch an EMBEDDING_DIM that would not match the Qdrant collection.
KNOWN_EMBEDDING_DIMS: dict[str, int] = {
    "intfloat/multilingual-e5-small": 384,
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
    llm_reasoning_effort: str | None = None  # "none" disables thinking on Ollama/Gemini thinking models

    # Embeddings
    embedding_provider: Literal["huggingface", "openai"] = "huggingface"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384
    embedding_batch_size: int = Field(default=32, ge=1)

    # Reranker
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    # Ingestion
    chunk_size: int = Field(default=500, ge=50)
    chunk_overlap: int = Field(default=50, ge=0)
    ocr_enabled: bool = True
    ocr_languages: str = "eng+hin"
    ocr_min_confidence: float = Field(default=60, ge=0, le=100)
    ocr_min_image_px: int = Field(default=100, ge=1)  # skip icons/logos smaller than this (width or height)
    scanned_page_min_chars: int = Field(default=50, ge=0)  # pages with less text are OCR'd as scans
    ocr_deskew: bool = True
    ocr_max_skew_degrees: float = Field(default=10, ge=0, le=45)

    # Optional: describe images that have no text (photos, charts) with a vision-capable LLM
    vision_captions_enabled: bool = False
    vision_model: str | None = None  # defaults to LLM_MODEL

    # Knowledge graph (LLM triple extraction)
    graph_enabled: bool = True
    extraction_min_confidence: float = Field(default=0.5, ge=0, le=1)
    known_entities_limit: int = Field(default=200, ge=0)  # existing entity names shown to the LLM for reuse
    llm_cache_dir: str = "/app/.cache/llm"
    # (Document)-[:SIMILAR_TO {score}]->(Document) from Qdrant. Score = cosine between a document's
    # centroid and the best-matching chunk of another document. Threshold is embedding-model specific:
    # multilingual-e5 puts related docs at ~0.93 and unrelated ones at ~0.87.
    similar_docs_enabled: bool = True
    similar_docs_top_n: int = Field(default=3, ge=1)
    similar_docs_min_score: float = Field(default=0.90, ge=0, le=1)

    # Retrieval
    top_k: int = Field(default=4, ge=1)
    graph_hops: int = Field(default=1, ge=1, le=2)

    def provider_warnings(self) -> list[str]:
        """Missing keys/models don't stop the server; they are logged and shown in /health."""
        warnings: list[str] = []
        if not self.llm_model:
            warnings.append(f"LLM_MODEL is empty: set a model for LLM_PROVIDER '{self.llm_provider}'")
        if self.llm_provider == "openai" and not self.openai_api_key:
            warnings.append("OPENAI_API_KEY is empty: LLM_PROVIDER 'openai' will not work until it is set")
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            warnings.append("GEMINI_API_KEY is empty: LLM_PROVIDER 'gemini' will not work until it is set")
        if self.vision_captions_enabled and not (self.vision_model or self.llm_model):
            warnings.append("VISION_CAPTIONS_ENABLED=true but neither VISION_MODEL nor LLM_MODEL is set")
        if not self.embedding_model:
            warnings.append(f"EMBEDDING_MODEL is empty: set a model for EMBEDDING_PROVIDER '{self.embedding_provider}'")
        if self.embedding_provider == "openai" and not self.openai_api_key:
            warnings.append("OPENAI_API_KEY is empty: EMBEDDING_PROVIDER 'openai' will not work until it is set")

        expected_dim = KNOWN_EMBEDDING_DIMS.get(self.embedding_model)
        if expected_dim is not None and expected_dim != self.embedding_dim:
            warnings.append(
                f"EMBEDDING_DIM={self.embedding_dim} does not match {self.embedding_model} ({expected_dim})"
            )
        return warnings


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as exc:
        problems = "; ".join(
            f"{str(err['loc'][0]).upper()}: {err['msg']}" if err["loc"] else err["msg"] for err in exc.errors()
        )
        raise RuntimeError(f"Invalid settings in .env -> {problems}") from exc
