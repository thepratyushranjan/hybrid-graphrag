from collections.abc import Iterator
from pathlib import Path

import pytest

from graphrag.config import Settings
from graphrag.embeddings.embedder import HuggingFaceEmbedder

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


@pytest.fixture(scope="session")
def settings() -> Settings:
    # DB URLs and the Neo4j password come from the container environment (env_file: .env)
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        llm_provider="ollama",
        collection_name="test_knowledge_vectors",
    )


@pytest.fixture(scope="session")
def embedder(settings: Settings) -> HuggingFaceEmbedder:
    return HuggingFaceEmbedder(settings)


@pytest.fixture(scope="session")
def samples() -> Path:
    return SAMPLES


@pytest.fixture
def clean_collection(settings: Settings) -> Iterator[None]:
    from graphrag.vector_store.qdrant_store import QdrantStore

    store = QdrantStore(settings)
    store.client.delete_collection(settings.collection_name)
    yield
    store.client.delete_collection(settings.collection_name)
    store.close()
