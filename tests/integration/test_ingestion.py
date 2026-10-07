"""Runs against the live Qdrant container, in a separate test collection."""

from pathlib import Path

import pytest

from graphrag.config import Settings
from graphrag.embeddings.embedder import HuggingFaceEmbedder
from graphrag.ingestion.pipeline import IngestionPipeline
from graphrag.vector_store.qdrant_store import QdrantStore

pytestmark = pytest.mark.usefixtures("clean_collection")


@pytest.fixture
def pipeline(settings: Settings, embedder: HuggingFaceEmbedder) -> IngestionPipeline:
    return IngestionPipeline(settings, embedder, QdrantStore(settings))


def test_reingesting_same_files_keeps_point_count(pipeline: IngestionPipeline, samples: Path) -> None:
    files = sorted(samples.glob("*.*"))
    for path in files:
        pipeline.ingest_path(path)
    first = pipeline.vector_store.count()
    for path in files:
        assert pipeline.ingest_path(path).replaced_points == 0
    assert pipeline.vector_store.count() == first > 0


def test_edited_file_replaces_its_old_chunks(pipeline: IngestionPipeline) -> None:
    pipeline.ingest("notes.txt", ("Old text about Shakti Steel Works. " * 200).encode())
    old_count = pipeline.vector_store.count({"source": "notes.txt"})
    result = pipeline.ingest("notes.txt", b"New short text about NovaGrid Electronics.")
    assert result.replaced_points == old_count
    assert pipeline.vector_store.count({"source": "notes.txt"}) == 1


def test_search_ranks_relevant_chunk_first_across_languages(
    pipeline: IngestionPipeline, embedder: HuggingFaceEmbedder, samples: Path
) -> None:
    for path in sorted(samples.glob("*.*")):
        pipeline.ingest_path(path)
    store = pipeline.vector_store

    hits = store.search(embedder.embed_query("Clause 11.4 water treatment chemical reports"), top_k=1)
    assert hits[0].chunk.page == 3

    # Hindi question, top hits are about the Shakti Steel probation (Hindi section or the scanned English page)
    hits = store.search(embedder.embed_query("शक्ति स्टील वर्क्स को परिवीक्षा पर क्यों रखा गया?"), top_k=2)
    assert all("Shakti" in h.chunk.text or "शक्ति" in h.chunk.text for h in hits)

    hits = store.search(embedder.embed_query("vendors"), top_k=10, filters={"doc_type": "pdf"})
    assert hits and all(h.chunk.doc_type == "pdf" for h in hits)
