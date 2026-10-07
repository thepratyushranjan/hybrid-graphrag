"""Neo4j writes against the live container, with a fake LLM. Uses `test_*` sources and cleans up."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from graphrag.config import Settings
from graphrag.embeddings.embedder import HuggingFaceEmbedder
from graphrag.graph.neo4j_store import Neo4jStore
from graphrag.ingestion.pipeline import IngestionPipeline
from graphrag.vector_store.qdrant_store import QdrantStore
from tests.unit.test_extractor import RESULT, TEXT, FakeLLM

pytestmark = pytest.mark.usefixtures("clean_collection")

SOURCE = "test_graph_doc.txt"


@pytest.fixture
def graph(settings: Settings) -> Iterator[Neo4jStore]:
    store = Neo4jStore(settings)
    store.ensure_schema()

    def cleanup() -> None:
        # relations hang between entities (possibly real ones), so delete them by their test chunk_id first
        store.driver.execute_query(
            "MATCH ()-[r:RELATES_TO]->() WHERE r.chunk_id IN "
            "COLLECT { MATCH (c:Chunk) WHERE c.source STARTS WITH 'test_' RETURN c.id } DELETE r"
        )
        store.driver.execute_query("MATCH (n) WHERE n.source STARTS WITH 'test_' DETACH DELETE n")
        store.driver.execute_query("MATCH (e:Entity) WHERE NOT (e)<-[:MENTIONS]-() DETACH DELETE e")

    cleanup()
    yield store
    cleanup()
    store.close()


@pytest.fixture
def pipeline(settings: Settings, embedder: HuggingFaceEmbedder, graph: Neo4jStore, tmp_path: Path) -> IngestionPipeline:
    test_settings = settings.model_copy(update={"llm_cache_dir": str(tmp_path)})
    return IngestionPipeline(test_settings, embedder, QdrantStore(test_settings), graph, llm=FakeLLM(RESULT))  # type: ignore[arg-type]


def test_tests_leave_no_trace(graph: Neo4jStore) -> None:
    """Runs last in this module (alphabetical order is not guaranteed, so check directly)."""
    records, _, _ = graph.driver.execute_query(
        "MATCH ()-[r:RELATES_TO]->() WHERE NOT EXISTS { MATCH (c:Chunk {id: r.chunk_id}) } RETURN count(r) AS n"
    )
    assert records[0]["n"] == 0


def _counts(graph: Neo4jStore) -> dict[str, int]:
    records, _, _ = graph.driver.execute_query(
        "MATCH (d:Document {source: $s})<-[:PART_OF]-(c:Chunk) OPTIONAL MATCH (c)-[m:MENTIONS]->(e:Entity) "
        "OPTIONAL MATCH ()-[r:RELATES_TO {chunk_id: c.id}]->() "
        "RETURN count(DISTINCT c) AS chunks, count(DISTINCT e) AS entities, count(DISTINCT r) AS relations",
        s=SOURCE,
    )
    return dict(records[0])


def test_ingest_twice_gives_identical_graph(pipeline: IngestionPipeline, graph: Neo4jStore) -> None:
    first = pipeline.ingest(SOURCE, TEXT.encode())
    assert first.graph_status == "complete" and first.relations == 3
    counts = _counts(graph)
    assert counts == {"chunks": 1, "entities": 5, "relations": 3}
    pipeline.ingest(SOURCE, TEXT.encode())
    assert _counts(graph) == counts


def test_edited_file_replaces_its_graph(pipeline: IngestionPipeline, graph: Neo4jStore) -> None:
    pipeline.ingest(SOURCE, TEXT.encode())
    pipeline.ingest(SOURCE, b"Completely different text without any of those facts.")
    assert _counts(graph)["relations"] == 0
    records, _, _ = graph.driver.execute_query("MATCH (d:Document {source: $s}) RETURN count(d) AS n", s=SOURCE)
    assert records[0]["n"] == 1


def test_llm_unavailable_keeps_existing_graph(pipeline: IngestionPipeline, graph: Neo4jStore) -> None:
    pipeline.ingest(SOURCE, TEXT.encode())
    pipeline.extractor = None  # simulate a missing API key
    pipeline.llm_error = "no key"
    result = pipeline.ingest(SOURCE, TEXT.encode())
    assert result.graph_status.startswith("skipped") and "kept the existing graph" in result.graph_status
    assert _counts(graph)["relations"] == 3


def test_structural_edges_have_source(pipeline: IngestionPipeline, graph: Neo4jStore) -> None:
    pipeline.ingest(SOURCE, TEXT.encode())
    records, _, _ = graph.driver.execute_query(
        "MATCH (:Document {source: $s})<-[p:PART_OF]-(c:Chunk)-[m:MENTIONS]->() "
        "RETURN collect(DISTINCT p.source) AS part_of, collect(DISTINCT m.source) AS mentions",
        s=SOURCE,
    )
    assert records[0]["part_of"] == ["structured"] and records[0]["mentions"] == ["llm"]


def test_similar_documents_linked_once(pipeline: IngestionPipeline, graph: Neo4jStore) -> None:
    pipeline.settings = pipeline.settings.model_copy(update={"similar_docs_min_score": 0.0})
    pipeline.ingest(SOURCE, TEXT.encode())
    second = pipeline.ingest("test_graph_doc_2.txt", (TEXT + " Updated.").encode())
    assert any(s.startswith(SOURCE) for s in second.similar_documents)
    pipeline.ingest(SOURCE, TEXT.encode())  # re-ingest: recomputes, must not duplicate the edge
    records, _, _ = graph.driver.execute_query(
        "MATCH (a:Document {source: $a})-[r:SIMILAR_TO]-(b:Document {source: $b}) RETURN count(r) AS n",
        a=SOURCE, b="test_graph_doc_2.txt",
    )
    assert records[0]["n"] == 1
