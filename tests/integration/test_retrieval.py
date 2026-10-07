"""Hybrid retrieval against live Qdrant (test collection) + Neo4j, with a fake LLM."""

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest

from graphrag.config import Settings
from graphrag.embeddings.embedder import HuggingFaceEmbedder
from graphrag.graph.neo4j_store import Neo4jStore
from graphrag.generation.synthesizer import AnswerGenerator
from graphrag.ingestion.pipeline import IngestionPipeline
from graphrag.llm.cache import JsonCache
from graphrag.retrieval.hybrid import HybridRetriever
from graphrag.retrieval.query_analyzer import QueryAnalyzer
from graphrag.vector_store.qdrant_store import QdrantStore
from tests.unit.test_extractor import RESULT, TEXT, FakeLLM

pytestmark = pytest.mark.usefixtures("clean_collection")

SOURCE = "test_retrieval_doc.txt"
ANALYSIS = {
    "QueryAnalysisLLM": {
        "entities": [{"name": "Clause 7.2", "english_name": "Clause 7.2", "type": "Concept"}],
        "query_type": "lookup",
    }
}


@pytest.fixture
def stores(settings: Settings) -> Iterator[tuple[QdrantStore, Neo4jStore]]:
    vectors, graph = QdrantStore(settings), Neo4jStore(settings)
    graph.ensure_schema()

    def cleanup() -> None:
        graph.driver.execute_query(
            "MATCH ()-[r:RELATES_TO]->() WHERE r.chunk_id IN "
            "COLLECT { MATCH (c:Chunk) WHERE c.source STARTS WITH 'test_' RETURN c.id } DELETE r"
        )
        graph.driver.execute_query("MATCH (n) WHERE n.source STARTS WITH 'test_' DETACH DELETE n")
        graph.driver.execute_query("MATCH (e:Entity) WHERE NOT (e)<-[:MENTIONS]-() DETACH DELETE e")

    cleanup()
    yield vectors, graph
    cleanup()


@pytest.fixture
def retriever(
    settings: Settings, embedder: HuggingFaceEmbedder, stores: tuple[QdrantStore, Neo4jStore], tmp_path: Path
) -> HybridRetriever:
    vectors, graph = stores
    test_settings = settings.model_copy(update={"llm_cache_dir": str(tmp_path), "rerank_enabled": False})
    llm = FakeLLM(RESULT, by_schema=ANALYSIS)
    IngestionPipeline(test_settings, embedder, vectors, graph, llm=llm).ingest(SOURCE, TEXT.encode())  # type: ignore[arg-type]
    analyzer = QueryAnalyzer(test_settings, graph, llm, JsonCache(tmp_path))  # type: ignore[arg-type]
    return HybridRetriever(test_settings, embedder, vectors, graph, analyzer, reranker=None)


def test_modes_differ(retriever: HybridRetriever) -> None:
    q = "What does Clause 7.2 impact?"
    vector = asyncio.run(retriever.retrieve(q, "vector"))
    graph = asyncio.run(retriever.retrieve(q, "graph"))
    hybrid = asyncio.run(retriever.retrieve(q, "hybrid"))

    assert vector.facts == [] and vector.chunks and vector.chunks[0].found_by == ["vector"]
    impacts = ("concept:clause_7_2", "IMPACTS", "org:datasecure_cloud")
    assert impacts in {f.key for f in graph.facts}
    assert graph.analysis.entities[0].entity_id == "concept:clause_7_2"
    assert graph.chunks and all(c.found_by == ["graph"] for c in graph.chunks)  # graph-to-vector bridge
    assert impacts in {f.key for f in hybrid.facts}
    assert hybrid.chunks[0].found_by == ["vector", "graph"]
    assert {"embed_query", "vector_search", "query_analysis", "graph_search", "total"} <= hybrid.timings_ms.keys()


def test_fulltext_fallback_without_llm(retriever: HybridRetriever) -> None:
    retriever.analyzer.llm = None
    result = asyncio.run(retriever.retrieve("Tell me about Shakti Steel Works", "graph"))
    assert result.analysis.analyzer == "fulltext"
    assert "org:shakti_steel_works" in result.analysis.entity_ids


def test_document_filter_scopes_chunks_and_facts(
    retriever: HybridRetriever, embedder: HuggingFaceEmbedder, stores: tuple[QdrantStore, Neo4jStore], tmp_path: Path
) -> None:
    vectors, graph = stores
    other = "test_other_notes.txt"
    settings = retriever.settings
    IngestionPipeline(settings, embedder, vectors, graph, llm=FakeLLM(RESULT)).ingest(  # type: ignore[arg-type]
        other, (TEXT + " A second copy kept in other notes.").encode()
    )
    retriever.analyzer.llm = FakeLLM(RESULT, by_schema={"QueryAnalysisLLM": {
        **ANALYSIS["QueryAnalysisLLM"], "document": other,
    }})  # type: ignore[assignment]
    retriever.analyzer.cache = JsonCache(tmp_path / "fresh")
    result = asyncio.run(retriever.retrieve("In the other notes, what does Clause 7.2 impact?", "hybrid"))

    assert result.analysis.filters == {"source": other}
    assert result.chunks and {c.chunk.source for c in result.chunks} == {other}
    allowed = set(graph.chunk_ids_matching({"source": other}))
    assert result.facts and all(set(f.chunk_ids) <= allowed for f in result.facts)
    assert "vector_search_filtered" in result.timings_ms


def test_time_filter_and_safety_net(
    retriever: HybridRetriever, embedder: HuggingFaceEmbedder, stores: tuple[QdrantStore, Neo4jStore]
) -> None:
    vectors, graph = stores
    dated = "test_dated_notes.txt"
    IngestionPipeline(retriever.settings, embedder, vectors, graph, llm=FakeLLM(RESULT)).ingest(  # type: ignore[arg-type]
        dated, (TEXT + " This happened on 14 October 2025.").encode()
    )
    in_range = asyncio.run(retriever.retrieve("What does Clause 7.2 impact in October 2025?", "hybrid"))
    assert in_range.analysis.time_range is not None
    assert in_range.chunks and {c.chunk.source for c in in_range.chunks} == {dated}
    assert all(set(f.chunk_ids) <= set(graph.chunk_ids_matching({}, in_range.analysis.time_range))
               for f in in_range.facts)

    # nothing is dated 1999: the filter is dropped (and reported) instead of hiding every result
    nothing = asyncio.run(retriever.retrieve("What does Clause 7.2 impact in 1999?", "hybrid"))
    assert nothing.analysis.time_range is None
    assert nothing.analysis.dropped_filters and "1999" in nothing.analysis.dropped_filters[0]
    assert nothing.chunks


def test_query_answer_is_grounded_and_validated(retriever: HybridRetriever) -> None:
    llm = retriever.analyzer.llm
    response = asyncio.run(AnswerGenerator(retriever.settings, retriever, llm).answer("What does Clause 7.2 impact?"))  # type: ignore[arg-type]
    assert response.grounded
    assert "[C1]" in llm.last_prompt and "[G1]" in llm.last_prompt  # type: ignore[union-attr]
    assert response.invalid_citations == ["C9"] and "[C9]" not in response.answer
    assert [c.cite_id for c in response.citations] == ["C1", "G1"]
    assert response.uncited_sentences == ["It also impacts every bank in India."]


def test_no_evidence_skips_the_llm(retriever: HybridRetriever) -> None:
    llm = FakeLLM(RESULT, by_schema={"QueryAnalysisLLM": {"entities": [], "query_type": "lookup"}})
    retriever.analyzer.llm = llm  # type: ignore[assignment]
    response = asyncio.run(AnswerGenerator(retriever.settings, retriever, llm).answer("anything", mode="graph"))  # type: ignore[arg-type]
    assert not response.grounded and "could not find" in response.answer
    assert not hasattr(llm, "last_prompt")  # complete() was never called


def test_llm_failure_returns_evidence_with_note(retriever: HybridRetriever) -> None:
    from graphrag.llm.client import LLMError

    class BrokenLLM(FakeLLM):
        def complete(self, *args: object, **kwargs: object) -> str:
            raise LLMError("ollama/gemma4 request failed: connection refused")

    response = asyncio.run(
        AnswerGenerator(retriever.settings, retriever, BrokenLLM(RESULT)).answer("What does Clause 7.2 impact?")  # type: ignore[arg-type]
    )
    assert not response.grounded and "connection refused" in response.answer
    assert response.chunks and response.graph_facts  # the evidence is still returned
    assert response.subgraph.edges and response.subgraph.nodes


def test_explicit_filters_are_never_dropped(retriever: HybridRetriever) -> None:
    from graphrag.models import QueryFilters

    result = asyncio.run(retriever.retrieve("What does Clause 7.2 impact?", "hybrid",
                                            filters=QueryFilters(source="no_such_file.pdf")))
    assert result.analysis.filters == {"source": "no_such_file.pdf"} and result.chunks == []
