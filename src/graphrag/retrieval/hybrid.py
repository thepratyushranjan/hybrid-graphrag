"""Hybrid retrieval: query analysis -> (vector || graph) in parallel -> graph-to-vector bridge -> fusion
-> rerank -> token budget. `mode` switches branches off so vector-only / graph-only / hybrid can be compared."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from graphrag.config import Settings
from graphrag.embeddings.embedder import Embedder
from graphrag.graph.neo4j_store import Neo4jStore
from graphrag.ingestion.chunker import detect_language
from graphrag.models import (
    Aggregate,
    Chunk,
    GraphFact,
    QueryAnalysis,
    QueryEntity,
    QueryFilters,
    RetrievalMode,
    RetrievalResult,
)
from graphrag.retrieval.fusion import fit_budget, graph_chunk_order, rank_facts, reciprocal_rank_fusion
from graphrag.retrieval.graph_retriever import GraphRetriever
from graphrag.retrieval.query_analyzer import QueryAnalyzer
from graphrag.retrieval.reranker import Reranker
from graphrag.vector_store.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)
T = TypeVar("T")


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        vector_store: QdrantStore,
        graph_store: Neo4jStore,
        analyzer: QueryAnalyzer,
        reranker: Reranker | None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.analyzer = analyzer
        self.graph_retriever = GraphRetriever(settings, graph_store)
        self.reranker = reranker

    async def retrieve(
        self,
        question: str,
        mode: RetrievalMode = "hybrid",
        top_k: int | None = None,
        hops: int | None = None,
        filters: QueryFilters | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self.settings.top_k
        hops = hops or self.settings.graph_hops
        timings: dict[str, float] = {}

        async def timed(name: str, fn: Callable[[], T]) -> T:
            start = time.perf_counter()
            result = await asyncio.to_thread(fn)
            timings[name] = round((time.perf_counter() - start) * 1000, 1)
            return result

        started = time.perf_counter()
        question_vector = await timed("embed_query", lambda: self.embedder.embed_query(question))

        # Step 1 + 2, in parallel:
        #   vector branch: Qdrant search (re-run with payload filters once the analysis finds any)
        #   graph branch:  LLM query analysis -> Cypher templates right away; it waits for the vector hits
        #                  only when it must seed the graph walk from them (no named entity in the question)
        async def vector_search(scope: QueryAnalysis | None) -> list[tuple[Chunk, float]]:
            filtered = scope is not None and bool(scope.filters or scope.time_range)
            hits = await timed(
                "vector_search_filtered" if filtered else "vector_search",
                lambda: self.vector_store.search(
                    question_vector,
                    self.settings.vector_candidates,
                    dict(scope.filters) if scope and scope.filters else None,
                    scope.time_range if scope else None,
                ),
            )
            return [(h.chunk, h.score) for h in hits]

        unfiltered = asyncio.create_task(vector_search(None)) if mode != "graph" else None
        seed_hits: list[tuple[Chunk, float]] | None = None  # reused below so a filtered search runs once

        async def graph_branch() -> tuple[QueryAnalysis, list[GraphFact], list[Aggregate], list[str]]:
            nonlocal seed_hits
            if mode == "vector":
                return self._apply_explicit(QueryAnalysis(language=detect_language(question)), filters), [], [], []
            analysis = await timed("query_analysis", lambda: self.analyzer.analyze(question))
            await timed("check_filters", lambda: self._drop_empty_filters(analysis))  # inferred filters only
            self._apply_explicit(analysis, filters)
            if mode == "hybrid" and not analysis.entity_ids and analysis.query_type == "relationship":
                # Graph-from-vector seeding: the question names no known entity ("the subsidiary whose steel
                # supplier was put on probation"), so start the graph walk from entities in the top chunks
                seed_hits = await self._vector_hits(unfiltered, analysis, vector_search)
                hits = seed_hits
                analysis.entities += await timed("vector_seeds", lambda: self._seed_entities(hits))
            facts, aggregates, templates = await timed(
                "graph_search", lambda: self.graph_retriever.retrieve(analysis, hops)
            )
            return analysis, facts, aggregates, templates

        analysis, raw_facts, aggregates, templates = await graph_branch()
        if seed_hits is not None:
            vector_hits = seed_hits
        elif unfiltered is not None:
            vector_hits = await self._vector_hits(unfiltered, analysis, vector_search)
        else:
            vector_hits = []

        # Step 3: graph-to-vector bridge - fetch the chunks that support the graph facts
        all_facts = await timed("rank_facts", lambda: rank_facts(raw_facts, self._fact_scorer(question, question_vector)))
        graph_chunk_ids = graph_chunk_order(all_facts)[: self.settings.vector_candidates]
        graph_chunks = await timed("bridge_fetch", lambda: self.vector_store.get_chunks(graph_chunk_ids))

        # Step 4: fusion (+ entity-mention boost), rerank, budget
        pool_ids = list({c.chunk_id for c, _ in vector_hits} | {c.chunk_id for c in graph_chunks})
        mentions = (
            await timed("entity_mentions", lambda: self.graph_store.chunk_entity_mentions(pool_ids, analysis.entity_ids))
            if mode != "vector"
            else {}
        )
        fused = reciprocal_rank_fusion(
            vector_hits, graph_chunks, mentions, self.settings.rrf_k, self.settings.entity_mention_boost
        )
        candidates = fused[: self.settings.rerank_candidates]
        if self.reranker is not None and candidates:
            candidates = await timed("rerank", lambda: self.reranker.rerank(question, candidates))  # type: ignore[union-attr]
        facts = all_facts[: self.settings.max_facts]
        chunks, facts, used = fit_budget(
            candidates[:top_k], facts, self.settings.context_token_budget, self.embedder.count_tokens
        )
        timings["total"] = round((time.perf_counter() - started) * 1000, 1)

        return RetrievalResult(
            question=question,
            mode=mode,
            analysis=analysis,
            chunks=chunks,
            facts=facts,
            aggregates=aggregates,
            templates=templates,
            context_tokens=used,
            timings_ms=timings,
        )

    @staticmethod
    async def _vector_hits(
        unfiltered: "asyncio.Task[list[tuple[Chunk, float]]] | None",
        analysis: QueryAnalysis,
        search: Callable[[QueryAnalysis | None], Awaitable[list[tuple[Chunk, float]]]],
    ) -> list[tuple[Chunk, float]]:
        """The unfiltered search started at once (so it overlaps the LLM analysis); if the analysis produced
        payload filters / a time range, search again with them - a cheap indexed Qdrant query."""
        if analysis.filters or analysis.time_range:
            if unfiltered is not None:
                unfiltered.cancel()
            return await search(analysis)
        return await unfiltered if unfiltered is not None else []

    @staticmethod
    def _apply_explicit(analysis: QueryAnalysis, filters: QueryFilters | None) -> QueryAnalysis:
        """Caller-given filters win over inferred ones and are never dropped."""
        if filters is not None:
            analysis.filters.update(filters.exact())
            if (time_range := filters.time()) is not None:
                analysis.time_range = time_range
        return analysis

    def _drop_empty_filters(self, analysis: QueryAnalysis) -> None:
        """Safety net: a filter that matches no chunk would silently hide everything. Drop the time range
        first (dates in text are the least reliable signal), then the document filters, and record it."""
        if not (analysis.filters or analysis.time_range):
            return
        if self.vector_store.count(dict(analysis.filters) or None, analysis.time_range) > 0:
            return
        if analysis.time_range is not None:
            analysis.dropped_filters.append(f"time: {analysis.time_range.expression} (no dated chunks match)")
            analysis.time_range = None
            if not analysis.filters or self.vector_store.count(dict(analysis.filters)) > 0:
                return
        analysis.dropped_filters.append(f"filters: {analysis.filters} (no chunks match)")
        analysis.filters = {}

    def _fact_scorer(self, question: str, question_vector: list[float]) -> Callable[[list[str]], list[float]]:
        """Cross-encoder relevance when the reranker is loaded, else embedding cosine."""
        if self.reranker is not None:
            reranker = self.reranker
            return lambda texts: reranker.relevance(question, texts)

        def cosine(texts: list[str]) -> list[float]:
            vectors = self.embedder.embed_documents(texts)
            return [sum(a * b for a, b in zip(v, question_vector, strict=True)) for v in vectors]

        return cosine

    def _seed_entities(self, vector_hits: list[tuple[Chunk, float]]) -> list[QueryEntity]:
        top_chunks = [c.chunk_id for c, _ in vector_hits[: self.settings.vector_seed_chunks]]
        scores: dict[str, float] = {}
        names: dict[str, str] = {}
        for row in self.graph_store.entities_in_chunks(top_chunks, self.settings.hub_degree_limit):
            scores[row["id"]] = scores.get(row["id"], 0.0) + 1 / (1 + row["chunk_rank"])  # earlier chunk = stronger
            names[row["id"]] = row["name"]
        best = sorted(scores, key=lambda e: -scores[e])[: self.settings.vector_seed_entities]
        return [QueryEntity(text=names[e], entity_id=e, entity_name=names[e], origin="vector") for e in best]


def run_sync(coro: Awaitable[T]) -> T:
    """For scripts: run an async retrieval from synchronous code."""
    return asyncio.run(coro)  # type: ignore[arg-type]
