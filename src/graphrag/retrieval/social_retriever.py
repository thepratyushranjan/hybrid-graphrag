"""Hybrid retrieval over the SQL corpus (social-media posts).

    question -> SocialQueryAnalyzer (filters, seed entities, query type, group-by)
      vector branch: Qdrant `social_posts`, payload-filtered (district / platform / sentiment / category / time)
      graph branch:  Cypher templates - posts linked to the seeds, 2-hop co-mentions, the context of the top
                     posts (topic, district, thana, category, people, accounts) and exact counts for
                     aggregation questions
    -> graph-to-vector bridge (posts behind the facts) -> RRF fusion -> rerank -> token budget

Returns the same RetrievalResult as the document retriever, so prompts, citations and the UI are shared.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from graphrag.config import Settings
from graphrag.embeddings.embedder import Embedder
from graphrag.graph.social_store import SocialGraphStore, filter_params
from graphrag.llm.client import LLMClient
from graphrag.models import Aggregate, Chunk, GraphFact, QueryAnalysis, QueryFilters, RetrievalMode, RetrievalResult
from graphrag.retrieval.fusion import fit_budget, graph_chunk_order, rank_facts, reciprocal_rank_fusion
from graphrag.retrieval.reranker import Reranker
from graphrag.retrieval.social_analyzer import SocialQueryAnalyzer
from graphrag.vector_store.qdrant_store import SocialPostStore

logger = logging.getLogger(__name__)
T = TypeVar("T")

# filter key -> Qdrant payload field
PAYLOAD_KEYS = {"district": "district", "thana": "thana", "platform": "platform", "sentiment": "sentiment",
                "category": "broad_category", "sub_category": "sub_category"}
# the least reliable filters are dropped first when a combination matches no post
DROP_ORDER = ("time", "sub_category", "category", "sentiment", "platform", "thana", "district")
AGG_TOP = 10
CONTEXT_POSTS = 8  # posts whose graph neighbourhood is added as facts
FACT_LIMIT = 60  # rows per Cypher template (every fact is scored by the cross-encoder)
# Prompt size: Hindi posts are token-heavy and local models often run with a 4k context (Ollama's default), where
# an overlong prompt is silently cut from the start - losing the instructions. Posts are shortened for the prompt
# only (the stored vectors keep the full text).
PROMPT_BUDGET = 1800
PROMPT_FACTS = 12
EXCERPT_TEXT_CHARS = 350
EXCERPT_SUMMARY_CHARS = 250
FACT_EVIDENCE_CHARS = 100
# 2-hop facts say what the data shows: both appear in the same posts (not a real-world relationship)
CO_PREDICATES = {"MENTIONS": "MENTIONED_TOGETHER_WITH", "STANCE": "MENTIONED_TOGETHER_WITH",
                 "MENTIONS_ACCOUNT": "MENTIONED_WITH_ACCOUNT", "POSTED_BY": "MENTIONED_IN_POSTS_BY",
                 "TAGGED": "MENTIONED_WITH_HASHTAG", "PART_OF": "MENTIONED_IN_TOPIC"}
TYPE_NAMES = {"person": "Person", "organisation": "Organization", "location": "Location", "incident": "Incident",
              "entity": "Entity"}


def node_type(label: str | None, kind: str | None) -> str | None:
    if label == "SocialEntity":
        return TYPE_NAMES.get(kind or "", "Entity")
    return label


def post_name(row: dict[str, Any]) -> str:
    who = f"@{row['author']}" if row.get("author") else row.get("platform") or "post"
    return f"Post #{row['row_id']} ({who}, {row.get('date') or 'undated'})"


def post_fact(row: dict[str, Any], template: str, hops: int = 1) -> GraphFact:
    predicate = row["predicate"]
    if predicate == "STANCE" and row.get("stance"):
        predicate = f"STANCE_{str(row['stance']).upper()}"
    return GraphFact(
        subject_id=row["post_id"], subject=post_name(row), subject_type="Post",
        predicate=predicate, object_id=row["object_id"], object=row["object"],
        object_type=node_type(row.get("label"), row.get("kind")),
        hops=hops, confidence=1.0, chunk_ids=[row["post_id"]], sources=[row.get("platform") or "post"],
        evidence=(row.get("evidence") or "")[:FACT_EVIDENCE_CHARS], template=template,
    )


def count_fact(
    node_id: str, name: str, node_type: str, n: int, scope: str, posts: list[str], predicate: str = "MATCHING_POSTS"
) -> GraphFact:
    """An exact database count as a citable fact: (Deoria) -[MATCHING_POSTS]-> (240 posts)."""
    return GraphFact(
        subject_id=node_id, subject=name, subject_type=node_type, predicate=predicate,
        object_id=f"count:{predicate}:{node_id}", object=f"{n} posts", object_type="Count", hops=1, confidence=1.0,
        chunk_ids=posts, sources=["database count"], evidence=f"Exact count over all posts matching: {scope}",
        template="count",
    )


def qdrant_filters(filters: dict[str, str]) -> dict[str, str]:
    return {PAYLOAD_KEYS[k]: v for k, v in filters.items() if k in PAYLOAD_KEYS}


class SocialRetriever:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        vector_store: SocialPostStore,
        graph_store: SocialGraphStore,
        analyzer: SocialQueryAnalyzer,
        reranker: Reranker | None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.analyzer = analyzer
        self.reranker = reranker

    async def retrieve(
        self,
        question: str,
        mode: RetrievalMode = "hybrid",
        top_k: int | None = None,
        hops: int | None = None,
        filters: QueryFilters | None = None,
        rerank: bool | None = None,
        llm: LLMClient | None = None,  # noqa: ARG002 - analysis is deterministic; same signature as HybridRetriever
    ) -> RetrievalResult:
        top_k = top_k or self.settings.top_k
        hops = hops or self.settings.graph_hops
        reranker = self.reranker if rerank is not False else None
        timings: dict[str, float] = {}

        async def timed(name: str, fn: Callable[[], T]) -> T:
            start = time.perf_counter()
            result = await asyncio.to_thread(fn)
            timings[name] = round((time.perf_counter() - start) * 1000, 1)
            return result

        started = time.perf_counter()
        question_vector, analysis = await asyncio.gather(
            timed("embed_query", lambda: self.embedder.embed_query(question)),
            timed("query_analysis", lambda: self.analyzer.analyze(question)),
        )
        if filters is not None:  # caller filters win and are never dropped
            analysis.filters.update(filters.social())
            if (time_range := filters.time()) is not None:
                analysis.time_range = time_range
        await timed("check_filters", lambda: self._drop_empty_filters(analysis, filters))

        async def vector_branch() -> list[tuple[Chunk, float]]:
            if mode == "graph":
                return []
            hits = await timed("vector_search", lambda: self.vector_store.search(
                question_vector, self.settings.vector_candidates,
                qdrant_filters(analysis.filters) or None, analysis.time_range))
            return [(h.chunk, h.score) for h in hits]

        vector_hits = await vector_branch()
        facts: list[GraphFact] = []
        aggregates: list[Aggregate] = []
        templates: list[str] = []
        if mode != "vector":
            seed_posts = [c.chunk_id for c, _ in vector_hits[: self.settings.vector_seed_chunks]] if mode == "hybrid" else []
            facts, aggregates, templates = await timed(
                "graph_search", lambda: self._graph(analysis, hops, seed_posts))

        # exact counts are facts too (citable [G#]) and always come first, in count order, then the sample
        # posts behind them; everything else is ranked by relevance
        counts = [f for f in facts if f.template == "count"]
        ranked = await timed("rank_facts", lambda: rank_facts(
            [f for f in facts if f.template != "count"], self._fact_scorer(question, question_vector, reranker), None))
        all_facts = (counts + [f for f in ranked if f.template == "aggregation"]
                     + [f for f in ranked if f.template != "aggregation"])
        graph_post_ids = graph_chunk_order(all_facts)[: self.settings.vector_candidates]
        graph_posts = await timed("bridge_fetch", lambda: self.vector_store.get_chunks(graph_post_ids))

        pool_ids = list({c.chunk_id for c, _ in vector_hits} | {c.chunk_id for c in graph_posts})
        mentions = (await timed("entity_mentions", lambda: self.graph_store.post_entity_mentions(
            pool_ids, analysis.entity_ids)) if mode != "vector" else {})
        fused = reciprocal_rank_fusion(
            vector_hits, graph_posts, mentions, self.settings.rrf_k, self.settings.entity_mention_boost)
        candidates = fused[: self.settings.rerank_candidates]
        if reranker is not None and candidates:
            candidates = await timed("rerank", lambda: reranker.rerank(question, candidates))
        chunks, kept_facts, used = fit_budget(
            [item.model_copy(update={"chunk": self._excerpt(item.chunk)}) for item in candidates[:top_k]],
            all_facts[: min(self.settings.max_facts, PROMPT_FACTS)],
            min(self.settings.context_token_budget, PROMPT_BUDGET), self.embedder.count_tokens)
        timings["total"] = round((time.perf_counter() - started) * 1000, 1)
        return RetrievalResult(
            question=question, mode=mode, analysis=analysis, chunks=chunks, facts=kept_facts,
            aggregates=aggregates, templates=templates, context_tokens=used, timings_ms=timings,
        )

    # ---------- graph branch ----------

    def _graph(
        self, analysis: QueryAnalysis, hops: int, seed_posts: list[str]
    ) -> tuple[list[GraphFact], list[Aggregate], list[str]]:
        ids = analysis.entity_ids
        limit = min(self.settings.graph_fact_limit, FACT_LIMIT)
        params = {**filter_params(analysis.filters, analysis.time_range), "ids": ids, "limit": limit}
        facts: list[GraphFact] = []
        aggregates: list[Aggregate] = []
        templates: list[str] = []

        entity_rows: list[dict[str, Any]] = []
        if ids:
            entity_rows = self.graph_store.run("entity_posts", params)
            facts += [post_fact(r, "entity_posts") for r in entity_rows]
            templates.append("entity_posts")
            if hops >= 2 or analysis.query_type == "relationship":
                facts += self._co_mentions(params)
                templates.append("co_mentions")
            stances = self.graph_store.run("stance_counts", params)
            if stances:
                templates.append("stance_counts")
            for r in stances:
                facts.append(count_fact(
                    r["id"], r["name"], node_type("SocialEntity", r["kind"]) or "Entity", r["n"],
                    f"stance '{r['stance']}' towards {r['name']} (sentiment_entities table)", r["posts"],
                    predicate=f"STANCE_{str(r['stance']).upper()}_POSTS",
                ))

        if analysis.query_type == "aggregation":
            aggs, agg_facts = self._aggregate(analysis, params)
            aggregates += aggs
            facts += agg_facts
            templates.append("aggregation")

        # context of the most relevant posts: posts about the seeds, the top vector hits, or (graph mode,
        # no named entity) the newest posts matching the filters
        if analysis.query_type == "aggregation":
            return facts, aggregates, templates  # the counts and their sample posts are the evidence
        context = list(dict.fromkeys([r["post_id"] for r in entity_rows][:CONTEXT_POSTS // 2] + seed_posts))
        if not context and not ids and analysis.filters:
            context = [r["id"] for r in self.graph_store.run("filtered_posts", {**params, "limit": CONTEXT_POSTS})]
        if context:
            rows = self.graph_store.run("post_context", {"post_ids": context[:CONTEXT_POSTS], "limit": limit})
            facts += [post_fact(r, "post_context") for r in rows]
            templates.append("post_context")
        return facts, aggregates, templates

    def _co_mentions(self, params: dict[str, Any]) -> list[GraphFact]:
        facts = []
        for r in self.graph_store.run("co_mentions", params):
            examples = [s for s in r["samples"] if s.get("evidence")]
            evidence = f"Both appear in {r['n']} post(s)" + (f", e.g. {post_name(examples[0])}: {examples[0]['evidence']}"
                                                         if examples else "")
            facts.append(GraphFact(
                subject_id=r["subject_id"], subject=r["subject"], subject_type=node_type(r["s_label"], r["s_kind"]),
                predicate=CO_PREDICATES.get(r["rel"], "MENTIONED_TOGETHER_WITH"), object_id=r["object_id"], object=r["object"],
                object_type=node_type(r["label"], r["kind"]), hops=2, confidence=1.0,
                chunk_ids=[s["post_id"] for s in r["samples"]], sources=["social graph"],
                evidence=evidence[:300], template="co_mentions",
            ))
        return facts

    def _aggregate(self, analysis: QueryAnalysis, params: dict[str, Any]) -> tuple[list[Aggregate], list[GraphFact]]:
        """Exact counts in Cypher. The total is always included; groups come from the question's group-by words,
        else a sensible default (by district, or by sub-category once a district is fixed)."""
        scope = ", ".join(f"{k}={v}" for k, v in analysis.filters.items()) or "all posts"
        if analysis.time_range:
            scope += f", time {analysis.time_range.start or '…'} → {analysis.time_range.end or '…'}"
        if analysis.entities:
            scope += ", linked to " + ", ".join(e.entity_name or e.text for e in analysis.entities)
        total = self.graph_store.total(params)
        aggregates = [Aggregate(name=f"total matching posts ({scope})", entity_id="total", count=total)]
        facts = [count_fact("total", "All matching posts", "Posts", total, scope, [])]
        dims = analysis.group_by or (["sub_category"] if "district" in analysis.filters else ["district"])
        for dim in dims:
            for rank, row in enumerate(self.graph_store.aggregate(dim, params, min(analysis.top_n or AGG_TOP, 25)), 1):
                aggregates.append(Aggregate(name=f"{dim}: {row['name']}", entity_id=row["id"], count=row["n"]))
                facts.append(count_fact(row["id"], row["name"], dim.replace("_", " ").title().replace(" ", ""),
                                        row["n"], f"{scope}; rank {rank} by {dim}",
                                        [s["post_id"] for s in row["samples"]]))
                for sample in row["samples"]:
                    facts.append(post_fact({**sample, "predicate": f"COUNTED_IN_{dim.upper()}",
                                            "object_id": row["id"], "object": row["name"],
                                            "label": dim.replace("_", " ").title().replace(" ", "")},
                                           "aggregation"))
        return aggregates, facts

    # ---------- helpers ----------

    def _excerpt(self, chunk: Chunk) -> Chunk:
        """Header + topic + the pipeline's summary + the start of the post: the parts an answer needs."""
        body, _, summary = chunk.text.partition("\nSummary: ")
        lines = body.split("\n")
        keep = [line for line in lines[:2] if line.startswith(("[", "Topic: "))]
        text = "\n".join(lines[len(keep):]).strip()
        if len(text) > EXCERPT_TEXT_CHARS:
            text = text[:EXCERPT_TEXT_CHARS].rsplit(" ", 1)[0] + " …"
        parts = [*keep, text]
        if summary:
            parts.append("Summary: " + (summary if len(summary) <= EXCERPT_SUMMARY_CHARS
                                        else summary[:EXCERPT_SUMMARY_CHARS].rsplit(" ", 1)[0] + " …"))
        excerpt = "\n".join(p for p in parts if p)
        return chunk.model_copy(update={"text": excerpt, "token_count": self.embedder.count_tokens(excerpt)})

    def _drop_empty_filters(self, analysis: QueryAnalysis, explicit: QueryFilters | None) -> None:
        """A filter combination that matches no post would hide everything: drop inferred filters (least
        reliable first) until something matches, and record what was dropped."""
        if not (analysis.filters or analysis.time_range):
            return
        keep = set(explicit.social()) if explicit else set()
        keep_time = explicit is not None and explicit.time() is not None
        for key in DROP_ORDER:
            if self.vector_store.count(qdrant_filters(analysis.filters) or None, analysis.time_range) > 0:
                return
            if key == "time" and analysis.time_range is not None and not keep_time:
                analysis.dropped_filters.append(f"time: {analysis.time_range.expression} (no posts in that range)")
                analysis.time_range = None
            elif key in analysis.filters and key not in keep:
                analysis.dropped_filters.append(f"{key}={analysis.filters.pop(key)} (no posts match)")

    def _fact_scorer(
        self, question: str, question_vector: list[float], reranker: Reranker | None
    ) -> Callable[[list[str]], list[float]]:
        if reranker is not None:
            return lambda texts: reranker.relevance(question, texts)

        def cosine(texts: list[str]) -> list[float]:
            vectors = self.embedder.embed_documents(texts)
            return [sum(a * b for a, b in zip(v, question_vector, strict=True)) for v in vectors]

        return cosine
