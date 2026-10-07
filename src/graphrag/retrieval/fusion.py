"""Step 4: fuse vector hits and graph-linked chunks, rank graph facts, fit a token budget."""

from collections.abc import Callable

from graphrag.models import Chunk, GraphFact, RankedChunk

HOP_PENALTY = 0.05  # relevance (0-1) subtracted per extra hop


def dedupe_facts(facts: list[GraphFact]) -> list[GraphFact]:
    """Merge facts found by several templates / chunks (union of supporting chunks, best hops/confidence)."""
    merged: dict[tuple[str, str, str], GraphFact] = {}
    for fact in facts:
        if fact.key in merged:
            kept = merged[fact.key]
            kept.chunk_ids = list(dict.fromkeys([*kept.chunk_ids, *fact.chunk_ids]))
            kept.sources = list(dict.fromkeys([*kept.sources, *fact.sources]))
            kept.hops = min(kept.hops, fact.hops)
            kept.confidence = max(kept.confidence, fact.confidence)
        else:
            merged[fact.key] = fact.model_copy(deep=True)
    return list(merged.values())


def rank_facts(facts: list[GraphFact], score: Callable[[list[str]], list[float]] | None) -> list[GraphFact]:
    """Rank by relevance to the question (0-1, from `score`) minus a small per-hop penalty; ties broken by
    confidence and support count. Without a scorer: hops, confidence, support."""
    facts = dedupe_facts(facts)
    if score is not None and facts:
        for fact, relevance in zip(facts, score([f.as_sentence() for f in facts]), strict=True):
            fact.relevance = round(float(relevance), 4)
        return sorted(
            facts,
            key=lambda f: (-((f.relevance or 0) - HOP_PENALTY * (f.hops - 1)), -f.confidence, -len(f.chunk_ids)),
        )
    return sorted(facts, key=lambda f: (f.hops, -f.confidence, -len(f.chunk_ids), f.as_text()))


def graph_chunk_order(facts: list[GraphFact]) -> list[str]:
    """Chunks ranked by the best fact they support, then by how many facts they support."""
    first_seen: dict[str, int] = {}
    support: dict[str, int] = {}
    for rank, fact in enumerate(facts):
        for chunk_id in fact.chunk_ids:
            first_seen.setdefault(chunk_id, rank)
            support[chunk_id] = support.get(chunk_id, 0) + 1
    return sorted(first_seen, key=lambda c: (first_seen[c], -support[c]))


def reciprocal_rank_fusion(
    vector_hits: list[tuple[Chunk, float]],
    graph_chunks: list[Chunk],
    mentions: dict[str, int],
    k: int,
    mention_boost: float,
) -> list[RankedChunk]:
    """RRF: score = sum over lists of 1 / (k + rank), plus a boost per query entity the chunk mentions."""
    pool: dict[str, RankedChunk] = {}
    for rank, (chunk, score) in enumerate(vector_hits, start=1):
        pool[chunk.chunk_id] = RankedChunk(
            chunk=chunk, score=0.0, vector_rank=rank, vector_score=round(score, 4), found_by=["vector"]
        )
    for rank, chunk in enumerate(graph_chunks, start=1):
        item = pool.setdefault(chunk.chunk_id, RankedChunk(chunk=chunk, score=0.0))
        item.graph_rank = rank
        item.found_by.append("graph")

    for item in pool.values():
        fused = sum(1 / (k + r) for r in (item.vector_rank, item.graph_rank) if r is not None)
        item.entity_mentions = mentions.get(item.chunk.chunk_id, 0)
        item.fused_score = round(fused + mention_boost * item.entity_mentions, 6)
        item.score = item.fused_score
    return sorted(pool.values(), key=lambda c: -c.fused_score)


def fit_budget(
    chunks: list[RankedChunk], facts: list[GraphFact], budget: int, count_tokens: Callable[[str], int]
) -> tuple[list[RankedChunk], list[GraphFact], int]:
    """Keep the best items within `budget` tokens, cutting the lowest-ranked first.
    Facts are cheap and dense, so they get up to a quarter of the budget first."""
    used = 0
    kept_facts: list[GraphFact] = []
    for fact in facts:
        cost = count_tokens(fact.as_text()) + 4
        if used + cost > budget // 4:
            break
        kept_facts.append(fact)
        used += cost
    kept_chunks: list[RankedChunk] = []
    for item in chunks:
        cost = item.chunk.token_count + 12  # + citation header
        if used + cost > budget:
            continue  # a smaller lower-ranked chunk may still fit
        kept_chunks.append(item)
        used += cost
    return kept_chunks, kept_facts, used
