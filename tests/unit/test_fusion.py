from graphrag.models import Chunk, GraphFact, QueryAnalysis, QueryEntity
from graphrag.retrieval.fusion import dedupe_facts, fit_budget, graph_chunk_order, rank_facts, reciprocal_rank_fusion
from graphrag.retrieval.graph_retriever import choose_templates


def _chunk(cid: str, tokens: int = 100) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d", source="s", doc_type="txt", chunk_index=0, text=cid, token_count=tokens)


def _fact(s: str, p: str, o: str, chunks: list[str], hops: int = 1, conf: float = 0.9) -> GraphFact:
    return GraphFact(subject_id=s, subject=s, predicate=p, object_id=o, object=o, hops=hops, confidence=conf,
                     chunk_ids=chunks, evidence=f"{s} {p} {o}", template="neighbourhood")


def test_rrf_rewards_chunks_found_by_both_branches() -> None:
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    fused = reciprocal_rank_fusion([(a, 0.9), (b, 0.8)], [c, b], mentions={}, k=60, mention_boost=0)
    assert [x.chunk.chunk_id for x in fused] == ["b", "a", "c"]  # b: 1/62 + 1/62 beats a: 1/61
    assert fused[0].found_by == ["vector", "graph"]


def test_entity_mention_boost_can_reorder() -> None:
    a, b = _chunk("a"), _chunk("b")
    fused = reciprocal_rank_fusion([(a, 0.9), (b, 0.8)], [], mentions={"b": 2}, k=60, mention_boost=0.01)
    assert fused[0].chunk.chunk_id == "b" and fused[0].entity_mentions == 2


def test_facts_are_deduped_with_merged_support() -> None:
    merged = dedupe_facts([_fact("x", "SUPPLIES", "y", ["c1"], hops=2), _fact("x", "SUPPLIES", "y", ["c2"], hops=1)])
    assert len(merged) == 1 and merged[0].chunk_ids == ["c1", "c2"] and merged[0].hops == 1


def test_rank_facts_by_relevance_with_hop_penalty() -> None:
    near = _fact("a", "LEADS", "b", ["c1"], hops=1)
    far = _fact("c", "LEADS", "d", ["c2"], hops=2)
    # far is slightly more relevant, but not by more than the hop penalty
    ranked = rank_facts([far, near], score=lambda texts: [0.53 if t.startswith("c") else 0.50 for t in texts])
    assert [f.subject for f in ranked] == ["a", "c"]


def test_graph_chunk_order_follows_best_fact() -> None:
    facts = [_fact("a", "P", "b", ["c2"]), _fact("c", "P", "d", ["c1", "c2"])]
    assert graph_chunk_order(facts) == ["c2", "c1"]


def test_budget_cuts_lowest_ranked_chunks() -> None:
    from graphrag.models import RankedChunk

    chunks = [RankedChunk(chunk=_chunk(f"c{i}", tokens=400), score=1.0) for i in range(5)]
    kept, facts, used = fit_budget(chunks, [], budget=1000, count_tokens=len)
    assert [c.chunk.chunk_id for c in kept] == ["c0", "c1"] and used <= 1000


def test_template_choice() -> None:
    one = QueryAnalysis(language="en", entities=[QueryEntity(text="x", entity_id="org:x")])
    two = QueryAnalysis(language="en", query_type="relationship",
                        entities=[QueryEntity(text="x", entity_id="org:x"), QueryEntity(text="y", entity_id="org:y")])
    agg = QueryAnalysis(language="en", query_type="aggregation")
    assert choose_templates(one, hops=1) == ["neighbourhood"]
    assert choose_templates(one, hops=2) == ["neighbourhood", "two_hop"]
    assert choose_templates(two, hops=1) == ["neighbourhood", "two_hop", "path", "intersection"]
    assert choose_templates(agg, hops=1) == ["aggregation"]
