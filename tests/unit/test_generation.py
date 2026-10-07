"""Prompt building, citation validation and the no-evidence decision (no LLM needed)."""

from graphrag.generation.citation_validator import validate
from graphrag.generation.prompts import build_user_prompt, number_evidence, system_prompt
from graphrag.generation.synthesizer import has_evidence
from graphrag.models import Chunk, GraphFact, QueryAnalysis, RankedChunk, RetrievalResult


def _result(rerank: float | None = 2.0, relevance: float | None = 0.9) -> RetrievalResult:
    chunk = Chunk(chunk_id="chunk:1", doc_id="d", source="bulletin.pdf", doc_type="pdf", chunk_index=0, page=2,
                  text="Ganga Roadways placed Shakti Steel Works on probation.", token_count=12,
                  dates=["14 October 2025", "20 October 2025"])
    fact = GraphFact(subject_id="person:v", subject="Vikram Singh", predicate="LEADS", object_id="org:g",
                     object="Ganga Roadways", subject_type="Person", object_type="Organization", hops=1, confidence=1.0, chunk_ids=["chunk:1", "chunk:other"],
                     evidence="Managing Director: Vikram Singh", template="neighbourhood", relevance=relevance)
    return RetrievalResult(question="q", mode="hybrid", analysis=QueryAnalysis(language="en"),
                           chunks=[RankedChunk(chunk=chunk, score=1.0, rerank_score=rerank)], facts=[fact])


def test_prompt_numbers_evidence_and_links_fact_support() -> None:
    chunks, facts = number_evidence(_result())
    assert [c.cite_id for c in chunks] == ["C1"] and [f.cite_id for f in facts] == ["G1"]
    assert facts[0].support == ["C1"]  # only supporting chunks that are in the prompt
    prompt = build_user_prompt("Who leads it?", chunks, facts, [])
    assert "[C1] (doc: bulletin.pdf | page 2 | dates: 14 October 2025, 20 October 2025)" in prompt
    assert (
        '[G1] (Person: Vikram Singh) -[LEADS]-> (Organization: Ganga Roadways)  src: C1, conf 1.00 — '
        '"Managing Director: Vikram Singh"'
    ) in prompt
    assert prompt.rstrip().endswith("QUESTION: Who leads it?")
    assert "Answer in Hindi" in system_prompt("hi")


def test_validator_removes_invented_ids_and_flags_uncited() -> None:
    chunks, facts = number_evidence(_result())
    answer = (
        "Vikram Singh leads Ganga Roadways [G1]. Shakti Steel Works was put on probation. [C1, C7] "
        "It is also the largest steel company in India.\n**Details:**"
    )
    checked = validate(answer, chunks, facts)
    assert checked.invalid == ["C7"]
    assert "[C7]" not in checked.answer and "[C1]" in checked.answer
    assert [c.cite_id for c in checked.citations] == ["G1", "C1"]
    assert checked.citations[0].kind == "fact" and checked.citations[1].page == 2
    # citation after the full stop still counts; the heading isn't flagged; the unsupported claim is
    assert checked.uncited_sentences == ["It is also the largest steel company in India."]


def test_no_evidence_decision() -> None:
    assert has_evidence(_result(), min_score=0.05)
    assert not has_evidence(_result(rerank=-8.0, relevance=0.001), min_score=0.05)
    empty = _result()
    empty.chunks, empty.facts = [], []
    assert not has_evidence(empty, min_score=0.05)
