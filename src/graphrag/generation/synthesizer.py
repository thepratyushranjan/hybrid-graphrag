"""Grounded answer: retrieval -> (no-evidence check) -> prompt -> LLM -> citation validation."""

import asyncio
import logging
import math
import time

from graphrag.config import Settings
from graphrag.generation.citation_validator import validate
from graphrag.generation.prompts import build_user_prompt, number_evidence, system_prompt
from graphrag.llm.client import LLMClient, LLMError
from graphrag.models import (
    CitedFact,
    QueryFilters,
    QueryResponse,
    RetrievalMode,
    RetrievalResult,
    Subgraph,
    SubgraphEdge,
    SubgraphNode,
)
from graphrag.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

NOT_FOUND = {
    "en": "I could not find information about this in the knowledge base.",
    "hi": "मुझे इस बारे में नॉलेज बेस में कोई जानकारी नहीं मिली।",
}
LLM_UNAVAILABLE = "The answer could not be generated ({reason}). The retrieved evidence is shown below."


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def has_evidence(result: RetrievalResult, min_score: float) -> bool:
    """Anything relevant enough to answer from? Uses the cross-encoder scores when available."""
    if result.aggregates:
        return True
    chunk_scores = [_sigmoid(c.rerank_score) for c in result.chunks if c.rerank_score is not None]
    fact_scores = [f.relevance for f in result.facts if f.relevance is not None]
    if not chunk_scores and not fact_scores:  # no reranker: any evidence at all
        return bool(result.chunks or result.facts)
    return max([*chunk_scores, *fact_scores], default=0.0) >= min_score


def build_subgraph(facts: list[CitedFact]) -> Subgraph:
    nodes: dict[str, SubgraphNode] = {}
    edges: list[SubgraphEdge] = []
    for cited in facts:
        f = cited.fact
        nodes.setdefault(f.subject_id, SubgraphNode(id=f.subject_id, name=f.subject, type=f.subject_type))
        nodes.setdefault(f.object_id, SubgraphNode(id=f.object_id, name=f.object, type=f.object_type))
        edges.append(SubgraphEdge(source=f.subject_id, target=f.object_id, predicate=f.predicate,
                                  cite_id=cited.cite_id, confidence=f.confidence))
    return Subgraph(nodes=list(nodes.values()), edges=edges)


class AnswerGenerator:
    def __init__(self, settings: Settings, retriever: HybridRetriever, llm: LLMClient | None) -> None:
        self.settings = settings
        self.retriever = retriever
        self.llm = llm

    async def answer(
        self,
        question: str,
        mode: RetrievalMode = "hybrid",
        top_k: int | None = None,
        hops: int | None = None,
        filters: QueryFilters | None = None,
    ) -> QueryResponse:
        result = await self.retriever.retrieve(question, mode, top_k, hops, filters)
        chunks, facts = number_evidence(result)
        language = result.analysis.language
        base = {
            "question": question, "language": language, "mode": mode, "chunks": chunks, "graph_facts": facts,
            "subgraph": build_subgraph(facts), "aggregates": result.aggregates, "analysis": result.analysis,
        }
        timings = dict(result.timings_ms)

        # No-evidence path: don't let the LLM guess
        if not has_evidence(result, self.settings.min_evidence_score):
            return QueryResponse(**base, answer=NOT_FOUND.get(language, NOT_FOUND["en"]), grounded=False,
                                 timings_ms=timings)
        if self.llm is None:
            return QueryResponse(**base, answer=LLM_UNAVAILABLE.format(reason="no LLM configured"),
                                 grounded=False, timings_ms=timings)

        user = build_user_prompt(question, chunks, facts, result.aggregates)
        started = time.perf_counter()
        try:
            raw = await asyncio.to_thread(
                self.llm.complete, system_prompt(language), user, self.settings.answer_max_tokens, 0.0
            )
        except LLMError as exc:
            logger.error("Answer generation failed: %s", exc)
            return QueryResponse(**base, answer=LLM_UNAVAILABLE.format(reason=str(exc)), grounded=False,
                                 timings_ms=timings)
        timings["generate"] = round((time.perf_counter() - started) * 1000, 1)
        timings["total"] = round(timings.get("total", 0) + timings["generate"], 1)

        checked = validate(raw, chunks, facts)
        if checked.invalid:
            logger.warning("Removed citations not in the prompt: %s", checked.invalid)
        return QueryResponse(
            **base,
            answer=checked.answer,
            grounded=True,
            citations=checked.citations,
            invalid_citations=checked.invalid,
            uncited_sentences=checked.uncited_sentences,
            timings_ms=timings,
        )
