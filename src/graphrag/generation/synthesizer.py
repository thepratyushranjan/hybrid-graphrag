"""Grounded answer: retrieval -> (no-evidence check) -> prompt -> LLM -> citation validation."""

import asyncio
import logging
import math
import time
from typing import Literal

from graphrag.config import Settings
from graphrag.generation.citation_validator import validate
from graphrag.generation.prompts import CITATION_REMINDER, build_user_prompt, number_evidence, system_prompt
from graphrag.generation.smalltalk import SmallTalk, detect_smalltalk, smalltalk_reply
from graphrag.llm.client import LLMClient, LLMError
from graphrag.llm.registry import LLMRegistry
from graphrag.models import (
    CitedFact,
    QueryAnalysis,
    QueryFilters,
    QueryResponse,
    RetrievalMode,
    RetrievalResult,
    Subgraph,
    SubgraphEdge,
    SubgraphNode,
)
from graphrag.retrieval.hybrid import HybridRetriever
from graphrag.retrieval.social_retriever import SocialRetriever

logger = logging.getLogger(__name__)

NOT_FOUND = {
    "en": "I could not find information about this in the knowledge base.",
    "hi": "मुझे इस बारे में नॉलेज बेस में कोई जानकारी नहीं मिली।",
}
GRAPH_NO_ENTITY = (
    "Graph mode answers only from the knowledge graph, and it needs a named entity that exists in the graph "
    "(a person, organization, place, project...). None was found in your question. Name the entity, or use "
    "hybrid mode, which also searches the document text."
)
GRAPH_NO_FACTS = (
    "The knowledge graph has no relationships for {names} that answer this. Try hybrid mode, which also searches "
    "the document text."
)
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


class ProviderUnavailable(LLMError):
    """The provider picked for a question can't be used (no key, not running, unknown model)."""


class CorpusUnavailable(RuntimeError):
    """corpus="sql" was asked for but the SQL retriever isn't set up."""


Corpus = Literal["docs", "sql"]


class AnswerGenerator:
    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        llm: LLMClient | None,
        registry: LLMRegistry | None = None,
        sql_retriever: SocialRetriever | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.sql_retriever = sql_retriever
        self.llm = llm  # default provider
        self.registry = registry

    async def answer(
        self,
        question: str,
        mode: RetrievalMode = "hybrid",
        top_k: int | None = None,
        hops: int | None = None,
        filters: QueryFilters | None = None,
        rerank: bool | None = None,
        llm_provider: str | None = None,
        corpus: Corpus = "docs",
    ) -> QueryResponse:
        retriever: HybridRetriever | SocialRetriever = self.retriever
        if corpus == "sql":
            if self.sql_retriever is None:
                raise CorpusUnavailable("The SQL corpus is not available on this server")
            retriever = self.sql_retriever
        if (kind := detect_smalltalk(question)) is not None:
            return await self._smalltalk(question, kind, mode)
        llm = self.llm
        if llm_provider and self.registry is not None:
            try:
                llm = await asyncio.to_thread(self.registry.get, llm_provider)
            except LLMError as exc:
                raise ProviderUnavailable(str(exc)) from exc
        # the same provider analyses the question and writes the answer; retrieval itself is provider-independent
        result = await retriever.retrieve(question, mode, top_k, hops, filters, rerank, llm)
        chunks, facts = number_evidence(result)
        language = result.analysis.language
        base = {
            "question": question, "language": language, "mode": mode, "chunks": chunks, "graph_facts": facts,
            "subgraph": build_subgraph(facts), "aggregates": result.aggregates, "analysis": result.analysis,
            "llm": {"provider": llm.provider, "model": llm.model} if llm else None,
        }
        timings = dict(result.timings_ms)

        # No-evidence path: don't let the LLM guess
        if not has_evidence(result, self.settings.min_evidence_score):
            return QueryResponse(**base, answer=self._not_found(result, mode, language), grounded=False,
                                 timings_ms=timings)
        if llm is None:
            return QueryResponse(**base, answer=LLM_UNAVAILABLE.format(reason="no LLM configured"),
                                 grounded=False, timings_ms=timings)

        user = build_user_prompt(question, chunks, facts, result.aggregates)
        started = time.perf_counter()
        try:
            raw = await asyncio.to_thread(
                llm.complete, system_prompt(language, corpus), user, self.settings.answer_max_tokens, 0.0
            )
        except LLMError as exc:
            logger.error("Answer generation failed: %s", exc)
            return QueryResponse(**base, answer=LLM_UNAVAILABLE.format(reason=str(exc)), grounded=False,
                                 timings_ms=timings)
        timings["generate"] = round((time.perf_counter() - started) * 1000, 1)
        timings["total"] = round(timings.get("total", 0) + timings["generate"], 1)

        checked = validate(raw, chunks, facts)
        if not checked.citations and (chunks or facts):
            # Local models sometimes ignore the citation rules (seen with graph-only evidence): retry once
            retry_user = user + "\n\n" + CITATION_REMINDER.format(answer=raw)
            try:
                raw = await asyncio.to_thread(
                    llm.complete, system_prompt(language, corpus), retry_user, self.settings.answer_max_tokens, 0.0
                )
                checked = validate(raw, chunks, facts)
                timings["generate_retry"] = round((time.perf_counter() - started) * 1000 - timings["generate"], 1)
            except LLMError as exc:
                logger.warning("Citation retry failed: %s", exc)
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

    @staticmethod
    def _not_found(result: RetrievalResult, mode: RetrievalMode, language: str) -> str:
        """In graph mode, explain why the graph alone couldn't answer instead of a bare "not found"."""
        if mode == "graph" and result.analysis.query_type != "aggregation":
            if not result.analysis.entity_ids:
                return GRAPH_NO_ENTITY
            names = ", ".join(e.entity_name or e.text for e in result.analysis.entities if e.entity_id)
            return GRAPH_NO_FACTS.format(names=names)
        return NOT_FOUND.get(language, NOT_FOUND["en"])

    async def _smalltalk(self, question: str, kind: SmallTalk, mode: RetrievalMode) -> QueryResponse:
        """Answer greetings / thanks / help directly: no retrieval, no LLM, no evidence."""
        language = "hi" if any("\u0900" <= ch <= "\u097f" for ch in question) else "en"
        try:
            documents = len(await asyncio.to_thread(self.retriever.graph_store.document_sources))
        except Exception:  # noqa: BLE001 - the document count is only a nicety
            documents = None
        return QueryResponse(
            question=question,
            answer=smalltalk_reply(kind, language, documents),
            language=language,
            mode=mode,
            grounded=False,
            intent="smalltalk",
            analysis=QueryAnalysis(language=language),
        )
