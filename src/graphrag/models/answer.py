"""Grounded answer with validated citations."""

from typing import Literal

from pydantic import BaseModel, Field

from graphrag.models.retrieval import Aggregate, GraphFact, QueryAnalysis, RankedChunk, RetrievalMode


class CitedChunk(BaseModel):
    cite_id: str  # "C1"
    item: RankedChunk


class CitedFact(BaseModel):
    cite_id: str  # "G1"
    fact: GraphFact
    support: list[str] = Field(default_factory=list)  # C# ids of the fact's evidence chunks that are in the prompt


class Citation(BaseModel):
    """One citation used in the answer, with what the UI needs to show / link it."""

    cite_id: str
    kind: Literal["chunk", "fact"]
    source: str  # file name, or "knowledge graph"
    page: int | None = None
    section: str | None = None
    text: str  # chunk excerpt, or the fact as text
    evidence: str | None = None  # facts: the quote the fact was extracted from


class QueryResponse(BaseModel):
    question: str
    answer: str
    language: str
    mode: RetrievalMode
    grounded: bool  # False = no LLM call (no evidence / LLM unavailable)
    citations: list[Citation] = Field(default_factory=list)  # only those actually used, in order of use
    invalid_citations: list[str] = Field(default_factory=list)  # cited ids that weren't in the prompt (removed)
    uncited_sentences: list[str] = Field(default_factory=list)  # answer sentences without any citation
    chunks: list[CitedChunk] = Field(default_factory=list)
    facts: list[CitedFact] = Field(default_factory=list)
    aggregates: list[Aggregate] = Field(default_factory=list)
    analysis: QueryAnalysis
    timings_ms: dict[str, float] = Field(default_factory=dict)
