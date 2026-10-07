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


class SubgraphNode(BaseModel):
    id: str
    name: str
    type: str | None = None


class SubgraphEdge(BaseModel):
    source: str
    target: str
    predicate: str
    cite_id: str
    confidence: float


class Subgraph(BaseModel):
    """The graph facts behind the answer, as nodes + edges for a graph view."""

    nodes: list[SubgraphNode] = Field(default_factory=list)
    edges: list[SubgraphEdge] = Field(default_factory=list)


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
    grounded: bool  # False = no LLM call (small talk / no evidence / LLM unavailable)
    intent: Literal["question", "smalltalk"] = "question"
    llm: dict[str, str] | None = None  # {"provider", "model"} that analysed the question and wrote the answer
    citations: list[Citation] = Field(default_factory=list)  # only those actually used, in order of use
    invalid_citations: list[str] = Field(default_factory=list)  # cited ids that weren't in the prompt (removed)
    uncited_sentences: list[str] = Field(default_factory=list)  # answer sentences without any citation
    chunks: list[CitedChunk] = Field(default_factory=list)
    graph_facts: list[CitedFact] = Field(default_factory=list)
    subgraph: Subgraph = Field(default_factory=Subgraph)
    aggregates: list[Aggregate] = Field(default_factory=list)
    analysis: QueryAnalysis
    timings_ms: dict[str, float] = Field(default_factory=dict)
