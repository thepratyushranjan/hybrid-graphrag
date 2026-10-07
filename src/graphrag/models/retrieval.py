"""Query analysis, graph facts and hybrid retrieval results."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from graphrag.models.documents import Chunk
from graphrag.models.graph import EntityType

QueryType = Literal["lookup", "relationship", "aggregation"]
RetrievalMode = Literal["vector", "graph", "hybrid"]


# ---------- LLM structured output for query analysis ----------

class QueryEntityLLM(BaseModel):
    name: str = Field(description="The entity exactly as written in the question")
    english_name: str = Field(description="Canonical English name (transliterate/translate if needed)")
    type: EntityType | None = None


class QueryAnalysisLLM(BaseModel):
    entities: list[QueryEntityLLM] = Field(default_factory=list)
    query_type: QueryType = Field(
        default="lookup",
        description="lookup = facts about something; relationship = how things are connected / multi-hop; "
        "aggregation = counting, ranking, 'which ... the most'",
    )
    predicate: str | None = Field(default=None, description="For aggregation/relationship: the relation asked about")
    target_type: EntityType | None = Field(default=None, description="For aggregation: the type being counted/ranked")
    document: str | None = Field(
        default=None, description="Only if the question restricts itself to one document: that document's name"
    )
    doc_type: Literal["pdf", "md", "txt"] | None = Field(
        default=None, description="Only if the question restricts itself to a file type (PDFs, Markdown, text)"
    )


# ---------- resolved analysis ----------

class TimeFilter(BaseModel):
    start: date | None = None  # None = open-ended
    end: date | None = None
    expression: str  # the words in the question, e.g. "since April 2026", "पिछले 3 दिन"


class QueryEntity(BaseModel):
    text: str  # as written in the question
    entity_id: str | None = None  # resolved graph node, if any
    entity_name: str | None = None
    match_score: float | None = None  # full-text score of the resolution
    origin: Literal["question", "vector"] = "question"  # vector = seeded from top vector chunks (hybrid)


class QueryAnalysis(BaseModel):
    language: str
    query_type: QueryType = "lookup"
    entities: list[QueryEntity] = Field(default_factory=list)
    predicate: str | None = None
    target_type: EntityType | None = None
    analyzer: Literal["llm", "fulltext"] = "llm"  # fulltext = LLM unavailable, matched query words directly
    # Payload filters applied to Qdrant (and to graph facts via their chunks), e.g. {"source": "x.pdf"}.
    # Only set when the question names a document / file type that really exists.
    filters: dict[str, str] = Field(default_factory=dict)
    time_range: TimeFilter | None = None  # from a time expression in the question
    dropped_filters: list[str] = Field(default_factory=list)  # filters removed because they matched nothing

    @property
    def entity_ids(self) -> list[str]:
        return list(dict.fromkeys(e.entity_id for e in self.entities if e.entity_id))


# ---------- retrieval results ----------

class GraphFact(BaseModel):
    subject_id: str
    subject: str
    predicate: str
    object_id: str
    object: str
    hops: int  # distance from a query entity (1 = directly connected)
    confidence: float
    relevance: float | None = None  # 0-1 relevance to the question (cross-encoder, or embedding cosine)
    chunk_ids: list[str]  # supporting chunks (evidence)
    evidence: str
    template: str  # which Cypher template found it

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.subject_id, self.predicate, self.object_id)

    def as_text(self) -> str:
        return f"{self.subject} -[{self.predicate}]-> {self.object}"

    def as_sentence(self) -> str:
        """Natural-language form used for relevance scoring: "Vikram Singh leads Ganga Roadways"."""
        return f"{self.subject} {self.predicate.lower().replace('_', ' ')} {self.object}. {self.evidence}"


class Aggregate(BaseModel):
    name: str
    entity_id: str
    count: int


class RankedChunk(BaseModel):
    chunk: Chunk
    score: float  # final ranking score (rerank score if reranked, else fused score)
    vector_rank: int | None = None  # 1-based rank in the vector list
    vector_score: float | None = None
    graph_rank: int | None = None  # 1-based rank among graph-linked chunks
    fused_score: float = 0.0
    rerank_score: float | None = None
    entity_mentions: int = 0  # query entities this chunk mentions (graph)
    found_by: list[Literal["vector", "graph"]] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    question: str
    mode: RetrievalMode
    analysis: QueryAnalysis
    chunks: list[RankedChunk]
    facts: list[GraphFact]
    aggregates: list[Aggregate] = Field(default_factory=list)
    templates: list[str] = Field(default_factory=list)  # Cypher templates that ran
    context_tokens: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)
