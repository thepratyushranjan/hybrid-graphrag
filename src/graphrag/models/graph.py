"""Knowledge-graph models: what the LLM returns, and the resolved records written to Neo4j."""

from typing import Literal

from pydantic import BaseModel, Field

EntityType = Literal["Person", "Organization", "Location", "Event", "Concept"]
ENTITY_TYPES: tuple[EntityType, ...] = ("Person", "Organization", "Location", "Event", "Concept")

# Relationship vocabulary (always English, even for Hindi text). Anything else becomes RELATED_TO
# with the LLM's original phrase kept on the edge.
PREDICATES: tuple[str, ...] = (
    "WORKS_FOR", "LEADS", "SUBSIDIARY_OF", "OWNS", "SUPPLIES", "CONTRACTED_BY", "SUBCONTRACTS_TO",
    "PARTNER_OF", "LOCATED_IN", "IMPACTS", "PARTICIPATED_IN", "CAUSED", "OCCURRED_AT", "MEMBER_OF",
    "PRODUCES", "RELATED_TO",
)


# ---------- LLM structured output ----------

class ExtractedEntity(BaseModel):
    name: str = Field(description="Canonical English name (transliterate/translate if the text is not English)")
    type: EntityType
    aliases: list[str] = Field(default_factory=list, description="Other spellings/languages as written in the text")


class ExtractedTriple(BaseModel):
    subject: str = Field(description="Entity name exactly as in the entities list")
    predicate: str = Field(description="One of the allowed predicates")
    object: str = Field(description="Entity name exactly as in the entities list")
    evidence: str = Field(description="Short exact quote from the text that states this fact")
    confidence: float = Field(ge=0, le=1)


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    triples: list[ExtractedTriple] = Field(default_factory=list)


# ---------- resolved records for Neo4j ----------

class GraphEntity(BaseModel):
    id: str  # e.g. org:ganga_smart_power
    name: str
    type: EntityType
    aliases: list[str] = Field(default_factory=list)


class GraphRelation(BaseModel):
    subject_id: str
    predicate: str
    object_id: str
    chunk_id: str
    evidence: str
    confidence: float
    original_predicate: str | None = None  # set when an unknown predicate was mapped to RELATED_TO


class ChunkGraph(BaseModel):
    """Entities mentioned in one chunk and the relations stated in it."""

    chunk_id: str
    entities: list[GraphEntity] = Field(default_factory=list)
    relations: list[GraphRelation] = Field(default_factory=list)
    dropped: int = 0  # triples rejected by the evidence/confidence/whitelist checks


class SimilarDocument(BaseModel):
    doc_id: str
    source: str
    score: float
