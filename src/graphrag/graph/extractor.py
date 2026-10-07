"""LLM triple extraction with entity resolution and hallucination checks.

For each chunk the LLM returns entities (canonical English name + aliases as written) and
(subject, predicate, object) triples with an evidence quote. A triple is kept only if:
  - its subject and object resolve to entities,
  - its evidence quote really appears in the chunk text,
  - its confidence is at or above EXTRACTION_MIN_CONFIDENCE.
Unknown predicates become RELATED_TO with the original phrase kept.
"""

import logging
import re
import unicodedata

from graphrag.config import Settings
from graphrag.llm.cache import JsonCache
from graphrag.llm.client import LLMClient
from graphrag.models import (
    ENTITY_TYPES,
    PREDICATES,
    Chunk,
    ChunkGraph,
    EntityType,
    ExtractionResult,
    GraphEntity,
    GraphRelation,
)

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v4"  # bump when the prompt changes, so cached extractions are refreshed

_ID_PREFIX: dict[EntityType, str] = {
    "Person": "person", "Organization": "org", "Location": "loc", "Event": "event", "Concept": "concept",
}
_ORG_SUFFIXES = {"ltd", "limited", "pvt", "private", "inc", "llp", "llc", "corp", "corporation", "co", "plc"}
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
_PARENTHETICAL = re.compile(r"\s*\([^)]*\)\s*$")
_SPACES = re.compile(r"\s+")
_DASHES_QUOTES = str.maketrans({"–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"'})

SYSTEM_PROMPT = """You build a knowledge graph from one chunk of a document. The text may be in any language \
(English, Hindi or mixed).

Return JSON with:
- "entities": every Person, Organization, Location, Event or Concept that takes part in a relation.
  * "name": the canonical ENGLISH name. Transliterate or translate non-English names \
(e.g. "शक्ति स्टील वर्क्स" -> "Shakti Steel Works"). If a known entity below is meant, copy its name exactly.
  * "type": Person | Organization | Location | Event | Concept (Concept = rules, clauses, products, projects, \
risks, programmes).
  * "aliases": other spellings exactly as they appear in the text (including non-English forms).
- "triples": facts stated in the text, as subject -> predicate -> object, using entity names from your list.
  * "predicate": one of {predicates}. Pick the closest; use RELATED_TO only if none fits.
  * "evidence": a SHORT EXACT quote (copied character for character) from the text that states the fact.
  * "confidence": 0-1, how clearly the text states it.

Rules:
- Every subject and object of a triple MUST also be listed in "entities".
- Implicit subject: many documents leave the subject out (resume / CV bullets like "Built X using Y", \
"Skills: Python, SQL", reports written by one organization). If the document context below shows the document \
is about ONE main person or organization, use it as the subject of such lines and list it in "entities". \
For lists like "Languages: Python, SQL" create one triple per item (Person HAS_SKILL Python, ...); \
the evidence is the list line.
- Document structure is a statement too. An employer heading with a job title and dates (CV "Experience") \
means main subject WORKS_FOR that organization (evidence: the organization name). A project, product or \
achievement listed under that heading means main subject PARTICIPATED_IN it and the organization PRODUCES it. \
An education entry means main subject STUDIED_AT the institution.
- Prefer specific named entities over generic nouns: if the text names the vendors a clause impacts, \
link the clause to each named vendor, not to the word "vendor".
- Extract only what the text states, never outside knowledge.
- Directions matter: "A supplies goods or services to B" / "A hosts data for B" -> A SUPPLIES B; \
"X is a subsidiary of Y" -> X SUBSIDIARY_OF Y; "B contracted vendor A" -> A CONTRACTED_BY B; \
"clause C applies to / impacts V" -> C IMPACTS V; "P is MD/CEO/head of O" -> P LEADS O; \
"O is based in L" -> O LOCATED_IN L; "P worked at O" -> P WORKS_FOR O; "P knows / is skilled in T" -> \
P HAS_SKILL T; "project X uses / is built with T" -> X USES T; "P studied at U" -> P STUDIED_AT U."""


def _fold(text: str) -> str:
    """Normalise for comparisons: NFKC, casefold, unify dashes/quotes, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_DASHES_QUOTES).casefold()
    return _SPACES.sub(" ", text).strip()


def _name_key(name: str, entity_type: EntityType) -> str:
    """Key used for matching and IDs: folded, punctuation-free; company suffixes (Ltd, Pvt...) dropped."""
    words = [w for w in _NON_WORD.split(_fold(name)) if w]
    if entity_type == "Organization":
        while len(words) > 1 and words[-1] in _ORG_SUFFIXES:
            words.pop()
    return "_".join(words)


def entity_id(name: str, entity_type: EntityType) -> str:
    return f"{_ID_PREFIX[entity_type]}:{_name_key(name, entity_type)}"


# Predicates whose subject must be a person / whose object must be a place. LLMs sometimes flip them
# ("Ganga Roadways LEADS Vikram Singh"); the entity types tell us which way round the fact must be.
_PERSON_SUBJECT = {"LEADS", "WORKS_FOR", "HAS_SKILL", "STUDIED_AT"}
_LOCATION_OBJECT = {"LOCATED_IN", "OCCURRED_AT"}


def orient(predicate: str, subject: GraphEntity, obj: GraphEntity) -> tuple[GraphEntity, GraphEntity]:
    """Swap subject and object when the entity types show the LLM wrote the fact backwards."""
    if predicate in _PERSON_SUBJECT and subject.type != "Person" and obj.type == "Person":
        return obj, subject
    if predicate in _LOCATION_OBJECT and obj.type != "Location" and subject.type == "Location":
        return obj, subject
    return subject, obj


def normalize_predicate(raw: str) -> tuple[str, str | None]:
    """Map the LLM's predicate onto the whitelist. Returns (predicate, original phrase if it was remapped)."""
    candidate = re.sub(r"[^A-Z0-9]+", "_", raw.upper()).strip("_")
    if candidate in PREDICATES:
        return candidate, None
    return "RELATED_TO", raw


class EntityResolver:
    """Maps names and aliases (any language or spelling) to one entity ID, so a Hindi mention and its
    English name end up on the same node. Seeded with entities already in Neo4j."""

    def __init__(self, known: list[GraphEntity] | None = None) -> None:
        self.entities: dict[str, GraphEntity] = {}
        self._by_key: dict[str, str] = {}  # name/alias key (type-independent) -> entity id
        for entity in known or []:
            self.add(entity)

    def _keys(self, entity: GraphEntity) -> set[str]:
        return {k for n in [entity.name, *entity.aliases] if (k := _name_key(n, entity.type))}

    def add(self, entity: GraphEntity) -> GraphEntity:
        existing = self.entities.get(entity.id)
        if existing:
            merged_aliases = [a for a in [*existing.aliases, *entity.aliases] if a != existing.name]
            entity = existing.model_copy(update={"aliases": list(dict.fromkeys(merged_aliases))})
        self.entities[entity.id] = entity
        for key in self._keys(entity):
            self._by_key.setdefault(key, entity.id)
        return entity

    def lookup(self, name: str) -> GraphEntity | None:
        for entity_type in ENTITY_TYPES:  # try every type's key rules (Org suffix stripping differs)
            entity_id_ = self._by_key.get(_name_key(name, entity_type))
            if entity_id_:
                return self.entities[entity_id_]
        return None

    def resolve(self, name: str, entity_type: EntityType, aliases: list[str]) -> GraphEntity:
        """Reuse an existing entity if the name or any alias matches one, otherwise create it."""
        for candidate in [name, *aliases]:
            match = self.lookup(candidate)
            if match and match.type == entity_type:
                return self.add(match.model_copy(update={"aliases": [*match.aliases, name, *aliases]}))
        return self.add(GraphEntity(id=entity_id(name, entity_type), name=name, type=entity_type, aliases=aliases))

    def prompt_list(self, limit: int) -> str:
        items = list(self.entities.values())[:limit]
        return "\n".join(f"- {e.name} ({e.type})" for e in items) or "(none yet)"


class TripleExtractor:
    def __init__(self, settings: Settings, llm: LLMClient, cache: JsonCache) -> None:
        self.settings = settings
        self.llm = llm
        self.cache = cache
        self.system_prompt = SYSTEM_PROMPT.format(predicates=", ".join(PREDICATES))

    def _call_llm(self, chunk: Chunk, resolver: EntityResolver, context: str) -> ExtractionResult:
        # Cache by chunk text + document context + model + prompt version: re-ingestion never re-calls the LLM
        effort = getattr(self.llm, "extraction_effort", None) or ""
        key = JsonCache.key(PROMPT_VERSION, self.llm.provider, self.llm.model, effort, context, chunk.text)
        cached = self.cache.get(key)
        if cached is not None:
            return ExtractionResult.model_validate(cached)
        user = (
            f"Document context (to identify an implicit main subject; extract facts ONLY from the Text):\n"
            f"{context}\n\n"
            f"Known entities (reuse these exact names when the text refers to them):\n"
            f"{resolver.prompt_list(self.settings.known_entities_limit)}\n\n"
            f"Text:\n<<<\n{chunk.text}\n>>>"
        )
        result = self.llm.complete_json(
            self.system_prompt, user, ExtractionResult, reasoning_effort=effort or None
        )
        self.cache.set(key, result.model_dump())
        return result

    @staticmethod
    def _endpoint(
        name: str, mentioned: dict[str, GraphEntity], haystack: str, resolver: EntityResolver
    ) -> GraphEntity | None:
        """Resolve a triple endpoint. It must be an entity the LLM listed for this chunk, or a proper name
        (capitalised, non-Latin script or containing a digit) that literally appears in the chunk text,
        which covers entities the LLM used but forgot to list. Entities already in the graph only decide
        which node a name maps to, never whether a fact is kept, so results don't depend on ingest order."""
        name = name.strip()
        # "Roboi.ai (Edge AI Surveillance)" in a triple usually means the listed entity "Roboi.ai"
        bare = _PARENTHETICAL.sub("", name).strip()
        match = resolver.lookup(name) or (resolver.lookup(bare) if bare and bare != name else None)
        if match and match.id in mentioned:
            return match
        if len(name) < 3 or _fold(name) not in haystack:
            return None
        first = name[0]
        if not (first.isupper() or not first.isascii() or any(ch.isdigit() for ch in name)):
            return None  # generic nouns like "vendor"
        return match or resolver.resolve(name, "Concept", [])

    def extract(self, chunk: Chunk, resolver: EntityResolver, context: str = "") -> ChunkGraph:
        result = self._call_llm(chunk, resolver, context)
        haystack = _fold(chunk.text)

        mentioned: dict[str, GraphEntity] = {}
        for e in result.entities:
            if not e.name.strip():
                continue
            aliases = [a for a in dict.fromkeys(a.strip() for a in e.aliases) if a and a != e.name]
            entity = resolver.resolve(e.name.strip(), e.type, aliases)
            mentioned[entity.id] = entity

        relations: list[GraphRelation] = []
        dropped = 0
        for t in result.triples:
            evidence = t.evidence.strip()
            subject = self._endpoint(t.subject, mentioned, haystack, resolver)
            obj = self._endpoint(t.object, mentioned, haystack, resolver)
            if (
                subject is None
                or obj is None
                or subject.id == obj.id
                or t.confidence < self.settings.extraction_min_confidence
                or len(evidence) < 4
                or _fold(evidence) not in haystack  # cheap hallucination check
            ):
                dropped += 1
                logger.debug("dropped triple %s -[%s]-> %s (evidence: %r)", t.subject, t.predicate, t.object, evidence)
                continue
            predicate, original = normalize_predicate(t.predicate)
            subject, obj = orient(predicate, subject, obj)
            if _name_key(subject.name, subject.type) == _name_key(obj.name, obj.type):
                dropped += 1  # same thing under two types ("Ganga Roadways PARTICIPATED_IN Ganga Roadways")
                continue
            relations.append(
                GraphRelation(
                    subject_id=subject.id,
                    predicate=predicate,
                    object_id=obj.id,
                    chunk_id=chunk.chunk_id,
                    evidence=evidence,
                    confidence=t.confidence,
                    original_predicate=original,
                )
            )
            mentioned.setdefault(subject.id, subject)
            mentioned.setdefault(obj.id, obj)

        # final alias state for each mentioned entity (resolution may have merged more aliases)
        entities = [resolver.entities[eid] for eid in mentioned]
        return ChunkGraph(chunk_id=chunk.chunk_id, entities=entities, relations=relations, dropped=dropped)


def document_context(source: str, first_chunk_text: str, chars: int = 300) -> str:
    """File name + the start of the document (usually its title / author / subject line)."""
    start = " ".join(first_chunk_text.split())[:chars]
    return f"File: {source}\nDocument starts with: {start}"
