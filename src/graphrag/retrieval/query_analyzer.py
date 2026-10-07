"""Step 1: understand the question — entities (resolved to graph nodes), query type and language."""

import logging
import re
from datetime import date

from graphrag.config import Settings
from graphrag.graph.extractor import _name_key, normalize_predicate
from graphrag.graph.neo4j_store import Neo4jStore
from graphrag.ingestion.chunker import detect_language
from graphrag.ingestion.dates import parse_time_range
from graphrag.llm.cache import JsonCache
from graphrag.llm.client import LLMClient, LLMError
from graphrag.models import ENTITY_TYPES, PREDICATES, QueryAnalysis, QueryAnalysisLLM, QueryEntity, TimeFilter

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v3"
_LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')
_WORD = re.compile(r"[\w.]+", re.UNICODE)
# words too common to identify an entity on their own (fallback matching only)
_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was", "were", "which", "what",
    "who", "whom", "how", "many", "much", "does", "do", "did", "with", "by", "from", "about", "that", "this",
    "most", "all", "any", "their", "its", "be", "has", "have", "tell", "me", "connected", "related",
    "के", "की", "का", "को", "में", "से", "पर", "है", "हैं", "और", "कौन", "क्या", "कैसे", "किस", "ने",
}

SYSTEM_PROMPT = f"""You analyse a question that will be answered from a knowledge graph and documents. \
The question may be in any language.

Return JSON with:
- "entities": the specific named things in the question (companies, people, places, clauses, projects, \
programmes...). "name" exactly as written; "english_name" = canonical English name (transliterate/translate \
non-English names); "type" = one of {", ".join(ENTITY_TYPES)}. Do not list generic words like "vendors" or "risks".
- "query_type": "lookup" (facts about something), "relationship" (how things are connected, multi-hop, \
"which X connected to Y ..."), or "aggregation" (counting, ranking, "which ... the most").
- "predicate": the relation the question asks about, as the closest of: {", ".join(PREDICATES)}. \
Always fill it when the question uses a relation verb ("supplies" -> SUPPLIES, "leads"/"heads" -> LEADS, \
"impacts"/"applies to" -> IMPACTS, "located in" -> LOCATED_IN); null only if no relation is named.
- "target_type": for aggregation, the type of the thing being counted or ranked (the answer's type); else null.

- "document": ONLY if the question explicitly limits itself to one document ("in the compliance bulletin", \
"according to the annual report"), that document's name as written; else null.
- "doc_type": ONLY if the question explicitly limits itself to a file type ("in the PDFs" -> "pdf", \
"markdown files" -> "md", "text files" -> "txt"); else null.

Example: "Which organization supplies the most subsidiaries?" -> query_type "aggregation", \
predicate "SUPPLIES", target_type "Organization", entities [], document null, doc_type null."""


_DOC_TYPE_WORDS = {
    "pdf": {"pdf", "pdfs"},
    "md": {"markdown", "md"},
    "txt": {"txt", "text"},
}
_FILE_WORDS = {"document", "doc", "file", "pdf", "md", "txt", "the", "a", "an", "of", "in", "from"}


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[\W_]+", text.casefold()) if t and t not in _FILE_WORDS}  # "_" splits file names


def match_document(name: str, sources: list[str]) -> str | None:
    """Best source file for a document name from the question ("compliance bulletin" ->
    vendor_compliance_bulletin_q3_2025.pdf). Needs at least half of the name's words in the file name."""
    wanted = _tokens(name)
    if not wanted:
        return None
    scored = [(len(wanted & _tokens(src)) / len(wanted), src) for src in sources]
    best = max(scored, default=(0.0, ""))
    return best[1] if best[0] >= 0.5 else None


def _escape(text: str) -> str:
    return _LUCENE_SPECIAL.sub(r"\\\1", text)


class QueryAnalyzer:
    def __init__(self, settings: Settings, graph: Neo4jStore, llm: LLMClient | None, cache: JsonCache) -> None:
        self.settings = settings
        self.graph = graph
        self.llm = llm
        self.cache = cache

    def analyze(self, question: str, today: date | None = None) -> QueryAnalysis:
        language = detect_language(question)
        time_range = self.time_filter(question, today)
        parsed = self._llm_analysis(question) if self.llm else None
        if parsed is None:
            return QueryAnalysis(
                language=language, entities=self._fulltext_entities(question), analyzer="fulltext",
                time_range=time_range,
            )

        entities: list[QueryEntity] = []
        for e in parsed.entities:
            resolved = self.resolve(e.english_name) or self.resolve(e.name)
            entities.append(
                QueryEntity(
                    text=e.name,
                    entity_id=resolved["id"] if resolved else None,
                    entity_name=resolved["name"] if resolved else None,
                    match_score=round(resolved["score"], 3) if resolved else None,
                )
            )
        predicate = normalize_predicate(parsed.predicate)[0] if parsed.predicate else None
        return QueryAnalysis(
            language=language,
            query_type=parsed.query_type,
            entities=entities,
            predicate=predicate if predicate != "RELATED_TO" else None,
            target_type=parsed.target_type,
            filters=self._filters(parsed, question),
            time_range=time_range,
        )

    @staticmethod
    def time_filter(question: str, today: date | None = None) -> TimeFilter | None:
        """Time expression in the question -> date range (deterministic parser, no LLM)."""
        found = parse_time_range(question, today)
        if found is None:
            return None
        return TimeFilter(
            start=None if found.start == date.min else found.start,
            end=None if found.end == date.max else found.end,
            expression=found.expression,
        )

    def _filters(self, parsed: QueryAnalysisLLM, question: str) -> dict[str, str]:
        """Turn the LLM's document / file-type hints into payload filters - only with high confidence, because a
        wrong filter hides the answer: the hint must be visible in the question itself AND match a real document."""
        documents = self.graph.document_sources()
        filters: dict[str, str] = {}
        asked = _tokens(question)
        if parsed.document:
            source = match_document(parsed.document, [d["source"] for d in documents])
            # the question must name the file: at least 2 of its words (or all, for 1-word names)
            if source and len(asked & _tokens(source)) >= min(2, len(_tokens(source))):
                filters["source"] = source
        if not filters and parsed.doc_type:
            in_question = _DOC_TYPE_WORDS[parsed.doc_type] & set(re.split(r"[^\w]+", question.casefold()))
            if in_question and any(d["doc_type"] == parsed.doc_type for d in documents):
                filters["doc_type"] = parsed.doc_type
        return filters

    def _llm_analysis(self, question: str) -> QueryAnalysisLLM | None:
        assert self.llm is not None
        key = JsonCache.key("query", PROMPT_VERSION, self.llm.provider, self.llm.model, question)
        if (cached := self.cache.get(key)) is not None:
            return QueryAnalysisLLM.model_validate(cached)
        try:
            documents = ", ".join(d["source"] for d in self.graph.document_sources(limit=50)) or "(none)"
            user = f"Available documents: {documents}\n\nQuestion: {question}"
            result = self.llm.complete_json(SYSTEM_PROMPT, user, QueryAnalysisLLM, max_tokens=1024)
        except LLMError as exc:
            logger.warning("Query analysis LLM failed, falling back to full-text matching: %s", exc)
            return None
        self.cache.set(key, result.model_dump())
        return result

    def resolve(self, name: str) -> dict | None:
        """Map a name to one graph entity: exact name/alias key first, then all words (AND), then best partial."""
        words = [w for w in _WORD.findall(name) if w]
        if not words:
            return None
        hits = self.graph.search_entities(" AND ".join(_escape(w) for w in words), limit=10)
        for entity_type in ENTITY_TYPES:  # exact key match (handles "Ltd", case, punctuation)
            key = _name_key(name, entity_type)
            for hit in hits:
                if key in {_name_key(n, hit["type"]) for n in [hit["name"], *hit["aliases"]]}:
                    return hit
        if hits:
            return hits[0]
        partial = self.graph.search_entities(" OR ".join(_escape(w) for w in words), limit=1)
        # partial matches must cover most of the name, else "Ganga Roadways" would resolve to "Ganga Smart Power"
        if partial:
            hit_words = {w.casefold() for n in [partial[0]["name"], *partial[0]["aliases"]] for w in _WORD.findall(n)}
            if sum(w.casefold() in hit_words for w in words) / len(words) >= 0.6:
                return partial[0]
        return None

    def _fulltext_entities(self, question: str) -> list[QueryEntity]:
        """No LLM: match the question's significant words against entity names directly."""
        words = [w for w in _WORD.findall(question) if w.casefold() not in _STOP and len(w) > 2]
        if not words:
            return []
        hits = self.graph.search_entities(" OR ".join(_escape(w) for w in words), limit=5)
        top = hits[0]["score"] if hits else 0
        return [
            QueryEntity(text=h["name"], entity_id=h["id"], entity_name=h["name"], match_score=round(h["score"], 3))
            for h in hits
            if h["score"] >= 0.5 * top  # keep the clear matches only
        ]
