"""Triple extraction checks with a fake LLM (no network): evidence check, whitelist, entity resolution."""

from pathlib import Path

from pydantic import BaseModel

from graphrag.config import Settings
from graphrag.graph.extractor import EntityResolver, TripleExtractor, entity_id, normalize_predicate
from graphrag.llm.cache import JsonCache
from graphrag.models import Chunk, ExtractionResult

TEXT = (
    "Clause 7.2 impacts DataSecure Cloud Pvt Ltd, which hosts data for Ganga Smart Power Ltd. "
    "शक्ति स्टील वर्क्स supplies guard rails to Ganga Roadways."
)


class FakeLLM:
    """Returns a canned result; `by_schema` gives different answers per response schema (e.g. query analysis)."""

    provider, model = "fake", "fake-1"

    def __init__(self, result: dict, by_schema: dict[str, dict] | None = None) -> None:
        self.result = result
        self.by_schema = by_schema or {}
        self.calls = 0

    def complete_json(self, system: str, user: str, schema: type[BaseModel], max_tokens: int = 4096) -> BaseModel:
        self.calls += 1
        return schema.model_validate(self.by_schema.get(schema.__name__, self.result))


RESULT = {
    "entities": [
        {"name": "Clause 7.2", "type": "Concept"},
        {"name": "DataSecure Cloud Pvt Ltd", "type": "Organization"},
        {"name": "Ganga Smart Power Ltd", "type": "Organization"},
        {"name": "Shakti Steel Works", "type": "Organization", "aliases": ["शक्ति स्टील वर्क्स"]},
    ],
    "triples": [
        # kept
        {"subject": "Clause 7.2", "predicate": "IMPACTS", "object": "DataSecure Cloud Pvt Ltd",
         "evidence": "Clause 7.2 impacts DataSecure Cloud Pvt Ltd", "confidence": 0.9},
        # unknown predicate -> RELATED_TO, original kept
        {"subject": "DataSecure Cloud Pvt Ltd", "predicate": "hosts data for", "object": "Ganga Smart Power Ltd",
         "evidence": "which hosts data for Ganga Smart Power Ltd", "confidence": 0.8},
        # endpoint not listed, but a proper name in the chunk text -> created
        {"subject": "Shakti Steel Works", "predicate": "SUPPLIES", "object": "Ganga Roadways",
         "evidence": "शक्ति स्टील वर्क्स supplies guard rails to Ganga Roadways", "confidence": 0.9},
        # hallucinated evidence -> dropped
        {"subject": "Clause 7.2", "predicate": "IMPACTS", "object": "Ganga Smart Power Ltd",
         "evidence": "Clause 7.2 impacts Ganga Smart Power", "confidence": 0.9},
        # low confidence -> dropped
        {"subject": "Clause 7.2", "predicate": "IMPACTS", "object": "Ganga Smart Power Ltd",
         "evidence": "Clause 7.2 impacts", "confidence": 0.2},
        # generic noun endpoint -> dropped
        {"subject": "Clause 7.2", "predicate": "IMPACTS", "object": "vendor",
         "evidence": "Clause 7.2 impacts", "confidence": 0.9},
    ],
}


def _chunk() -> Chunk:
    return Chunk(chunk_id="chunk:abc", doc_id="doc:x", source="x.md", doc_type="md", chunk_index=0,
                 text=TEXT, token_count=40)


def test_extract_applies_all_checks(settings: Settings, tmp_path: Path) -> None:
    graph = TripleExtractor(settings, FakeLLM(RESULT), JsonCache(tmp_path)).extract(_chunk(), EntityResolver())  # type: ignore[arg-type]
    rels = {(r.subject_id, r.predicate, r.object_id) for r in graph.relations}
    assert rels == {
        ("concept:clause_7_2", "IMPACTS", "org:datasecure_cloud"),
        ("org:datasecure_cloud", "RELATED_TO", "org:ganga_smart_power"),
        ("org:shakti_steel_works", "SUPPLIES", "concept:ganga_roadways"),
    }
    assert graph.dropped == 3
    related = next(r for r in graph.relations if r.predicate == "RELATED_TO")
    assert related.original_predicate == "hosts data for"


def test_llm_called_once_per_chunk_text(settings: Settings, tmp_path: Path) -> None:
    llm = FakeLLM(RESULT)
    extractor = TripleExtractor(settings, llm, JsonCache(tmp_path))  # type: ignore[arg-type]
    extractor.extract(_chunk(), EntityResolver())
    extractor.extract(_chunk(), EntityResolver())
    assert llm.calls == 1


def test_resolver_merges_hindi_alias_and_company_suffix() -> None:
    resolver = EntityResolver()
    english = resolver.resolve("Shakti Steel Works Pvt Ltd", "Organization", ["शक्ति स्टील वर्क्स"])
    hindi = resolver.resolve("Shakti Steel", "Organization", ["शक्ति स्टील वर्क्स"])
    assert english.id == hindi.id == "org:shakti_steel_works"
    assert resolver.lookup("शक्ति स्टील वर्क्स") == resolver.entities["org:shakti_steel_works"]


def test_entity_ids_and_predicates() -> None:
    assert entity_id("DataSecure Cloud Pvt. Ltd.", "Organization") == "org:datasecure_cloud"
    assert entity_id("Neha Kapoor", "Person") == "person:neha_kapoor"
    assert normalize_predicate("subcontracts to") == ("SUBCONTRACTS_TO", None)
    assert normalize_predicate("is near") == ("RELATED_TO", "is near")


def test_extraction_schema_round_trip() -> None:
    assert ExtractionResult.model_validate(RESULT).triples[0].predicate == "IMPACTS"
