"""SQL corpus: dump parsing, normalisation and the deterministic question analysis (made-up rows only)."""

from datetime import date
from typing import Any

import pytest

from graphrag.graph.social_store import filter_params
from graphrag.ingestion.sql.dump_parser import DumpParseError, load_tables
from graphrag.ingestion.sql.normalizer import TABLES, mask_phones, normalize
from graphrag.ingestion.sql.pipeline import post_chunk, post_payload
from graphrag.retrieval.social_analyzer import SocialQueryAnalyzer
from graphrag.retrieval.social_retriever import qdrant_filters

COLUMNS = (
    "`id`, `input_text`, `contextual_understanding`, `topic_title`, `unique_topic_id`, `primary_district`, "
    "`primary_thana`, `broad_category`, `sub_category`, `person_names`, `organisation_names`, `hashtags`, "
    "`mention_ids_extracted`, `sentiment_label`, `post_bank_core_source`, `post_bank_author_username`, "
    "`post_bank_post_url`, `post_bank_post_timestamp`, `detected_language`"
)
DUMP = f"""
-- made-up rows for tests
INSERT INTO `analyzed_data` ({COLUMNS}) VALUES
  (1, 'बरेली में मारपीट, call 9876543210 now\\nsecond line', 'दो पक्षों में मारपीट हुई', 'बरेली - मारपीट', 't-1',
   '["बरेली"]', '["बारादरी", "क्षेत्र"]', '["CRIME"]', '["ASSAULT"]', '["राम कुमार"]', '["UP Police"]',
   '["#bareilly"]', '["@bareillypolice"]', 'negative', 'TWITTER', 'news_one', 'https://x.com/a/1',
   '2026-10-03 14:06:57', 'hinglish'),
  (2, 'It''s a test with a backslash \\\\ here', NULL, NULL, NULL, '[]', '[]', '["GRIEVANCE"]',
   '["GENERAL COMPLAINTS"]', '[]', '[]', '[]', '[]', 'neutral', 'facebook', 'unknown_id', NULL,
   '2026-10-02 08:00:00', 'english');
INSERT INTO `thana_matrix` (`id`, `commissionarate`, `dist_range`, `district`, `thana`, `zone`) VALUES
  (7, 'Comm. Gautambudh Nagar', NULL, 'Gautambudh Nagar', 'Cyber thana', NULL);
INSERT INTO `post_bank` (`id`, `post_title`) VALUES (9, 'ignored table; with ) and ( inside');
"""


@pytest.fixture(scope="module")
def tables() -> dict[str, list[dict[str, Any]]]:
    return load_tables(DUMP.encode(), TABLES)


def test_parser_handles_escapes_nulls_and_numbers(tables: dict[str, list[dict[str, Any]]]) -> None:
    rows = tables["analyzed_data"]
    assert [r["id"] for r in rows] == [1, 2]
    assert rows[0]["input_text"].endswith("now\nsecond line")
    assert rows[1]["input_text"] == "It's a test with a backslash \\ here"
    assert rows[1]["contextual_understanding"] is None
    assert tables["thana_matrix"][0]["dist_range"] is None
    assert "post_bank" not in tables  # only the requested tables are kept


def test_parser_rejects_broken_rows() -> None:
    with pytest.raises(DumpParseError):
        load_tables(b"INSERT INTO `t` (`a`, `b`) VALUES (1);", {"t"})


def test_normalize_builds_posts(tables: dict[str, list[dict[str, Any]]]) -> None:
    corpus = normalize(tables, post_chars=500)
    newest, older = corpus.posts  # newest (highest id) first
    assert newest.row_id == 2 and older.row_id == 1
    assert [d.name for d in older.districts] == ["Bareilly"]  # Hindi name -> canonical district
    assert [t.name for t in older.thanas] == ["बारादरी"]  # "क्षेत्र" (area) is not a thana
    assert older.thanas[0].id == "thana:bareilly:बारादरी"
    assert "9876543210" not in older.text and "[phone removed]" in older.text
    assert older.account and older.account.id == "account:twitter:news_one"
    assert [m.name for m in older.mentions] == ["@bareillypolice"]
    assert {e.kind for e in older.entities} == {"person", "organisation"}
    assert older.platform == "twitter" and older.language == "hi"
    assert newest.account is None  # "unknown_id" is not an author
    assert corpus.hierarchy.thanas[0]["district"] == "Gautam Buddha Nagar"


def test_normalize_limit_keeps_newest(tables: dict[str, list[dict[str, Any]]]) -> None:
    corpus = normalize(tables, post_chars=500, limit=1)
    assert [p.row_id for p in corpus.posts] == [2]
    assert corpus.skipped == {"over SQL_INGEST_LIMIT": 1}


def test_post_chunk_and_payload(tables: dict[str, list[dict[str, Any]]]) -> None:
    post = normalize(tables, post_chars=500).posts[1]
    chunk = post_chunk(post, token_count=42)
    assert chunk.chunk_id == "post:1" and chunk.doc_type == "post"
    assert chunk.source == "https://x.com/a/1"
    assert chunk.date_start == "2026-10-03T14:06:57Z"
    assert chunk.text.startswith("[twitter | @news_one | 2026-10-03 | district: Bareilly")
    payload = post_payload(post)
    assert payload["district"] == ["Bareilly"] and payload["sub_category"] == ["ASSAULT"]


def test_mask_phones_keeps_other_numbers() -> None:
    assert mask_phones("call +91 98765 43210 or 2106302472755245265") == "call [phone removed] or 2106302472755245265"


class FakeGraph:
    """Stands in for SocialGraphStore: a fixed name index."""

    NODES = [
        {"id": "district:bareilly", "name": "Bareilly", "label": "District", "kind": None, "aliases": ["बरेली"], "posts": 5},
        {"id": "district:lucknow", "name": "Lucknow", "label": "District", "kind": None, "aliases": ["लखनऊ"], "posts": 9},
        {"id": "category:crime", "name": "CRIME", "label": "Category", "kind": None, "aliases": [], "posts": 7},
        {"id": "subcategory:accident", "name": "ACCIDENT", "label": "SubCategory", "kind": None, "aliases": [], "posts": 3},
        {"id": "subcategory:assault", "name": "ASSAULT", "label": "SubCategory", "kind": None, "aliases": ["मारपीट"],
         "posts": 4},
        {"id": "entity:विकास_यादव", "name": "विकास यादव", "label": "SocialEntity", "kind": "person", "aliases": [],
         "posts": 3},
        {"id": "hashtag:bareilly", "name": "#bareilly", "label": "Hashtag", "kind": None, "aliases": [], "posts": 2},
    ]

    def post_count(self) -> int:
        return 10

    def name_index(self) -> list[dict[str, Any]]:
        return self.NODES


@pytest.fixture
def analyzer() -> SocialQueryAnalyzer:
    return SocialQueryAnalyzer(FakeGraph())  # type: ignore[arg-type]


def test_lookup_hindi_district_and_person(analyzer: SocialQueryAnalyzer) -> None:
    a = analyzer.analyze("बरेली में विकास यादव के बारे में पोस्ट")
    assert a.query_type == "lookup"
    assert a.filters == {"district": "Bareilly"}
    assert a.entity_ids == ["entity:विकास_यादव"]


def test_aggregation_with_filters_group_and_top_n(analyzer: SocialQueryAnalyzer) -> None:
    a = analyzer.analyze("Top 5 districts by negative CRIME posts on twitter")
    assert a.query_type == "aggregation"
    assert a.filters == {"platform": "twitter", "sentiment": "negative", "category": "CRIME"}
    assert a.group_by == ["district"] and a.top_n == 5


def test_keyword_alias_only_narrows_counts(analyzer: SocialQueryAnalyzer) -> None:
    assert "sub_category" not in analyzer.analyze("मारपीट की खबरें").filters
    assert analyzer.analyze("कितने मारपीट के मामले हैं").filters["sub_category"] == "ASSAULT"


def test_plural_category_hashtag_and_time(analyzer: SocialQueryAnalyzer) -> None:
    a = analyzer.analyze("road accidents in लखनऊ last week", today=date(2026, 10, 8))
    assert a.filters == {"sub_category": "ACCIDENT", "district": "Lucknow"}
    assert a.time_range is not None and a.time_range.end is not None
    assert analyzer.analyze("posts about bareilly").entity_ids == []  # hashtags only with an explicit "#"
    assert analyzer.analyze("posts tagged #bareilly").entity_ids == ["hashtag:bareilly"]


def test_filter_mapping_for_qdrant_and_cypher() -> None:
    filters = {"district": "Bareilly", "category": "CRIME", "sub_category": "ASSAULT"}
    assert qdrant_filters(filters) == {"district": "Bareilly", "broad_category": "CRIME", "sub_category": "ASSAULT"}
    params = filter_params(filters, None)
    assert params["district"] == "Bareilly" and params["platform"] is None and params["start"] is None
