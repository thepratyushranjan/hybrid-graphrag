"""Dump rows -> PostRecord / StanceRecord / Hierarchy.

The upstream pipeline already extracted entities, districts, categories and sentiment into columns (JSON
arrays stored as text), so the graph is built from those columns directly: no LLM call, deterministic,
and re-runs produce identical node IDs.
"""

import json
import logging
import re
import unicodedata
from collections import Counter
from datetime import datetime
from typing import Any

from graphrag.ingestion.sql.gazetteer import DISTRICT_ALIASES, DISTRICT_CANONICAL
from graphrag.models.social import Hierarchy, NamedRef, PostRecord, SqlCorpus, StanceRecord

logger = logging.getLogger(__name__)

TABLES = {"analyzed_data", "sentiment_entities", "thana_matrix", "broad_category", "sub_category", "keywords"}

# Indian mobile numbers (with optional +91 / 0 prefix) are never embedded, stored or shown
_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")
MAX_NAME_CHARS = 80

PLATFORMS = {
    "twitter": "twitter", "x": "twitter", "facebook": "facebook", "instagram": "instagram",
    "whatsapp": "whatsapp", "youtube": "youtube", "news_rss_feed": "news", "web": "news", "news": "news",
}
LANGUAGES = {"hinglish": "hi", "hindi": "hi", "english": "en"}
# words the upstream extractor returns as a "thana" that name no police station
_NOT_A_THANA = {"क्षेत्र", "थाना", "थाना क्षेत्र", "क्षेत्र का मामला", "इलाका", "इलाके", "कोतवाली क्षेत्र", "area"}
# placeholder author names some sources use instead of a real handle
NO_AUTHOR = {"unknown_id", "unknown", "none", "null", "anonymous"}
ENTITY_COLUMNS = {
    "person_names": "person",
    "organisation_names": "organisation",
    "location_names": "location",
    "incidents": "incident",
}


def mask_phones(text: str) -> str:
    return _PHONE.sub("[phone removed]", text)


def name_key(name: str) -> str:
    """Case/space/punctuation-insensitive key ("UP  Police." == "up police"), keeping Devanagari marks."""
    folded = unicodedata.normalize("NFC", name).casefold()
    kept = "".join(ch if unicodedata.category(ch)[0] in "LMN" else " " for ch in folded)
    return "_".join(kept.split())


def clean_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    name = " ".join(value.split()).strip(" ,.;:-'\"")
    if not name or len(name) > MAX_NAME_CHARS or not name_key(name):
        return None
    return name


def json_list(value: Any) -> list[str]:
    """Columns hold JSON arrays as text ('["Lucknow"]'); tolerate NULL, '' and plain strings."""
    if value is None or value == "":
        return []
    if isinstance(value, list):
        items = value
    else:
        try:
            items = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            items = [value]
        if not isinstance(items, list):
            items = [items]
    seen: dict[str, str] = {}
    for item in items:
        if (name := clean_name(item)) is not None:
            seen.setdefault(name_key(name), name)
    return list(seen.values())


def canonical_district(name: str) -> str:
    key = name.casefold().strip()
    if key in DISTRICT_CANONICAL:
        return DISTRICT_CANONICAL[key]
    for english, aliases in DISTRICT_ALIASES.items():
        if key == english.casefold() or name in aliases or key in {a.casefold() for a in aliases}:
            return english
    return name.strip()


def district_ref(name: str) -> NamedRef:
    canonical = canonical_district(name)
    return NamedRef(id=f"district:{name_key(canonical)}", name=canonical)


def _ref(prefix: str, name: str, kind: str | None = None) -> NamedRef:
    return NamedRef(id=f"{prefix}:{name_key(name)}", name=name, kind=kind)


def account_ref(platform: str, username: str) -> NamedRef | None:
    handle = username.strip().lstrip("@")
    if not handle or not name_key(handle):
        return None
    return NamedRef(id=f"account:{platform}:{handle.casefold()}", name=f"@{handle}")


def hashtag_ref(tag: str) -> NamedRef | None:
    bare = tag.strip().lstrip("#")
    if not name_key(bare):
        return None
    return NamedRef(id=f"hashtag:{name_key(bare)}", name=f"#{bare}")


def _clean_text(text: str, limit: int) -> str:
    text = _BLANK_LINES.sub("\n\n", _SPACES.sub(" ", mask_phones(text))).strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def _time(row: dict[str, Any]) -> datetime | None:
    for column in ("post_bank_post_timestamp", "created_at"):
        value = row.get(column)
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                continue
    return None


def _thanas(names: list[str], districts: list[NamedRef]) -> list[NamedRef]:
    """A thana is keyed by its district: "कोतवाली" exists in every district."""
    district = districts[0] if districts else None
    refs = []
    for name in names:
        if name in _NOT_A_THANA or name.casefold() in _NOT_A_THANA or len(name.split()) > 3:
            continue
        scope = district.id.removeprefix("district:") if district else "unknown"
        refs.append(NamedRef(id=f"thana:{scope}:{name_key(name)}", name=name,
                             kind=district.name if district else None))
    return refs


def normalize_post(row: dict[str, Any], post_chars: int) -> PostRecord | None:
    text = row.get("input_text") or row.get("post_bank_post_snippet") or row.get("post_bank_post_title") or ""
    summary = row.get("contextual_understanding") or None
    if not str(text).strip() and not summary:
        return None
    platform_raw = str(row.get("post_bank_core_source") or row.get("post_bank_source") or row.get("source_type") or "")
    platform = PLATFORMS.get(platform_raw.casefold(), platform_raw.casefold() or "unknown")
    districts = list({d.id: d for d in map(district_ref, json_list(row.get("primary_district"))
                                             or json_list(row.get("district_names")))}.values())
    entities: dict[str, NamedRef] = {}
    for column, kind in ENTITY_COLUMNS.items():
        for name in json_list(row.get(column)):
            entities.setdefault(f"entity:{name_key(name)}", NamedRef(id=f"entity:{name_key(name)}", name=name,
                                                                     kind=kind))
    author = row.get("post_bank_author_username")
    account = account_ref(platform, author) if isinstance(author, str) and author.casefold() not in NO_AUTHOR else None
    mentions = [m for m in (account_ref(platform, h) for h in json_list(row.get("mention_ids_extracted"))) if m]
    topic_id, topic_title = row.get("unique_topic_id"), clean_name(row.get("topic_title")) or None
    return PostRecord(
        post_id=f"post:{row['id']}",
        row_id=int(row["id"]),
        text=_clean_text(str(text), post_chars),
        summary=_clean_text(summary, post_chars) if summary else None,
        url=row.get("post_bank_post_url") or None,
        platform=platform,
        author=account.name.lstrip("@") if account else None,
        author_name=clean_name(row.get("post_bank_author_name")),
        posted_at=_time(row),
        language=LANGUAGES.get(str(row.get("detected_language") or "").casefold(), "unknown"),
        topic=NamedRef(id=f"topic:{topic_id}", name=topic_title or str(topic_id)) if topic_id else None,
        districts=districts,
        thanas=_thanas(json_list(row.get("primary_thana")) or json_list(row.get("thana_names")), districts),
        categories=[_ref("category", c.upper()) for c in json_list(row.get("broad_category"))],
        sub_categories=[_ref("subcategory", c.upper()) for c in json_list(row.get("sub_category"))],
        entities=list(entities.values()),
        hashtags=[h for h in map(hashtag_ref, json_list(row.get("hashtags"))) if h],
        mentions=[m for m in {m.id: m for m in mentions}.values() if not account or m.id != account.id],
        account=account,
        sentiment=(str(row["sentiment_label"]).casefold() if row.get("sentiment_label") else None),
        emotion=(str(row["emotional_primary_emotion"]).casefold() if row.get("emotional_primary_emotion") else None),
    )


def _hierarchy(tables: dict[str, list[dict[str, Any]]]) -> Hierarchy:
    thanas = []
    for row in tables.get("thana_matrix", []):
        district, thana = clean_name(row.get("district")), clean_name(row.get("thana"))
        if not district or not thana:
            continue
        d = district_ref(district)
        unit_name = clean_name(row.get("commissionarate")) or clean_name(row.get("dist_range")) or clean_name(row.get("zone"))
        unit_kind = ("commissionerate" if row.get("commissionarate") else "range" if row.get("dist_range")
                     else "zone" if row.get("zone") else None)
        thanas.append({
            "id": f"thana:{d.id.removeprefix('district:')}:{name_key(thana)}", "name": thana,
            "district_id": d.id, "district": d.name,
            "unit_id": f"unit:{name_key(unit_name)}" if unit_name else None, "unit": unit_name, "unit_kind": unit_kind,
        })
    broad = {row["id"]: clean_name(row.get("broad_category")) for row in tables.get("broad_category", [])}
    subs = []
    for row in tables.get("sub_category", []):
        name, parent = clean_name(row.get("sub_category")), broad.get(row.get("broad_category_id"))
        if name:
            subs.append({
                "id": f"subcategory:{name_key(name.upper())}", "name": name.upper(),
                "category_id": f"category:{name_key(parent.upper())}" if parent else None,
                "category": parent.upper() if parent else None, "hint": row.get("hint"),
            })
    aliases: dict[str, list[str]] = {}
    for row in tables.get("keywords", []):
        sub = clean_name(row.get("sub_category_name"))
        if not sub:
            continue
        words = [w.strip() for col in ("hindi_keyword", "english_keyword", "hinglish_keyword")
                 for w in str(row.get(col) or "").split(",") if 3 <= len(w.strip()) <= 40]
        aliases.setdefault(f"subcategory:{name_key(sub.upper())}", []).extend(words)
    return Hierarchy(thanas=thanas, sub_categories=subs,
                     keyword_aliases={k: list(dict.fromkeys(v)) for k, v in aliases.items()})


def normalize(tables: dict[str, list[dict[str, Any]]], post_chars: int, limit: int = 0) -> SqlCorpus:
    """Newest posts first (the dump is "latest N"), so a limited ingest keeps the most recent ones."""
    skipped: Counter[str] = Counter()
    rows = sorted(tables.get("analyzed_data", []), key=lambda r: r.get("id") or 0, reverse=True)
    posts: list[PostRecord] = []
    for row in rows:
        if limit and len(posts) >= limit:
            skipped["over SQL_INGEST_LIMIT"] += 1
            continue
        try:
            post = normalize_post(row, post_chars)
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Skipping analyzed_data row %s: %s", row.get("id"), exc)
            skipped["malformed row"] += 1
            continue
        if post is None:
            skipped["no text"] += 1
            continue
        posts.append(post)

    kept = {p.row_id for p in posts}
    stances = []
    for row in tables.get("sentiment_entities", []):
        name = clean_name(row.get("entity_name"))
        if row.get("analyzed_data_id") not in kept:
            skipped["stance for a post not in the dump"] += 1
            continue
        if not name:
            continue
        stances.append(StanceRecord(
            post_id=f"post:{row['analyzed_data_id']}",
            entity=NamedRef(id=f"entity:{name_key(name)}", name=name, kind="entity"),
            stance=str(row.get("stance") or "neutral").casefold(),
            confidence=float(row.get("confidence") or 0.0),
        ))
    return SqlCorpus(posts=posts, stances=stances, hierarchy=_hierarchy(tables), skipped=dict(skipped))
