"""Step 1 for the SQL corpus: deterministic question analysis (no LLM call).

Matches the question's words against the names in the social graph (districts with their Hindi names,
categories, thanas, people/organisations, @accounts, #hashtags) and finds:
  - filters:   district / platform / sentiment / category / sub_category / time range  -> Qdrant + Cypher
  - entities:  graph nodes to start the traversal from                                -> Cypher seeds
  - query type and group-by dimensions for counting questions ("top 5 districts by ...")
"""

import logging
import re
import threading
import time
import unicodedata
from datetime import date
from typing import Any

from graphrag.graph.social_store import SocialGraphStore
from graphrag.ingestion.chunker import detect_language
from graphrag.ingestion.sql.normalizer import name_key
from graphrag.models import QueryAnalysis, QueryEntity
from graphrag.retrieval.query_analyzer import QueryAnalyzer

logger = logging.getLogger(__name__)

MAX_NGRAM = 5
INDEX_TTL_SECONDS = 30  # re-check the post count (cheap) at most this often; rebuild the index when it changed

AGGREGATION_WORDS = {
    "how many", "count", "number of", "top", "most", "highest", "least", "lowest", "ranking", "rank",
    "distribution", "breakdown", "statistics", "stats", "trend", "total", "compare", "maximum", "minimum",
    "कितने", "कितनी", "कितना", "सबसे ज्यादा", "सबसे अधिक", "सबसे कम", "सर्वाधिक", "गिनती", "संख्या", "कुल", "टॉप",
    "kitne", "kitni", "sabse jyada", "sabse zyada",
}
RELATIONSHIP_WORDS = {
    "connected", "linked", "related", "relation", "relationship", "together", "who else", "along with", "network",
    "associated", "जुड़े", "जुड़ा", "संबंध", "साथ", "रिश्ता",
}
# group-by dimension -> words that ask for it
GROUP_WORDS: dict[str, set[str]] = {
    "district": {"district", "districts", "जिला", "जिले", "जिलों", "जनपद", "जनपदों", "zila"},
    "thana": {"thana", "thanas", "police station", "police stations", "थाना", "थाने", "थानों"},
    "sub_category": {"sub category", "sub-category", "subcategory", "sub categories", "उप श्रेणी", "crime type",
                     "type of crime", "types of crime", "incident type"},
    "category": {"category", "categories", "श्रेणी", "श्रेणियों", "कैटेगरी"},
    "platform": {"platform", "platforms", "source", "sources", "प्लेटफॉर्म", "प्लेटफार्म", "माध्यम"},
    "sentiment": {"sentiment", "sentiments", "भावना", "सेंटीमेंट"},
    "emotion": {"emotion", "emotions", "भाव", "भावनाएं"},
    "account": {"account", "accounts", "user", "users", "author", "authors", "handle", "handles", "who posted",
                "posted the most", "अकाउंट", "यूजर", "किसने"},
    "mentioned_account": {"tagged account", "tagged accounts", "mentioned account", "mentioned accounts"},
    "hashtag": {"hashtag", "hashtags", "हैशटैग"},
    "topic": {"topic", "topics", "issue", "issues", "विषय", "मुद्दे", "मुद्दा", "मामले"},
    "person": {"person", "people", "persons", "leader", "leaders", "व्यक्ति", "लोग", "लोगों", "नेता"},
    "organisation": {"organisation", "organization", "organisations", "organizations", "party", "parties",
                     "संगठन", "पार्टी"},
    "incident": {"incident", "incidents", "घटना", "घटनाएं", "घटनाओं"},
    "date": {"day", "daily", "date", "per day", "दिन", "तारीख", "रोज"},
}
PLATFORM_WORDS = {
    "twitter": {"twitter", "tweet", "tweets", "ट्विटर", "ट्वीट"},
    "facebook": {"facebook", "fb", "फेसबुक"},
    "whatsapp": {"whatsapp", "व्हाट्सएप", "व्हाट्सऐप"},
    "youtube": {"youtube", "यूट्यूब"},
    "news": {"news", "newspaper", "news article", "news articles", "न्यूज़", "न्यूज", "समाचार", "खबर", "खबरें"},
    "instagram": {"instagram", "इंस्टाग्राम"},
}
SENTIMENT_WORDS = {
    "negative": {"negative", "नकारात्मक", "नेगेटिव"},
    "positive": {"positive", "सकारात्मक", "पॉजिटिव", "appreciation", "praise", "तारीफ", "प्रशंसा"},
    "neutral": {"neutral", "तटस्थ", "न्यूट्रल"},
}
# single words that name nothing specific on their own (never a graph seed)
STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was", "were", "which", "what", "who",
    "how", "many", "much", "posts", "post", "about", "show", "me", "give", "list", "tell", "with", "by", "from",
    "police", "पुलिस", "up", "news", "case", "cases", "मामला", "मामले", "घटना", "पोस्ट", "के", "की", "का", "को",
    "में", "से", "पर", "है", "हैं", "और", "कौन", "क्या", "कैसे", "किस", "ने", "वाली", "वाले", "दिखाओ", "बताओ",
    "सरकार", "प्रशासन", "जिला", "थाना", "क्षेत्र", "लोग", "user", "district", "thana",
}
_TOP_N = re.compile(r"(?:top|टॉप|सबसे\s+(?:ज्यादा|अधिक)\s+वाले)\s+(\d{1,2})|(\d{1,2})\s+(?:सबसे|most|top)", re.I)
_HANDLE = re.compile(r"@([A-Za-z0-9_]{2,})")
_HASHTAG = re.compile(r"#([\wऀ-ॿ]+)", re.UNICODE)
LABEL_PRIORITY = {"District": 0, "SubCategory": 1, "Category": 2, "Thana": 3, "SocialEntity": 4, "Account": 5}


def words(text: str) -> list[str]:
    cleaned = "".join(" " if ch == "_" or unicodedata.category(ch)[0] in "PSZ" else ch
                      for ch in unicodedata.normalize("NFC", text).casefold())
    return cleaned.split()


def _has_phrase(tokens: list[str], phrases: set[str]) -> bool:
    joined = " " + " ".join(tokens) + " "
    return any(" " + " ".join(words(p)) + " " in joined for p in phrases)


class SocialQueryAnalyzer:
    def __init__(self, graph: SocialGraphStore, hub_limit: int = 50) -> None:
        self.graph = graph
        self.hub_limit = hub_limit
        self._lock = threading.Lock()
        self._index: dict[str, list[dict[str, Any]]] = {}  # name key -> nodes
        self._posts = -1
        self._checked = 0.0

    # ---------- name index ----------

    def _refresh(self) -> None:
        with self._lock:
            if time.monotonic() - self._checked < INDEX_TTL_SECONDS and self._index:
                return
            self._checked = time.monotonic()
            posts = self.graph.post_count()
            if posts == self._posts and self._index:
                return
            index: dict[str, list[dict[str, Any]]] = {}
            for node in self.graph.name_index():
                names = [node["name"], *node["aliases"]]
                if node["label"] == "Account":
                    names = [node["name"].lstrip("@")]
                for name in names:
                    if name and (key := name_key(name)):
                        index.setdefault(key, []).append({**node, "alias": node["name"] != name})
            self._index, self._posts = index, posts
            logger.info("Social name index: %d names over %d posts", len(index), posts)

    def invalidate(self) -> None:
        with self._lock:
            self._checked = 0.0
            self._posts = -1

    # ---------- analysis ----------

    def analyze(self, question: str, today: date | None = None) -> QueryAnalysis:
        self._refresh()
        tokens = words(question)
        query_type = "lookup"
        if _has_phrase(tokens, AGGREGATION_WORDS):
            query_type = "aggregation"
        elif _has_phrase(tokens, RELATIONSHIP_WORDS):
            query_type = "relationship"

        filters: dict[str, str] = {}
        for platform, phrases in PLATFORM_WORDS.items():
            if _has_phrase(tokens, phrases):
                filters["platform"] = platform
                break
        for sentiment, phrases in SENTIMENT_WORDS.items():
            if _has_phrase(tokens, phrases):
                filters["sentiment"] = sentiment
                break

        entities: list[QueryEntity] = []
        for node, text in self._match_names(tokens, aggregation=query_type == "aggregation"):
            label = node["label"]
            if label == "District" and "district" not in filters:
                filters["district"] = node["name"]
            elif label == "Category" and "category" not in filters:
                filters["category"] = node["name"]
            elif label == "SubCategory" and "sub_category" not in filters:
                filters["sub_category"] = node["name"]
            elif label in ("SocialEntity", "Account", "Thana", "Hashtag"):
                entities.append(QueryEntity(text=text, entity_id=node["id"], entity_name=node["name"], match_score=1.0))

        for handle in _HANDLE.findall(question):
            for node in self._index.get(name_key(handle), []):
                if node["label"] == "Account":
                    entities.append(QueryEntity(text=f"@{handle}", entity_id=node["id"], entity_name=node["name"],
                                                match_score=1.0))
        for tag in _HASHTAG.findall(question):
            for node in self._index.get(name_key(tag), []):
                if node["label"] == "Hashtag":
                    entities.append(QueryEntity(text=f"#{tag}", entity_id=node["id"], entity_name=node["name"],
                                                match_score=1.0))

        group_by = [dim for dim, phrases in GROUP_WORDS.items() if _has_phrase(tokens, phrases)]
        if query_type == "aggregation":
            # "how many posts per district in Lucknow" groups by district; the district filter alone doesn't
            group_by = [d for d in group_by if not (d == "district" and "district" in filters and len(group_by) > 1)]
        unique = list({e.entity_id: e for e in entities}.values())
        top = next((int(a or b) for a, b in _TOP_N.findall(question)), None)
        return QueryAnalysis(
            language=detect_language(question),
            query_type=query_type,  # type: ignore[arg-type]
            entities=unique,
            analyzer="fulltext",
            filters=filters,
            time_range=QueryAnalyzer.time_filter(question, today),
            group_by=group_by[:2],
            top_n=top if query_type == "aggregation" else None,
        )

    def _match_names(self, tokens: list[str], aggregation: bool) -> list[tuple[dict[str, Any], str]]:
        """Longest n-grams first; every question word is used by at most one match."""
        used: set[int] = set()
        found: list[tuple[dict[str, Any], str]] = []
        for size in range(min(MAX_NGRAM, len(tokens)), 0, -1):
            for start in range(len(tokens) - size + 1):
                span = set(range(start, start + size))
                if span & used:
                    continue
                phrase = tokens[start : start + size]
                if size == 1 and (phrase[0] in STOP or len(phrase[0]) < 3):
                    continue
                key = name_key(" ".join(phrase))
                candidates = [n for n in self._index.get(key, []) if self._usable(n, aggregation)]
                if not candidates and key.endswith("s"):  # "accidents" -> ACCIDENT, "murders" -> MURDER
                    candidates = [n for n in self._index.get(key[:-1], []) + self._index.get(key[:-2], [])
                                  if n["label"] in ("Category", "SubCategory") and self._usable(n, aggregation)]
                if not candidates:
                    continue
                best = min(candidates, key=lambda n: (LABEL_PRIORITY.get(n["label"], 9), -n["posts"]))
                found.append((best, " ".join(phrase)))
                used |= span
        return found

    @staticmethod
    def _usable(node: dict[str, Any], aggregation: bool) -> bool:
        if node["label"] == "Hashtag":
            return False  # only with an explicit "#tag"
        if node["label"] == "Account":
            return len(node["name"]) >= 7  # bare usernames only when distinctive ("@" handles always work)
        if node["label"] == "SubCategory" and node["alias"]:
            # keyword aliases ("मारपीट" -> ASSAULT) only narrow counting questions; lookups use vector search
            return aggregation
        if node["label"] in ("SocialEntity", "Thana"):
            return node["posts"] > 0
        return True
