"""Neo4j social graph for the SQL corpus, built from structured columns (no LLM):

    (Post)-[:PART_OF]->(Topic)            (Post)-[:POSTED_BY]->(Account)     (Post)-[:MENTIONS_ACCOUNT]->(Account)
    (Post)-[:IN_DISTRICT]->(District)     (Post)-[:IN_THANA]->(Thana)-[:IN_DISTRICT]->(District)-[:IN_UNIT]->(PoliceUnit)
    (Post)-[:IN_CATEGORY]->(Category)     (Post)-[:IN_SUBCATEGORY]->(SubCategory)-[:UNDER]->(Category)
    (Post)-[:MENTIONS]->(SocialEntity {kind: person|organisation|location|incident})
    (Post)-[:STANCE {stance, confidence}]->(SocialEntity)    (Post)-[:TAGGED]->(Hashtag)

Labels are separate from the document graph (Document / Chunk / Entity), so the two corpora never mix.
Every node is MERGEd on `id` (unique constraint) and a post's outgoing edges are rebuilt on re-ingest, so
running the ingest twice leaves the counts unchanged.
"""

import logging
from collections.abc import Callable, Iterator
from typing import Any

from neo4j import ManagedTransaction

from graphrag.graph.neo4j_store import Neo4jStore
from graphrag.ingestion.sql.gazetteer import DISTRICT_ALIASES
from graphrag.ingestion.sql.normalizer import name_key
from graphrag.models import TimeFilter
from graphrag.models.social import PostRecord, SqlCorpus

logger = logging.getLogger(__name__)

BATCH = 500
PREVIEW = 220
SOCIAL_LABELS = ("Post", "Topic", "District", "Thana", "PoliceUnit", "Category", "SubCategory", "Account",
                 "Hashtag", "SocialEntity")
SCHEMA = [
    *(f"CREATE CONSTRAINT social_{label.lower()}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
      for label in SOCIAL_LABELS),
    "CREATE INDEX post_posted_at IF NOT EXISTS FOR (n:Post) ON (n.posted_at)",
    "CREATE INDEX post_platform IF NOT EXISTS FOR (n:Post) ON (n.platform)",
    "CREATE INDEX post_sentiment IF NOT EXISTS FOR (n:Post) ON (n.sentiment)",
]

# PostRecord field -> (target label, relationship). Labels/types can't be Cypher parameters, so they come
# from this fixed whitelist only.
LINKS: dict[str, tuple[str, str]] = {
    "topic": ("Topic", "PART_OF"),
    "account": ("Account", "POSTED_BY"),
    "mentions": ("Account", "MENTIONS_ACCOUNT"),
    "districts": ("District", "IN_DISTRICT"),
    "thanas": ("Thana", "IN_THANA"),
    "categories": ("Category", "IN_CATEGORY"),
    "sub_categories": ("SubCategory", "IN_SUBCATEGORY"),
    "entities": ("SocialEntity", "MENTIONS"),
    "hashtags": ("Hashtag", "TAGGED"),
}

# Aggregation dimensions: name -> (label, relationship, SocialEntity kind) or a Post property
NODE_DIMENSIONS: dict[str, tuple[str, str, str | None]] = {
    "district": ("District", "IN_DISTRICT", None),
    "thana": ("Thana", "IN_THANA", None),
    "category": ("Category", "IN_CATEGORY", None),
    "sub_category": ("SubCategory", "IN_SUBCATEGORY", None),
    "account": ("Account", "POSTED_BY", None),
    "mentioned_account": ("Account", "MENTIONS_ACCOUNT", None),
    "hashtag": ("Hashtag", "TAGGED", None),
    "topic": ("Topic", "PART_OF", None),
    "person": ("SocialEntity", "MENTIONS", "person"),
    "organisation": ("SocialEntity", "MENTIONS", "organisation"),
    "location": ("SocialEntity", "MENTIONS", "location"),
    "incident": ("SocialEntity", "MENTIONS", "incident"),
}
PROPERTY_DIMENSIONS: dict[str, str] = {
    "sentiment": "p.sentiment", "platform": "p.platform", "emotion": "p.emotion", "date": "p.date",
    "language": "p.language",
}

# Post filters ($district etc. = null means "any"); dates are RFC 3339 strings, so string order = time order
POST_WHERE = (
    "($district IS NULL OR $district IN p.districts) AND ($thana IS NULL OR $thana IN p.thanas) "
    "AND ($platform IS NULL OR p.platform = $platform) AND ($sentiment IS NULL OR p.sentiment = $sentiment) "
    "AND ($category IS NULL OR $category IN p.categories) "
    "AND ($sub_category IS NULL OR $sub_category IN p.sub_categories) "
    "AND ($start IS NULL OR p.posted_at >= $start) AND ($end IS NULL OR p.posted_at <= $end)"
)
FILTER_KEYS = ("district", "thana", "platform", "sentiment", "category", "sub_category")
_SEEDED = "(size($ids) = 0 OR EXISTS { MATCH (p)-->(s) WHERE s.id IN $ids })"
_POST = "{post_id: q.id, row_id: q.row_id, author: q.author, date: q.date, platform: q.platform, " \
        "evidence: coalesce(q.summary, q.preview)}"
_FACT = (
    "RETURN p.id AS post_id, p.row_id AS row_id, p.author AS author, p.date AS date, p.platform AS platform, "
    "coalesce(p.summary, p.preview) AS evidence, type(r) AS predicate, r.stance AS stance, "
    "n.id AS object_id, n.name AS object, labels(n)[0] AS label, n.kind AS kind"
)

TEMPLATES: dict[str, str] = {
    # posts that link to the entities named in the question (newest first)
    "entity_posts": (
        "MATCH (n) WHERE n.id IN $ids MATCH (p:Post)-[r]->(n) "
        f"WHERE {POST_WHERE} WITH p, r, n ORDER BY p.posted_at DESC LIMIT $limit {_FACT}"
    ),
    # everything a set of posts links to: topic, district, thana, category, people, accounts, hashtags
    "post_context": f"MATCH (p:Post)-[r]->(n) WHERE p.id IN $post_ids AND NOT n:Post WITH p, r, n LIMIT $limit {_FACT}",
    # 2 hops: what else appears in the posts that mention the query entities
    "co_mentions": (
        "MATCH (s) WHERE s.id IN $ids "
        "MATCH (s)<--(p:Post)-[r:MENTIONS|MENTIONS_ACCOUNT|POSTED_BY|TAGGED|STANCE|PART_OF]->(m) "
        f"WHERE m <> s AND NOT m.id IN $ids AND {POST_WHERE} "
        "WITH s, m, type(r) AS rel, collect(DISTINCT p) AS posts ORDER BY size(posts) DESC, m.name LIMIT $limit "
        "RETURN s.id AS subject_id, s.name AS subject, labels(s)[0] AS s_label, s.kind AS s_kind, "
        "m.id AS object_id, m.name AS object, labels(m)[0] AS label, m.kind AS kind, rel, size(posts) AS n, "
        f"[q IN posts[..5] | {_POST}] AS samples"
    ),
    # stance of posts towards the query entities (sentiment_entities): exact counts per stance + sample posts.
    # Separate from entity_posts because stance edges are rare and would be crowded out by plain mentions.
    "stance_counts": (
        "MATCH (p:Post)-[r:STANCE]->(n) WHERE n.id IN $ids "
        f"AND {POST_WHERE} WITH n, r.stance AS stance, collect(DISTINCT p) AS posts ORDER BY size(posts) DESC "
        "RETURN n.id AS id, n.name AS name, n.kind AS kind, stance, size(posts) AS n, "
        "[q IN posts[..3] | q.id] AS posts"
    ),
    # newest posts matching the filters (graph mode with filters but no named entity)
    "filtered_posts": f"MATCH (p:Post) WHERE {POST_WHERE} RETURN p.id AS id ORDER BY p.posted_at DESC LIMIT $limit",
}


def _batches(rows: list[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(rows), BATCH):
        yield rows[start : start + BATCH]


def _post_props(p: PostRecord) -> dict[str, Any]:
    return {
        "id": p.post_id, "row_id": p.row_id, "preview": p.text[:PREVIEW], "summary": (p.summary or "")[:PREVIEW] or None,
        "url": p.url, "platform": p.platform, "author": p.author, "posted_at": p.iso_time, "date": p.date,
        "sentiment": p.sentiment, "emotion": p.emotion, "language": p.language,
        "districts": [d.name for d in p.districts], "thanas": [t.name for t in p.thanas],
        "categories": [c.name for c in p.categories], "sub_categories": [s.name for s in p.sub_categories],
        "topic": p.topic.name if p.topic else None,
    }


def filter_params(filters: dict[str, str], time_range: TimeFilter | None) -> dict[str, Any]:
    params: dict[str, Any] = {key: filters.get(key) for key in FILTER_KEYS}
    params["start"] = f"{time_range.start.isoformat()}T00:00:00Z" if time_range and time_range.start else None
    params["end"] = f"{time_range.end.isoformat()}T23:59:59Z" if time_range and time_range.end else None
    return params


class SocialGraphStore:
    def __init__(self, store: Neo4jStore) -> None:
        self.driver = store.driver

    def ensure_schema(self) -> None:
        with self.driver.session() as session:
            for statement in SCHEMA:
                session.run(statement).consume()

    # ---------- writes ----------

    def write(self, corpus: SqlCorpus, progress: Callable[[str], None] | None = None) -> None:
        self.ensure_schema()
        self._write_hierarchy(corpus)
        posts = corpus.posts
        for done, batch in enumerate(_batches([self._post_row(p) for p in posts]), start=1):
            with self.driver.session() as session:
                session.execute_write(self._write_posts, batch)
            if progress:
                progress(f"graph: posts {min(done * BATCH, len(posts))}/{len(posts)}")
        stance_rows = [s.model_dump() for s in corpus.stances]
        with self.driver.session() as session:
            for rows in _batches(stance_rows):
                session.run(
                    "UNWIND $rows AS row MATCH (p:Post {id: row.post_id}) "
                    "MERGE (e:SocialEntity {id: row.entity.id}) ON CREATE SET e.name = row.entity.name, e.kind = row.entity.kind "
                    "MERGE (p)-[r:STANCE]->(e) SET r.stance = row.stance, r.confidence = row.confidence",
                    rows=rows,
                ).consume()
            # nodes no post links to any more (e.g. after a normaliser fix) would be dead ends in the graph
            session.run(
                "MATCH (n) WHERE (n:Account OR n:Hashtag OR n:SocialEntity OR n:Topic) AND NOT (n)--() DELETE n"
            ).consume()

    @staticmethod
    def _post_row(p: PostRecord) -> dict[str, Any]:
        row: dict[str, Any] = {"id": p.post_id, "props": _post_props(p)}
        for field in LINKS:
            value = getattr(p, field)
            refs = value if isinstance(value, list) else [value] if value else []
            row[field] = [r.model_dump() for r in refs]
        row["thana_districts"] = [
            {"thana": t.id, "district": p.districts[0].id} for t in p.thanas if p.districts
        ]
        return row

    @staticmethod
    def _write_posts(tx: ManagedTransaction, rows: list[dict[str, Any]]) -> None:
        tx.run("UNWIND $rows AS row MERGE (p:Post {id: row.id}) SET p = row.props", rows=rows).consume()
        # rebuild the posts' links (STANCE edges are rewritten after the posts)
        tx.run("MATCH (p:Post)-[r]->() WHERE p.id IN $ids DELETE r", ids=[r["id"] for r in rows]).consume()
        for field, (label, rel) in LINKS.items():
            tx.run(
                f"UNWIND $rows AS row MATCH (p:Post {{id: row.id}}) UNWIND row.{field} AS it "
                f"MERGE (n:{label} {{id: it.id}}) ON CREATE SET n.name = it.name "
                "SET n.kind = coalesce(n.kind, it.kind) "
                f"MERGE (p)-[:{rel}]->(n)",
                rows=rows,
            ).consume()
        tx.run(
            "UNWIND $rows AS row UNWIND row.thana_districts AS td "
            "MATCH (t:Thana {id: td.thana}), (d:District {id: td.district}) MERGE (t)-[:IN_DISTRICT]->(d)",
            rows=rows,
        ).consume()

    def _write_hierarchy(self, corpus: SqlCorpus) -> None:
        districts = [
            {"id": f"district:{name_key(name)}", "name": name, "aliases": aliases}
            for name, aliases in DISTRICT_ALIASES.items()
        ]
        h = corpus.hierarchy
        subs = [{**s, "aliases": h.keyword_aliases.get(str(s["id"]), [])} for s in h.sub_categories]
        with self.driver.session() as session:
            session.run(
                "UNWIND $rows AS row MERGE (d:District {id: row.id}) SET d.name = row.name, d.aliases = row.aliases",
                rows=districts,
            ).consume()
            session.run(
                "UNWIND $rows AS row MERGE (d:District {id: row.district_id}) ON CREATE SET d.name = row.district "
                "MERGE (t:Thana {id: row.id}) ON CREATE SET t.name = row.name SET t.kind = row.district "
                "MERGE (t)-[:IN_DISTRICT]->(d) "
                "WITH d, row WHERE row.unit_id IS NOT NULL "
                "MERGE (u:PoliceUnit {id: row.unit_id}) SET u.name = row.unit, u.kind = row.unit_kind "
                "MERGE (d)-[:IN_UNIT]->(u)",
                rows=h.thanas,
            ).consume()
            session.run(
                "UNWIND $rows AS row MERGE (s:SubCategory {id: row.id}) "
                "SET s.name = row.name, s.hint = row.hint, s.aliases = row.aliases "
                "WITH s, row WHERE row.category_id IS NOT NULL "
                "MERGE (c:Category {id: row.category_id}) ON CREATE SET c.name = row.category "
                "MERGE (s)-[:UNDER]->(c)",
                rows=subs,
            ).consume()

    # ---------- reads ----------

    def post_count(self) -> int:
        records, _, _ = self.driver.execute_query("MATCH (p:Post) RETURN count(p) AS n")
        return int(records[0]["n"])

    def name_index(self) -> list[dict[str, Any]]:
        """Every nameable node a question can refer to, with aliases and how many posts link to it."""
        records, _, _ = self.driver.execute_query(
            "MATCH (n) WHERE n:SocialEntity OR n:Account OR n:Hashtag OR n:District OR n:Thana "
            "OR n:Category OR n:SubCategory "
            "RETURN n.id AS id, n.name AS name, labels(n)[0] AS label, n.kind AS kind, "
            "coalesce(n.aliases, []) AS aliases, COUNT { (n)<--(:Post) } AS posts"
        )
        return [dict(r) for r in records]

    def run(self, template: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        records, _, _ = self.driver.execute_query(TEMPLATES[template], params)
        return [dict(r) for r in records]

    def aggregate(self, dimension: str, params: dict[str, Any], top: int) -> list[dict[str, Any]]:
        """Top groups by number of matching posts, each with 2 sample posts (citable evidence)."""
        if dimension in NODE_DIMENSIONS:
            label, rel, kind = NODE_DIMENSIONS[dimension]
            query = (
                f"MATCH (p:Post)-[:{rel}]->(x:{label}) WHERE {POST_WHERE} AND ($kind IS NULL OR x.kind = $kind) "
                f"AND {_SEEDED} WITH x, count(DISTINCT p) AS n, collect(DISTINCT p)[..2] AS sample "
                "ORDER BY n DESC, x.name LIMIT $top "
                f"RETURN x.id AS id, x.name AS name, n, [q IN sample | {_POST}] AS samples"
            )
            records, _, _ = self.driver.execute_query(query, {**params, "kind": kind, "top": top})
        elif dimension in PROPERTY_DIMENSIONS:
            prop = PROPERTY_DIMENSIONS[dimension]
            query = (
                f"MATCH (p:Post) WHERE {POST_WHERE} AND {prop} IS NOT NULL AND {_SEEDED} "
                f"WITH {prop} AS key, count(p) AS n, collect(p)[..2] AS sample ORDER BY n DESC, key LIMIT $top "
                f"RETURN '{dimension}:' + key AS id, key AS name, n, [q IN sample | {_POST}] AS samples"
            )
            records, _, _ = self.driver.execute_query(query, {**params, "top": top})
        else:
            raise ValueError(f"unknown aggregation dimension {dimension!r}")
        return [dict(r) for r in records]

    def total(self, params: dict[str, Any]) -> int:
        records, _, _ = self.driver.execute_query(
            f"MATCH (p:Post) WHERE {POST_WHERE} AND {_SEEDED} RETURN count(p) AS n", params
        )
        return int(records[0]["n"])

    def post_entity_mentions(self, post_ids: list[str], entity_ids: list[str]) -> dict[str, int]:
        if not post_ids or not entity_ids:
            return {}
        records, _, _ = self.driver.execute_query(
            "MATCH (p:Post)-->(n) WHERE p.id IN $posts AND n.id IN $ids RETURN p.id AS id, count(DISTINCT n) AS n",
            posts=post_ids, ids=entity_ids,
        )
        return {r["id"]: r["n"] for r in records}

    def counts(self) -> dict[str, int]:
        records, _, _ = self.driver.execute_query(
            "UNWIND $labels AS label CALL (label) { MATCH (n) WHERE label IN labels(n) RETURN count(n) AS n } "
            "RETURN label, n",
            labels=list(SOCIAL_LABELS),
        )
        return {r["label"]: r["n"] for r in records}
