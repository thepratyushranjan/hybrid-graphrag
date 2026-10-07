"""Step 2 (graph branch): Cypher templates chosen by query type.

Every template returns facts (subject, predicate, object) plus the chunk_ids that support them,
so the graph can pull in text that vector search missed (step 3, the graph-to-vector bridge).

Hub pruning: never expand *through* an entity with more than HUB_DEGREE_LIMIT edges (it connects
everything to everything), and every template has a LIMIT.
"""

import logging
from typing import Any

from graphrag.config import Settings
from graphrag.graph.neo4j_store import Neo4jStore
from graphrag.models import Aggregate, GraphFact, QueryAnalysis

logger = logging.getLogger(__name__)

# $allowed = null (no filter) or the chunk IDs a payload filter allows; facts must come from those chunks
_SCOPE = "($allowed IS NULL OR r.chunk_id IN $allowed)"

_FACT = (
    "RETURN DISTINCT startNode(r).id AS subject_id, startNode(r).name AS subject, r.predicate AS predicate, "
    "endNode(r).id AS object_id, endNode(r).name AS object, r.confidence AS confidence, "
    "startNode(r).type AS subject_type, endNode(r).type AS object_type, "
    "COLLECT { MATCH (c:Chunk {id: r.chunk_id}) RETURN c.source } AS sources, "
    "r.chunk_id AS chunk_id, r.evidence AS evidence"
)

TEMPLATES: dict[str, str] = {
    # "Tell me about X": every relation touching the query entities
    "neighbourhood": (
        f"MATCH (e:Entity)-[r:RELATES_TO]-(n:Entity) WHERE e.id IN $ids AND {_SCOPE} "
        f"WITH r, 1 AS hops {_FACT}, hops ORDER BY confidence DESC LIMIT $limit"
    ),
    # Indirect links: X - m - n, never expanding through hubs
    "two_hop": (
        "MATCH (e:Entity)-[:RELATES_TO]-(m:Entity)-[r:RELATES_TO]-(n:Entity) "
        "WHERE e.id IN $ids AND n <> e AND NOT m.id IN $ids AND COUNT { (m)--() } <= $hub "
        f"AND {_SCOPE} WITH r, 2 AS hops {_FACT}, hops ORDER BY confidence DESC LIMIT $limit"
    ),
    # "How is X related to Y?": shortest paths (max 3 hops) between each pair of query entities
    "path": (
        "MATCH (a:Entity), (b:Entity) WHERE a.id IN $ids AND b.id IN $ids AND a.id < b.id "
        "MATCH p = allShortestPaths((a)-[:RELATES_TO*..3]-(b)) "
        "WHERE all(x IN nodes(p)[1..-1] WHERE COUNT { (x)--() } <= $hub) "
        "WITH p LIMIT 10 UNWIND relationships(p) AS r "
        f"WITH r, length(p) AS hops WHERE {_SCOPE} WITH r, hops {_FACT}, hops LIMIT $limit"
    ),
    # Things connected to *all* query entities (e.g. a vendor linked to both a subsidiary and a clause)
    "intersection": (
        f"MATCH (n:Entity)-[r:RELATES_TO]-(e:Entity) WHERE e.id IN $ids AND NOT n.id IN $ids AND {_SCOPE} "
        "WITH n, collect(DISTINCT e.id) AS linked, collect(r) AS rels WHERE size(linked) = size($ids) "
        f"UNWIND rels AS r WITH r, 1 AS hops {_FACT}, hops LIMIT $limit"
    ),
}

# Counts and rankings ("which X has the most ...") - something vector search cannot do.
# With a predicate the edge direction matters: "X SUPPLIES the most" counts X's outgoing SUPPLIES edges.
AGGREGATION = (
    "MATCH (x:Entity)-[r:RELATES_TO]-(y:Entity) "
    "WHERE ($predicate IS NULL OR (r.predicate = $predicate AND startNode(r) = x)) "
    "AND ($type IS NULL OR x.type = $type) "
    "AND (size($ids) = 0 OR y.id IN $ids) AND NOT x.id IN $ids "
    f"AND {_SCOPE} "
    "WITH x, count(DISTINCT y) AS n, collect(r)[..5] AS rels ORDER BY n DESC, x.name LIMIT 10 "
    "RETURN x.id AS entity_id, x.name AS name, n AS count, "
    "[r IN rels | {subject_id: startNode(r).id, subject: startNode(r).name, predicate: r.predicate, "
    "object_id: endNode(r).id, object: endNode(r).name, confidence: r.confidence, chunk_id: r.chunk_id, "
    "evidence: r.evidence, subject_type: startNode(r).type, object_type: endNode(r).type, "
    "sources: COLLECT { MATCH (c:Chunk {id: r.chunk_id}) RETURN c.source }}] AS facts"
)


def choose_templates(analysis: QueryAnalysis, hops: int) -> list[str]:
    ids = analysis.entity_ids
    chosen: list[str] = []
    if ids:
        chosen.append("neighbourhood")
    if ids and (hops >= 2 or analysis.query_type == "relationship"):
        chosen.append("two_hop")
    if len(ids) >= 2:
        chosen += ["path", "intersection"]
    if analysis.query_type == "aggregation":
        chosen.append("aggregation")
    return chosen


class GraphRetriever:
    def __init__(self, settings: Settings, graph: Neo4jStore) -> None:
        self.settings = settings
        self.graph = graph

    def retrieve(self, analysis: QueryAnalysis, hops: int) -> tuple[list[GraphFact], list[Aggregate], list[str]]:
        templates = choose_templates(analysis, hops)
        params: dict[str, Any] = {
            "ids": analysis.entity_ids,
            "hub": self.settings.hub_degree_limit,
            "limit": self.settings.graph_fact_limit,
            "allowed": (
                self.graph.chunk_ids_matching(analysis.filters, analysis.time_range)
                if analysis.filters or analysis.time_range
                else None
            ),
        }
        facts: list[GraphFact] = []
        aggregates: list[Aggregate] = []
        for name in templates:
            if name == "aggregation":
                aggs, agg_facts = self._aggregate(analysis, params)
                aggregates += aggs
                facts += agg_facts
                continue
            records, _, _ = self.graph.driver.execute_query(TEMPLATES[name], params)
            facts += [GraphFact(**_fact_fields(dict(r)), template=name) for r in records]
        return facts, aggregates, templates

    def _aggregate(self, analysis: QueryAnalysis, params: dict[str, Any]) -> tuple[list[Aggregate], list[GraphFact]]:
        records, _, _ = self.graph.driver.execute_query(
            AGGREGATION, params, predicate=analysis.predicate, type=analysis.target_type
        )
        aggregates = [Aggregate(name=r["name"], entity_id=r["entity_id"], count=r["count"]) for r in records]
        facts = [
            GraphFact(**_fact_fields({**f, "hops": 1}), template="aggregation") for r in records for f in r["facts"]
        ]
        return aggregates, facts


def _fact_fields(row: dict[str, Any]) -> dict[str, Any]:
    chunk_id = row.pop("chunk_id")
    return {**row, "chunk_ids": [chunk_id] if chunk_id else [], "confidence": row.get("confidence") or 0.0}
