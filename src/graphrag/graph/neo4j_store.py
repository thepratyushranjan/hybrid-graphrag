"""Neo4j writes: lexical graph (Document, Chunk, PART_OF, NEXT) + entity graph (Entity, MENTIONS, RELATES_TO).

All writes MERGE on `id` only and then SET properties (merging on several properties is the classic
cause of duplicate nodes), so re-ingesting a document leaves node and edge counts unchanged.
"""

import logging
from collections.abc import Iterator
from typing import Any

from neo4j import Driver, GraphDatabase, ManagedTransaction

from graphrag.config import Settings
from graphrag.models import ENTITY_TYPES, Chunk, ChunkGraph, GraphEntity, SimilarDocument

logger = logging.getLogger(__name__)

BATCH = 500
TEXT_PREVIEW = 300

SCHEMA = [
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
    "CREATE INDEX chunk_source IF NOT EXISTS FOR (n:Chunk) ON (n.source)",
    "CREATE INDEX document_source IF NOT EXISTS FOR (n:Document) ON (n.source)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.type)",
    "CREATE INDEX relates_predicate IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.predicate)",
    "CREATE INDEX relates_chunk IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.chunk_id)",
    # Full-text over names + aliases, used to match query words to nodes (Milestone 4).
    # standard-folding: Unicode word splitting (works for Devanagari) + case/accent folding.
    "CREATE FULLTEXT INDEX entity_names IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.alias_text] "
    "OPTIONS {indexConfig: {`fulltext.analyzer`: 'standard-folding'}}",
]


def _batches(rows: list[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(rows), BATCH):
        yield rows[start : start + BATCH]


class Neo4jStore:
    def __init__(self, settings: Settings) -> None:
        self.driver: Driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            # an empty database warns that MENTIONS / aliases "don't exist yet"; that is expected, not an error
            notifications_min_severity="OFF",
        )

    def ping(self) -> None:
        self.driver.verify_connectivity()

    def close(self) -> None:
        self.driver.close()

    def ensure_schema(self) -> None:
        with self.driver.session() as session:
            for statement in SCHEMA:
                session.run(statement).consume()

    def has_source(self, source: str) -> bool:
        records, _, _ = self.driver.execute_query(
            "MATCH (d:Document {source: $source}) RETURN count(d) > 0 AS found", source=source
        )
        return bool(records[0]["found"])

    def known_entities(self, limit: int) -> list[GraphEntity]:
        """Most-mentioned existing entities, so the extractor reuses their names across documents."""
        records, _, _ = self.driver.execute_query(
            "MATCH (e:Entity) OPTIONAL MATCH (e)<-[m:MENTIONS]-() "
            "WITH e, count(m) AS mentions ORDER BY mentions DESC, e.name LIMIT $limit "
            "RETURN e.id AS id, e.name AS name, e.type AS type, coalesce(e.aliases, []) AS aliases",
            limit=limit,
        )
        return [GraphEntity.model_validate(dict(r)) for r in records if r["type"] in ENTITY_TYPES]

    def write_document(
        self,
        doc_id: str,
        source: str,
        doc_type: str,
        chunks: list[Chunk],
        graphs: list[ChunkGraph],
        similar: list[SimilarDocument] | None = None,
    ) -> None:
        """Replace everything stored for `source` with this version of the document, in one transaction."""
        chunk_rows = [
            {
                "id": c.chunk_id, "doc_id": c.doc_id, "source": c.source, "chunk_index": c.chunk_index,
                "page": c.page, "section": c.section, "language": c.language,
                "extraction_method": c.extraction_method, "token_count": c.token_count,
                "text_preview": c.text[:TEXT_PREVIEW],
            }
            for c in chunks
        ]
        next_rows = [{"a": a.chunk_id, "b": b.chunk_id} for a, b in zip(chunks, chunks[1:], strict=False)]
        entities: dict[str, GraphEntity] = {e.id: e for g in graphs for e in g.entities}
        mention_rows = [{"chunk_id": g.chunk_id, "entity_id": e.id} for g in graphs for e in g.entities]
        relation_rows = [r.model_dump() for g in graphs for r in g.relations]

        def tx_write(tx: ManagedTransaction) -> None:
            # 1. Remove the previous version of this file: its chunks, their mentions and relations
            tx.run(
                "MATCH ()-[r:RELATES_TO]->() WHERE r.chunk_id IN "
                "COLLECT { MATCH (c:Chunk {source: $source}) RETURN c.id } + $chunk_ids DELETE r",
                source=source, chunk_ids=[c["id"] for c in chunk_rows],
            ).consume()
            tx.run(
                "MATCH (c:Chunk {source: $source}) WHERE NOT c.id IN $chunk_ids DETACH DELETE c",
                source=source, chunk_ids=[c["id"] for c in chunk_rows],
            ).consume()
            tx.run("MATCH (c:Chunk {source: $source})-[m:MENTIONS]->() DELETE m", source=source).consume()
            tx.run("MATCH (d:Document {source: $source}) WHERE d.id <> $doc_id DETACH DELETE d",
                   source=source, doc_id=doc_id).consume()

            # 2. Lexical graph
            tx.run(
                "MERGE (d:Document {id: $doc_id}) SET d.source = $source, d.doc_type = $doc_type, "
                "d.chunks = $n, d.status = 'processing'",
                doc_id=doc_id, source=source, doc_type=doc_type, n=len(chunk_rows),
            ).consume()
            for rows in _batches(chunk_rows):
                tx.run(
                    "UNWIND $rows AS row MERGE (c:Chunk {id: row.id}) SET c += row "
                    "WITH c, row MATCH (d:Document {id: row.doc_id}) MERGE (c)-[r:PART_OF]->(d) SET r.source = 'structured'",
                    rows=rows,
                ).consume()
            for rows in _batches(next_rows):
                tx.run(
                    "UNWIND $rows AS row MATCH (a:Chunk {id: row.a}), (b:Chunk {id: row.b}) "
                    "MERGE (a)-[r:NEXT]->(b) SET r.source = 'structured'",
                    rows=rows,
                ).consume()

            # 3. Entities: one statement per type, since labels can't be query parameters
            for entity_type in ENTITY_TYPES:
                rows = [e.model_dump() for e in entities.values() if e.type == entity_type]
                for batch in _batches(rows):
                    tx.run(
                        f"UNWIND $rows AS row MERGE (e:Entity {{id: row.id}}) SET e:{entity_type} "
                        "SET e.name = coalesce(e.name, row.name), e.type = row.type, "
                        "e.aliases = [a IN coalesce(e.aliases, []) WHERE NOT a IN row.aliases] + row.aliases "
                        "SET e.alias_text = reduce(s = '', a IN e.aliases | s + ' ' + a)",
                        rows=batch,
                    ).consume()

            # 4. Chunk -[:MENTIONS]-> Entity, Entity -[:RELATES_TO {predicate, chunk_id}]-> Entity
            for rows in _batches(mention_rows):
                tx.run(
                    "UNWIND $rows AS row MATCH (c:Chunk {id: row.chunk_id}), (e:Entity {id: row.entity_id}) "
                    "MERGE (c)-[r:MENTIONS]->(e) SET r.source = 'llm'",
                    rows=rows,
                ).consume()
            for rows in _batches(relation_rows):
                tx.run(
                    "UNWIND $rows AS row MATCH (s:Entity {id: row.subject_id}), (o:Entity {id: row.object_id}) "
                    "MERGE (s)-[r:RELATES_TO {predicate: row.predicate, chunk_id: row.chunk_id}]->(o) "
                    "SET r.evidence = row.evidence, r.confidence = row.confidence, r.source = 'llm', "
                    "r.original_predicate = row.original_predicate",
                    rows=rows,
                ).consume()

            # 5. Integrity: relations must point at an existing chunk (their evidence), and entities
            #    must still be mentioned somewhere (e.g. after a file was edited)
            tx.run(
                "MATCH ()-[r:RELATES_TO]->() WHERE NOT EXISTS { MATCH (c:Chunk {id: r.chunk_id}) } DELETE r"
            ).consume()
            tx.run("MATCH (e:Entity) WHERE NOT (e)<-[:MENTIONS]-() DETACH DELETE e").consume()
            # 6. Document similarity (recomputed on every ingest; undirected MERGE avoids A->B plus B->A)
            tx.run("MATCH (:Document {id: $doc_id})-[r:SIMILAR_TO]-() DELETE r", doc_id=doc_id).consume()
            if similar:
                tx.run(
                    "UNWIND $rows AS row MATCH (d:Document {id: $doc_id}), (o:Document {id: row.doc_id}) "
                    "MERGE (d)-[r:SIMILAR_TO]-(o) SET r.score = row.score, r.source = 'vector'",
                    rows=[s.model_dump() for s in similar], doc_id=doc_id,
                ).consume()
            tx.run("MATCH (d:Document {id: $doc_id}) SET d.status = 'complete'", doc_id=doc_id).consume()

        with self.driver.session() as session:
            session.execute_write(tx_write)

    def counts(self) -> dict[str, int]:
        records, _, _ = self.driver.execute_query(
            "CALL () { MATCH (n) UNWIND labels(n) AS label RETURN 'node:' + label AS k, count(*) AS v "
            "UNION ALL MATCH ()-[r]->() RETURN 'rel:' + type(r) AS k, count(*) AS v } RETURN k, v ORDER BY k"
        )
        return {r["k"]: r["v"] for r in records}
