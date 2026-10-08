"""SQL dump -> parse -> normalise -> Neo4j social graph + Qdrant `social_posts` vectors.

Idempotent: point IDs are UUID5(post_id) and graph nodes are MERGEd on id, so re-running updates in place.
"""

import logging
import time
from collections.abc import Callable
from pathlib import Path

from graphrag.config import Settings
from graphrag.embeddings.embedder import Embedder
from graphrag.graph.social_store import SocialGraphStore
from graphrag.ingestion.sql.dump_parser import load_tables
from graphrag.ingestion.sql.normalizer import TABLES, normalize
from graphrag.models import Chunk
from graphrag.models.social import PostRecord, SqlIngestResult
from graphrag.vector_store.qdrant_store import SocialPostStore

logger = logging.getLogger(__name__)

EMBED_BATCH = 256


def post_chunk(post: PostRecord, token_count: int) -> Chunk:
    """A post as a Chunk, so fusion, reranking, prompts and citations work unchanged."""
    return Chunk(
        chunk_id=post.post_id,
        doc_id=post.topic.id if post.topic else post.post_id,
        source=post.url or post.post_id,
        doc_type="post",
        chunk_index=0,
        text=post.document_text(),
        token_count=token_count,
        section=post.topic.name if post.topic else None,
        language=post.language,
        date_start=post.iso_time,
        date_end=post.iso_time,
        dates=[post.date] if post.date else [],
        metadata={
            k: v for k, v in {
                "platform": post.platform, "author": post.author, "district": ", ".join(d.name for d in post.districts),
                "category": " / ".join(c.name for c in [*post.categories, *post.sub_categories]),
                "sentiment": post.sentiment, "posted_at": post.iso_time, "url": post.url, "row_id": post.row_id,
            }.items() if v
        },
    )


def post_payload(post: PostRecord) -> dict[str, object]:
    """Filterable payload next to the Chunk fields (keyword-indexed in SocialPostStore)."""
    return {
        "district": [d.name for d in post.districts],
        "thana": [t.name for t in post.thanas],
        "platform": post.platform,
        "sentiment": post.sentiment,
        "broad_category": [c.name for c in post.categories],
        "sub_category": [s.name for s in post.sub_categories],
        "topic_id": post.topic.id if post.topic else None,
        "author": post.author,
        "entity_ids": [e.id for e in post.entities],
    }


class SqlIngestionPipeline:
    def __init__(
        self, settings: Settings, embedder: Embedder, vectors: SocialPostStore, graph: SocialGraphStore
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.vectors = vectors
        self.graph = graph

    def ingest(
        self, source: str | Path | bytes, name: str, progress: Callable[[str], None] | None = None
    ) -> SqlIngestResult:
        report = progress or (lambda stage: logger.info("SQL ingest: %s", stage))
        started = time.perf_counter()

        report("parsing dump")
        tables = load_tables(source, TABLES)
        report("normalising rows")
        corpus = normalize(tables, self.settings.sql_post_chars, self.settings.sql_ingest_limit)
        if not corpus.posts:
            raise ValueError("The dump has no analyzed_data rows with text")
        logger.info("SQL ingest: %d posts, %d stances, skipped %s", len(corpus.posts), len(corpus.stances),
                    corpus.skipped)

        report("writing graph")
        self.graph.write(corpus, report)

        self.vectors.ensure_collection()
        posts = corpus.posts
        for start in range(0, len(posts), EMBED_BATCH):
            batch = posts[start : start + EMBED_BATCH]
            texts = [p.document_text() for p in batch]
            vectors = self.embedder.embed_documents(texts)
            chunks = [post_chunk(p, self.embedder.count_tokens(t)) for p, t in zip(batch, texts, strict=True)]
            self.vectors.upsert(chunks, vectors, [post_payload(p) for p in batch])
            report(f"embedding posts {start + len(batch)}/{len(posts)}")

        counts = self.graph.counts()
        return SqlIngestResult(
            source=name,
            posts=len(posts),
            vectors=self.vectors.count(),
            topics=len({p.topic.id for p in posts if p.topic}),
            accounts=len({a.id for p in posts for a in ([p.account] if p.account else []) + p.mentions}),
            entities=len({e.id for p in posts for e in p.entities}),
            stances=len(corpus.stances),
            skipped=corpus.skipped,
            graph=counts,
            seconds=round(time.perf_counter() - started, 1),
        )
