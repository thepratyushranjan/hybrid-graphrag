"""Load the SQL dump (SQL_DUMP_PATH, or the path given) into Neo4j + the `social_posts` Qdrant collection
(run with `make ingest-sql`). Idempotent: running it again updates in place."""

import sys
from pathlib import Path

from graphrag.config import get_settings
from graphrag.embeddings.embedder import build_embedder
from graphrag.graph.neo4j_store import Neo4jStore
from graphrag.graph.social_store import SocialGraphStore
from graphrag.ingestion.sql.pipeline import SqlIngestionPipeline
from graphrag.vector_store.qdrant_store import SocialPostStore


def main() -> int:
    settings = get_settings()
    path = Path(sys.argv[1] if len(sys.argv) > 1 else settings.sql_dump_path)
    if not path.is_file():
        print(f"No dump at {path}. Copy it to data/sql/ or pass its path.")
        return 1
    neo4j = Neo4jStore(settings)
    pipeline = SqlIngestionPipeline(
        settings, build_embedder(settings), SocialPostStore(settings), SocialGraphStore(neo4j)
    )
    limit = f" (SQL_INGEST_LIMIT={settings.sql_ingest_limit})" if settings.sql_ingest_limit else ""
    print(f"Ingesting {path.name}{limit} ...", flush=True)
    r = pipeline.ingest(str(path), path.name, progress=lambda stage: print(f"  {stage}", flush=True))
    print(
        f"\nposts={r.posts} vectors={r.vectors} topics={r.topics} accounts={r.accounts} entities={r.entities} "
        f"stances={r.stances} ({r.seconds}s)\nskipped={r.skipped or '-'}\ngraph={r.graph}"
    )
    neo4j.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
