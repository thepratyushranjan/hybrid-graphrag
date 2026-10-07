"""Ingest every PDF / Markdown / text file in data/samples into Qdrant + Neo4j (run with `make seed`).
Idempotent: running it again doesn't duplicate anything."""

import sys
from pathlib import Path

from graphrag.config import get_settings
from graphrag.embeddings.embedder import build_embedder
from graphrag.graph.neo4j_store import Neo4jStore
from graphrag.ingestion.loaders import SUPPORTED_TYPES
from graphrag.ingestion.pipeline import IngestionPipeline
from graphrag.vector_store.qdrant_store import QdrantStore

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


def main() -> int:
    files = sorted(p for p in SAMPLES.glob("*") if p.suffix.lower() in SUPPORTED_TYPES)
    if not files:
        print(f"No sample files in {SAMPLES}")
        return 1

    settings = get_settings()
    vectors, graph = QdrantStore(settings), Neo4jStore(settings)
    vectors.ensure_collection()
    graph.ensure_schema()
    pipeline = IngestionPipeline(settings, build_embedder(settings), vectors, graph_store=graph)
    if pipeline.llm_error:
        print(f"WARNING: LLM unavailable ({pipeline.llm_error}). Loading vectors only; fix .env and run "
              "`make seed` again to build the knowledge graph.")
    for path in files:
        print(f"Ingesting {path.name} ...", flush=True)
        r = pipeline.ingest_path(path)
        print(
            f"  pages={r.pages} chunks={r.chunks} langs={r.languages} methods={r.extraction_methods}\n"
            f"  entities={r.entities} relations={r.relations} dropped={r.dropped_relations} "
            f"graph={r.graph_status} replaced_points={r.replaced_points} ({r.seconds}s)\n"
            f"  similar={r.similar_documents or '-'}"
        )
    print(f"\nQdrant points in '{settings.collection_name}': {vectors.count()}")
    print("Neo4j:", ", ".join(f"{k}={v}" for k, v in graph.counts().items()))
    vectors.close()
    graph.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
