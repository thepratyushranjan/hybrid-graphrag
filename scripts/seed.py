"""Ingest every PDF / Markdown / text file in data/samples (run with `make seed`)."""

import sys
from pathlib import Path

from graphrag.config import get_settings
from graphrag.embeddings.embedder import build_embedder
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
    store = QdrantStore(settings)
    pipeline = IngestionPipeline(settings, build_embedder(settings), store)
    for path in files:
        r = pipeline.ingest_path(path)
        print(
            f"{r.source:<40} pages={r.pages:<3} chunks={r.chunks:<3} langs={r.languages} "
            f"methods={r.extraction_methods} replaced={r.replaced_points} ({r.seconds}s)"
        )
    print(f"\nTotal points in '{settings.collection_name}': {store.count()}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
