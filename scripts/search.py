"""Plain vector search against Qdrant (run with `make search q="..."`)."""

import argparse

from graphrag.config import get_settings
from graphrag.embeddings.embedder import build_embedder
from graphrag.vector_store.qdrant_store import QdrantStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--doc-type", choices=["pdf", "md", "txt"])
    parser.add_argument("--language", help="ISO code from langdetect, e.g. en, hi")
    parser.add_argument("--source", help="file name, e.g. incident_bulletin.pdf")
    args = parser.parse_args()

    settings = get_settings()
    filters: dict[str, str | int] = {
        k: v for k, v in {"doc_type": args.doc_type, "language": args.language, "source": args.source}.items() if v
    }
    store = QdrantStore(settings)
    hits = store.search(build_embedder(settings).embed_query(args.query), args.top_k or settings.top_k, filters)
    print(f"Query: {args.query}  filters={filters or '-'}\n")
    for rank, hit in enumerate(hits, start=1):
        c = hit.chunk
        where = f"p{c.page}" if c.page else (c.section or "-")
        print(f"#{rank} score={hit.score:.3f} {c.source} [{where}] lang={c.language} via={c.extraction_method}")
        print("   " + c.text[:220].replace("\n", " ") + ("…" if len(c.text) > 220 else ""))
    store.close()


if __name__ == "__main__":
    main()
