from typing import Annotated, Any

from fastapi import APIRouter, Depends

from graphrag.api.deps import get_neo4j, get_qdrant
from graphrag.config import get_settings
from graphrag.graph.neo4j_store import Neo4jStore
from graphrag.vector_store.qdrant_store import QdrantStore

router = APIRouter(tags=["stats"])


@router.get("/stats")
def stats(
    qdrant: Annotated[QdrantStore, Depends(get_qdrant)], neo4j: Annotated[Neo4jStore, Depends(get_neo4j)]
) -> dict[str, Any]:
    """Node/edge counts by label, Qdrant point count, and the ingested documents."""
    counts = neo4j.counts()
    return {
        "qdrant": {"collection": get_settings().collection_name, "points": qdrant.count()},
        "neo4j": {
            "nodes": {k.removeprefix("node:"): v for k, v in counts.items() if k.startswith("node:")},
            "relationships": {k.removeprefix("rel:"): v for k, v in counts.items() if k.startswith("rel:")},
        },
        "documents": neo4j.document_sources(),
    }
