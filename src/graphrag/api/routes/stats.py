from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from graphrag.api.deps import get_neo4j, get_qdrant
from graphrag.config import get_settings
from graphrag.graph.neo4j_store import Neo4jStore
from graphrag.vector_store.qdrant_store import QdrantStore

router = APIRouter(tags=["stats"])


@router.get("/stats")
def stats(
    request: Request,
    qdrant: Annotated[QdrantStore, Depends(get_qdrant)],
    neo4j: Annotated[Neo4jStore, Depends(get_neo4j)],
) -> dict[str, Any]:
    """Node/edge counts by label, Qdrant point counts (documents + SQL posts), and the ingested documents."""
    counts = neo4j.counts()
    social = getattr(request.app.state, "social_vectors", None)
    sql_points = 0
    if social is not None and social.client.collection_exists(social.collection_name):
        sql_points = social.count()
    return {
        "sql": {"collection": get_settings().sql_collection_name, "points": sql_points,
                "posts": counts.get("node:Post", 0)},
        "qdrant": {"collection": get_settings().collection_name, "points": qdrant.count()},
        "neo4j": {
            "nodes": {k.removeprefix("node:"): v for k, v in counts.items() if k.startswith("node:")},
            "relationships": {k.removeprefix("rel:"): v for k, v in counts.items() if k.startswith("rel:")},
        },
        "documents": neo4j.document_sources(),
    }
