from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from graphrag.api.deps import get_neo4j, get_qdrant
from graphrag.config import get_settings
from graphrag.graph.neo4j_store import Neo4jStore
from graphrag.vector_store.qdrant_store import QdrantStore

router = APIRouter(tags=["health"])


def _check(ping: object) -> str:
    try:
        ping()  # type: ignore[operator]
        return "ok"
    except Exception as exc:  # noqa: BLE001 - report any connection failure
        return f"error: {exc.__class__.__name__}: {exc}"


@router.get("/health")
def health(
    response: Response,
    qdrant: Annotated[QdrantStore, Depends(get_qdrant)],
    neo4j: Annotated[Neo4jStore, Depends(get_neo4j)],
) -> dict:
    settings = get_settings()
    checks = {"qdrant": _check(qdrant.ping), "neo4j": _check(neo4j.ping)}
    healthy = all(v == "ok" for v in checks.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ok" if healthy else "degraded",
        "app": settings.app_name,
        "env": settings.app_env,
        "checks": checks,
        "llm": {"provider": settings.llm_provider, "model": settings.llm_model},
        "embedding": {"provider": settings.embedding_provider, "model": settings.embedding_model},
        "warnings": settings.provider_warnings(),
    }
