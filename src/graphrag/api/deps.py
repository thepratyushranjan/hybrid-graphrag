from fastapi import HTTPException, Request, status

from graphrag.graph.neo4j_store import Neo4jStore
from graphrag.ingestion.pipeline import IngestionPipeline
from graphrag.vector_store.qdrant_store import QdrantStore


def get_qdrant(request: Request) -> QdrantStore:
    return request.app.state.qdrant


def get_neo4j(request: Request) -> Neo4jStore:
    return request.app.state.neo4j


def get_pipeline(request: Request) -> IngestionPipeline:
    pipeline: IngestionPipeline | None = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ingestion unavailable: {request.app.state.pipeline_error}",
        )
    return pipeline
