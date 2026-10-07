from fastapi import HTTPException, Request, status

from graphrag.graph.neo4j_store import Neo4jStore
from graphrag.api.jobs import JobStore
from graphrag.generation.synthesizer import AnswerGenerator
from graphrag.ingestion.pipeline import IngestionPipeline
from graphrag.retrieval.hybrid import HybridRetriever
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


def get_retriever(request: Request) -> HybridRetriever:
    retriever: HybridRetriever | None = request.app.state.retriever
    if retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Retrieval unavailable: {request.app.state.pipeline_error}",
        )
    return retriever


def get_generator(request: Request) -> AnswerGenerator:
    generator: AnswerGenerator | None = request.app.state.generator
    if generator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Query unavailable: {request.app.state.pipeline_error}",
        )
    return generator


def get_jobs(request: Request) -> JobStore:
    jobs: JobStore | None = request.app.state.jobs
    if jobs is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ingestion unavailable: {request.app.state.pipeline_error}",
        )
    return jobs
