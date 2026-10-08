import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from neo4j.exceptions import AuthError, ServiceUnavailable, SessionExpired
from qdrant_client.http.exceptions import ResponseHandlingException

from graphrag.api.jobs import JobStore, SqlJobStore
from graphrag.api.routes import documents, health, ingest, ingest_sql, llm, query, retrieve, stats
from graphrag.config import get_settings
from graphrag.embeddings.embedder import build_embedder
from graphrag.graph.neo4j_store import Neo4jStore
from graphrag.graph.social_store import SocialGraphStore
from graphrag.generation.synthesizer import AnswerGenerator
from graphrag.ingestion.pipeline import IngestionPipeline
from graphrag.ingestion.sql.pipeline import SqlIngestionPipeline
from graphrag.llm.registry import LLMRegistry
from graphrag.retrieval.factory import build_retriever, build_sql_retriever
from graphrag.vector_store.qdrant_store import QdrantStore, SocialPostStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("graphrag")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    for warning in settings.provider_warnings():
        logger.warning(warning)

    app.state.qdrant = QdrantStore(settings)
    app.state.neo4j = Neo4jStore(settings)
    app.state.pipeline = None
    app.state.retriever = None
    app.state.generator = None
    app.state.jobs = None
    app.state.social_vectors = SocialPostStore(settings)
    app.state.sql_retriever = None
    app.state.sql_jobs = None
    app.state.llm_registry = LLMRegistry(settings)
    app.state.pipeline_error = None
    try:
        app.state.qdrant.ensure_collection()
        app.state.neo4j.ensure_schema()
        embedder = build_embedder(settings)
        app.state.pipeline = IngestionPipeline(
            settings, embedder, app.state.qdrant, graph_store=app.state.neo4j, llm=app.state.llm_registry.default
        )
        app.state.retriever = build_retriever(
            settings, embedder, app.state.qdrant, app.state.neo4j, app.state.pipeline.llm
        )
        social_graph = SocialGraphStore(app.state.neo4j)
        social_graph.ensure_schema()
        app.state.social_vectors.ensure_collection()
        app.state.sql_retriever = build_sql_retriever(
            settings, embedder, app.state.social_vectors, social_graph, app.state.retriever.reranker
        )
        app.state.generator = AnswerGenerator(
            settings, app.state.retriever, app.state.pipeline.llm, registry=app.state.llm_registry,
            sql_retriever=app.state.sql_retriever,
        )
        app.state.jobs = JobStore(app.state.pipeline)
        app.state.sql_jobs = SqlJobStore(
            SqlIngestionPipeline(settings, embedder, app.state.social_vectors, social_graph)
        )
        logger.info("Ingestion ready: %s (%d-dim)", settings.embedding_model, settings.embedding_dim)
    except Exception as exc:  # noqa: BLE001 - keep the server up; /ingest reports the reason
        app.state.pipeline_error = f"{exc.__class__.__name__}: {exc}"
        logger.error("Ingestion disabled: %s", app.state.pipeline_error)
    try:
        yield
    finally:
        if app.state.jobs is not None:
            app.state.jobs.shutdown()
        if app.state.sql_jobs is not None:
            app.state.sql_jobs.shutdown()
        app.state.qdrant.close()
        app.state.social_vectors.close()
        app.state.neo4j.close()


app = FastAPI(
    title=settings.app_name,
    description="Hybrid GraphRAG: Qdrant vector search + Neo4j knowledge graph, grounded answers with citations.",
    lifespan=lifespan,
)


def _unavailable(service: str, exc: Exception) -> JSONResponse:
    logger.error("%s unavailable: %s", service, exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": f"{service} is unavailable ({exc.__class__.__name__}). Check `docker compose ps`."},
    )


@app.exception_handler(ServiceUnavailable)
@app.exception_handler(SessionExpired)
@app.exception_handler(AuthError)
async def neo4j_down(_request: Request, exc: Exception) -> JSONResponse:
    return _unavailable("Neo4j", exc)


@app.exception_handler(ResponseHandlingException)
async def qdrant_down(_request: Request, exc: Exception) -> JSONResponse:
    return _unavailable("Qdrant", exc)
app.include_router(health.router)
app.include_router(ingest_sql.router)  # before ingest: /ingest/sql must not be read as a job id
app.include_router(ingest.router)
app.include_router(retrieve.router)
app.include_router(query.router)
app.include_router(stats.router)
app.include_router(documents.router)
app.include_router(llm.router)
