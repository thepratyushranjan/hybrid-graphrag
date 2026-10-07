import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from graphrag.api.routes import health, ingest
from graphrag.config import get_settings
from graphrag.embeddings.embedder import build_embedder
from graphrag.graph.neo4j_store import Neo4jStore
from graphrag.ingestion.pipeline import IngestionPipeline
from graphrag.vector_store.qdrant_store import QdrantStore

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
    app.state.pipeline_error = None
    try:
        app.state.qdrant.ensure_collection()
        app.state.pipeline = IngestionPipeline(settings, build_embedder(settings), app.state.qdrant)
        logger.info("Ingestion ready: %s (%d-dim)", settings.embedding_model, settings.embedding_dim)
    except Exception as exc:  # noqa: BLE001 - keep the server up; /ingest reports the reason
        app.state.pipeline_error = f"{exc.__class__.__name__}: {exc}"
        logger.error("Ingestion disabled: %s", app.state.pipeline_error)
    try:
        yield
    finally:
        app.state.qdrant.close()
        app.state.neo4j.close()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(health.router)
app.include_router(ingest.router)
