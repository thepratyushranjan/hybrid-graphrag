from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from graphrag.api.routes import health
from graphrag.config import get_settings
from graphrag.graph.neo4j_store import Neo4jStore
from graphrag.vector_store.qdrant_store import QdrantStore

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.qdrant = QdrantStore(settings)
    app.state.neo4j = Neo4jStore(settings)
    try:
        yield
    finally:
        app.state.qdrant.close()
        app.state.neo4j.close()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(health.router)
