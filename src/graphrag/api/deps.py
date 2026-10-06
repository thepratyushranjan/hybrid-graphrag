from fastapi import Request

from graphrag.graph.neo4j_store import Neo4jStore
from graphrag.vector_store.qdrant_store import QdrantStore


def get_qdrant(request: Request) -> QdrantStore:
    return request.app.state.qdrant


def get_neo4j(request: Request) -> Neo4jStore:
    return request.app.state.neo4j
