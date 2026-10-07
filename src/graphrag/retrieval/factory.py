from graphrag.config import Settings
from graphrag.embeddings.embedder import Embedder
from graphrag.graph.neo4j_store import Neo4jStore
from graphrag.llm.cache import JsonCache
from graphrag.llm.client import LLMClient
from graphrag.retrieval.hybrid import HybridRetriever
from graphrag.retrieval.query_analyzer import QueryAnalyzer
from graphrag.retrieval.reranker import Reranker
from graphrag.vector_store.qdrant_store import QdrantStore


def build_retriever(
    settings: Settings, embedder: Embedder, vectors: QdrantStore, graph: Neo4jStore, llm: LLMClient | None
) -> HybridRetriever:
    analyzer = QueryAnalyzer(settings, graph, llm, JsonCache(settings.llm_cache_dir))
    reranker = Reranker(settings) if settings.rerank_enabled else None
    return HybridRetriever(settings, embedder, vectors, graph, analyzer, reranker)
