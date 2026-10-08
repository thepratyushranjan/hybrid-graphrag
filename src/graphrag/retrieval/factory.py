from graphrag.config import Settings
from graphrag.embeddings.embedder import Embedder
from graphrag.graph.neo4j_store import Neo4jStore
from graphrag.graph.social_store import SocialGraphStore
from graphrag.llm.cache import JsonCache
from graphrag.llm.client import LLMClient
from graphrag.retrieval.hybrid import HybridRetriever
from graphrag.retrieval.query_analyzer import QueryAnalyzer
from graphrag.retrieval.reranker import Reranker
from graphrag.retrieval.social_analyzer import SocialQueryAnalyzer
from graphrag.retrieval.social_retriever import SocialRetriever
from graphrag.vector_store.qdrant_store import QdrantStore, SocialPostStore


def build_retriever(
    settings: Settings, embedder: Embedder, vectors: QdrantStore, graph: Neo4jStore, llm: LLMClient | None
) -> HybridRetriever:
    analyzer = QueryAnalyzer(settings, graph, llm, JsonCache(settings.llm_cache_dir))
    reranker = Reranker(settings) if settings.rerank_enabled else None
    return HybridRetriever(settings, embedder, vectors, graph, analyzer, reranker)


def build_sql_retriever(
    settings: Settings, embedder: Embedder, vectors: SocialPostStore, graph: SocialGraphStore, reranker: Reranker | None
) -> SocialRetriever:
    """The SQL corpus shares the embedder and the reranker with the document retriever."""
    analyzer = SocialQueryAnalyzer(graph, settings.hub_degree_limit)
    return SocialRetriever(settings, embedder, vectors, graph, analyzer, reranker)
