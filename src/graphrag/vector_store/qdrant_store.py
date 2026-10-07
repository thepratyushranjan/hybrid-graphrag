import uuid

from qdrant_client import QdrantClient, models

from graphrag.config import Settings
from graphrag.models import Chunk, RetrievedChunk

# HNSW graph settings (Qdrant defaults, set explicitly so they're documented):
#   m = edges per node (recall vs memory), ef_construct = build-time search width (recall vs build time)
HNSW_M = 16
HNSW_EF_CONSTRUCT = 100
UPSERT_BATCH = 64

KEYWORD_INDEXES = ("doc_id", "chunk_id", "source", "doc_type", "language", "extraction_method")
INTEGER_INDEXES = ("page",)


class VectorStoreError(RuntimeError):
    pass


def point_id(chunk_id: str) -> str:
    """Qdrant only accepts UUIDs or integers; derive a stable UUID from the chunk ID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _match(filters: dict[str, str | int]) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k, v in filters.items()]
    )


class QdrantStore:
    def __init__(self, settings: Settings) -> None:
        self.collection_name = settings.collection_name
        self.dim = settings.embedding_dim
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)

    def ping(self) -> None:
        self.client.get_collections()

    def close(self) -> None:
        self.client.close()

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            vectors = self.client.get_collection(self.collection_name).config.params.vectors
            size = vectors.size if isinstance(vectors, models.VectorParams) else None
            if size != self.dim:
                raise VectorStoreError(
                    f"Collection '{self.collection_name}' has {size}-dim vectors but EMBEDDING_DIM={self.dim}. "
                    "Run `make reset` (deletes data) or use another COLLECTION_NAME."
                )
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT),
        )
        for field in KEYWORD_INDEXES:
            self.client.create_payload_index(self.collection_name, field, models.PayloadSchemaType.KEYWORD)
        for field in INTEGER_INDEXES:
            self.client.create_payload_index(self.collection_name, field, models.PayloadSchemaType.INTEGER)

    def count(self, filters: dict[str, str | int] | None = None) -> int:
        return self.client.count(
            self.collection_name, count_filter=_match(filters) if filters else None, exact=True
        ).count

    def replace_document(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """Upsert a document's chunks and drop every other point stored for the same file.

        Stale points appear when the file is edited, or when the same file is re-chunked differently
        (new CHUNK_SIZE, OCR settings...). Returns how many were removed. Re-ingesting identical content
        is a no-op: chunk IDs (and so point IDs) are deterministic, so upserts overwrite in place.
        """
        if len(chunks) != len(vectors):
            raise VectorStoreError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return 0
        stale = models.Filter(
            must=[models.FieldCondition(key="source", match=models.MatchValue(value=chunks[0].source))],
            must_not=[models.HasIdCondition(has_id=[point_id(c.chunk_id) for c in chunks])],
        )
        removed = self.client.count(self.collection_name, count_filter=stale, exact=True).count
        if removed:
            self.client.delete(self.collection_name, points_selector=models.FilterSelector(filter=stale), wait=True)

        for start in range(0, len(chunks), UPSERT_BATCH):
            batch = zip(chunks[start : start + UPSERT_BATCH], vectors[start : start + UPSERT_BATCH], strict=True)
            self.client.upsert(
                self.collection_name,
                points=[
                    models.PointStruct(id=point_id(c.chunk_id), vector=v, payload=c.model_dump()) for c, v in batch
                ],
                wait=True,
            )
        return removed

    def search(
        self, vector: list[float], top_k: int, filters: dict[str, str | int] | None = None
    ) -> list[RetrievedChunk]:
        hits = self.client.query_points(
            self.collection_name,
            query=vector,
            limit=top_k,
            query_filter=_match(filters) if filters else None,
            with_payload=True,
        ).points
        return [RetrievedChunk(chunk=Chunk.model_validate(h.payload), score=h.score) for h in hits]
