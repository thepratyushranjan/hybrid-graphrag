import uuid

from qdrant_client import QdrantClient, models

from graphrag.config import Settings
from graphrag.models import Chunk, RetrievedChunk, SimilarDocument, TimeFilter

# HNSW graph settings (Qdrant defaults, set explicitly so they're documented):
#   m = edges per node (recall vs memory), ef_construct = build-time search width (recall vs build time)
HNSW_M = 16
HNSW_EF_CONSTRUCT = 100
UPSERT_BATCH = 64

KEYWORD_INDEXES = ("doc_id", "chunk_id", "source", "doc_type", "language", "extraction_method")
INTEGER_INDEXES = ("page",)
DATETIME_INDEXES = ("date_start", "date_end")


class VectorStoreError(RuntimeError):
    pass


def point_id(chunk_id: str) -> str:
    """Qdrant only accepts UUIDs or integers; derive a stable UUID from the chunk ID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def build_filter(
    filters: dict[str, str] | dict[str, str | int] | None, time_range: TimeFilter | None = None
) -> models.Filter | None:
    """Exact-match payload filters plus an optional date-range overlap: the chunk's date span must overlap
    [start, end]. Undated chunks have no date fields, so a time filter excludes them."""
    must: list[models.Condition] = [
        models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k, v in (filters or {}).items()
    ]
    if time_range is not None:
        if time_range.start:
            must.append(models.FieldCondition(
                key="date_end", range=models.DatetimeRange(gte=f"{time_range.start.isoformat()}T00:00:00Z")))
        if time_range.end:
            must.append(models.FieldCondition(
                key="date_start", range=models.DatetimeRange(lte=f"{time_range.end.isoformat()}T23:59:59Z")))
    return models.Filter(must=must) if must else None


def _match(filters: dict[str, str | int]) -> models.Filter | None:
    return build_filter(filters)


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
            info = self.client.get_collection(self.collection_name)
            vectors = info.config.params.vectors
            size = vectors.size if isinstance(vectors, models.VectorParams) else None
            if size != self.dim:
                raise VectorStoreError(
                    f"Collection '{self.collection_name}' has {size}-dim vectors but EMBEDDING_DIM={self.dim}. "
                    "Run `make reset` (deletes data) or use another COLLECTION_NAME."
                )
            self._ensure_indexes(set(info.payload_schema))  # collections created before a new index was added
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT),
        )
        self._ensure_indexes(set())

    def _ensure_indexes(self, existing: set[str]) -> None:
        for fields, schema in (
            (KEYWORD_INDEXES, models.PayloadSchemaType.KEYWORD),
            (INTEGER_INDEXES, models.PayloadSchemaType.INTEGER),
            (DATETIME_INDEXES, models.PayloadSchemaType.DATETIME),
        ):
            for field in fields:
                if field not in existing:
                    self.client.create_payload_index(self.collection_name, field, schema)

    def count(self, filters: dict[str, str | int] | None = None, time_range: TimeFilter | None = None) -> int:
        return self.client.count(
            self.collection_name, count_filter=build_filter(filters, time_range), exact=True
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
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, str | int] | None = None,
        time_range: TimeFilter | None = None,
    ) -> list[RetrievedChunk]:
        hits = self.client.query_points(
            self.collection_name,
            query=vector,
            limit=top_k,
            query_filter=build_filter(filters, time_range),
            with_payload=True,
        ).points
        return [RetrievedChunk(chunk=Chunk.model_validate(h.payload), score=h.score) for h in hits]

    def similar_documents(
        self, centroid: list[float], exclude_doc_id: str, limit: int, min_score: float
    ) -> list[SimilarDocument]:
        """Other documents whose best chunk is closest to this document's centroid (one hit per doc_id)."""
        groups = self.client.query_points_groups(
            self.collection_name,
            query=centroid,
            group_by="doc_id",
            group_size=1,
            limit=limit,
            query_filter=models.Filter(
                must_not=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=exclude_doc_id))]
            ),
            with_payload=["source"],
        ).groups
        return [
            SimilarDocument(doc_id=str(g.id), source=g.hits[0].payload["source"], score=round(g.hits[0].score, 4))
            for g in groups
            if g.hits and g.hits[0].score >= min_score and g.hits[0].payload
        ]

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        """Fetch chunks by ID (the graph-to-vector bridge), in the order requested."""
        if not chunk_ids:
            return []
        points = self.client.retrieve(self.collection_name, ids=[point_id(c) for c in chunk_ids], with_payload=True)
        by_id = {p.payload["chunk_id"]: Chunk.model_validate(p.payload) for p in points if p.payload}
        return [by_id[c] for c in chunk_ids if c in by_id]
