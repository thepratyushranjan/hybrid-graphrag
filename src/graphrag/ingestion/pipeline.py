"""Ingestion: load -> chunk -> embed -> Qdrant. (The Neo4j graph branch is added in Milestone 3.)"""

import io
import logging
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from graphrag.config import Settings
from graphrag.embeddings.embedder import Embedder
from graphrag.ingestion.chunker import Chunker
from graphrag.ingestion.loaders import doc_type_for, load_document, make_doc_id
from graphrag.ingestion.loaders.pdf_loader import ImageCaptioner
from graphrag.llm.client import LLMClient
from graphrag.models import IngestResult
from graphrag.vector_store.qdrant_store import QdrantStore

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


class EmptyDocumentError(ValueError):
    pass


class IngestionPipeline:
    def __init__(self, settings: Settings, embedder: Embedder, vector_store: QdrantStore) -> None:
        self.settings = settings
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunker = Chunker(settings.chunk_size, settings.chunk_overlap, embedder.count_tokens)
        self.captioner = self._build_captioner(settings)

    @staticmethod
    def _build_captioner(settings: Settings) -> ImageCaptioner | None:
        if not settings.vision_captions_enabled:
            return None
        llm = LLMClient(settings)
        logger.info("Image descriptions on: %s/%s", llm.provider, llm.vision_model)

        def caption(image: "Image.Image") -> str:
            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="PNG")
            return llm.describe_image(buf.getvalue())

        return caption

    def ingest(self, filename: str, data: bytes) -> IngestResult:
        started = time.perf_counter()
        doc_type = doc_type_for(filename)  # raises UnsupportedFileType early
        pages = load_document(filename, data, self.settings, self.captioner)
        chunks = self.chunker.split(pages)
        if not chunks:
            raise EmptyDocumentError(f"No text could be extracted from {filename}")

        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.vector_store.ensure_collection()
        removed = self.vector_store.replace_document(chunks, vectors)

        result = IngestResult(
            doc_id=make_doc_id(data),
            source=filename,
            doc_type=doc_type,
            pages=len(pages),
            chunks=len(chunks),
            languages=dict(Counter(c.language for c in chunks)),
            extraction_methods=dict(Counter(p.extraction_method for p in pages)),
            replaced_points=removed,
            seconds=round(time.perf_counter() - started, 2),
        )
        logger.info("Ingested %s: %d pages, %d chunks in %.2fs", filename, result.pages, result.chunks, result.seconds)
        return result

    def ingest_path(self, path: Path) -> IngestResult:
        return self.ingest(path.name, path.read_bytes())
