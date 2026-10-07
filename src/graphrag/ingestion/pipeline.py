"""Ingestion: load -> chunk -> embed -> Qdrant -> LLM triples -> Neo4j.

Qdrant and Neo4j are not one transaction: Qdrant is written first, then Neo4j (which marks the
Document 'complete' last). Every step is idempotent, so re-ingesting after a failure is safe.
"""

import io
import logging
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from graphrag.config import Settings
from graphrag.embeddings.embedder import Embedder
from graphrag.graph.extractor import EntityResolver, TripleExtractor, document_context
from graphrag.graph.neo4j_store import Neo4jStore
from graphrag.ingestion.chunker import Chunker
from graphrag.ingestion.loaders import doc_type_for, load_document, make_doc_id
from graphrag.ingestion.loaders.pdf_loader import ImageCaptioner
from graphrag.llm.cache import JsonCache
from graphrag.llm.client import LLMClient, LLMError
from graphrag.models import Chunk, ChunkGraph, DocMetadata, IngestResult, SimilarDocument

Progress = Callable[[str], None]  # receives the current stage, e.g. "extracting graph 3/9"
from graphrag.vector_store.qdrant_store import QdrantStore

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


class EmptyDocumentError(ValueError):
    pass


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        vector_store: QdrantStore,
        graph_store: Neo4jStore | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.chunker = Chunker(settings.chunk_size, settings.chunk_overlap, embedder.count_tokens)

        self.llm = llm
        self.llm_error: str | None = None
        if self.llm is None and (settings.graph_enabled or settings.vision_captions_enabled):
            try:
                self.llm = LLMClient(settings)
            except LLMError as exc:
                self.llm_error = str(exc)
                logger.warning("LLM unavailable, graph extraction and image descriptions are off: %s", exc)

        self.captioner = self._build_captioner() if settings.vision_captions_enabled else None
        self.extractor = (
            TripleExtractor(settings, self.llm, JsonCache(settings.llm_cache_dir))
            if settings.graph_enabled and self.llm and graph_store
            else None
        )

    def _build_captioner(self) -> ImageCaptioner | None:
        llm = self.llm
        if llm is None:
            return None
        logger.info("Image descriptions on: %s/%s", llm.provider, llm.vision_model)

        def caption(image: "Image.Image") -> str:
            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="PNG")
            return llm.describe_image(buf.getvalue())

        return caption

    def _graph_skip_reason(self) -> str | None:
        if not self.settings.graph_enabled:
            return "GRAPH_ENABLED=false"
        if self.graph_store is None:
            return "no Neo4j connection"
        if self.extractor is None:
            return f"LLM unavailable ({self.llm_error})"
        return None

    def _extract_graph(self, chunks: list[Chunk], progress: Progress) -> list[ChunkGraph]:
        assert self.extractor is not None and self.graph_store is not None
        resolver = EntityResolver(self.graph_store.known_entities(self.settings.known_entities_limit))
        graphs: list[ChunkGraph] = []
        context = document_context(chunks[0].source, chunks[0].text)
        # Sequential on purpose: each chunk sees the entities found in earlier chunks, which keeps names
        # consistent, and it stays within free-tier / local-GPU rate limits.
        for i, chunk in enumerate(chunks, start=1):
            progress(f"extracting graph {i}/{len(chunks)}")
            graphs.append(self.extractor.extract(chunk, resolver, context))
        return graphs

    def _similar_documents(self, doc_id: str, vectors: list[list[float]]) -> list[SimilarDocument]:
        if not self.settings.similar_docs_enabled:
            return []
        centroid = np.mean(np.asarray(vectors), axis=0)
        centroid /= np.linalg.norm(centroid) or 1.0
        return self.vector_store.similar_documents(
            centroid.tolist(), doc_id, self.settings.similar_docs_top_n, self.settings.similar_docs_min_score
        )

    def ingest(
        self, filename: str, data: bytes, metadata: DocMetadata | None = None, progress: Progress | None = None
    ) -> IngestResult:
        report: Progress = progress or (lambda _stage: None)
        started = time.perf_counter()
        doc_type = doc_type_for(filename)  # raises UnsupportedFileType early
        doc_id = make_doc_id(data)
        report("loading" + (" (OCR)" if doc_type == "pdf" and self.settings.ocr_enabled else ""))
        pages = load_document(filename, data, self.settings, self.captioner)
        report("chunking")
        chunks = self.chunker.split(pages)
        if not chunks:
            raise EmptyDocumentError(f"No text could be extracted from {filename}")
        if metadata:
            chunks = [c.model_copy(update={"metadata": metadata}) for c in chunks]

        # 1. Vector branch
        report(f"embedding {len(chunks)} chunks")
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.vector_store.ensure_collection()
        removed = self.vector_store.replace_document(chunks, vectors)

        # 2. Graph branch (the lexical graph is written even when LLM extraction is unavailable)
        graphs: list[ChunkGraph] = []
        graph_status = "complete"
        if skip := self._graph_skip_reason():
            graph_status = f"skipped: {skip}"
        else:
            try:
                graphs = self._extract_graph(chunks, report)
            except LLMError as exc:
                graph_status = f"failed: {exc}"
                logger.error("Graph extraction failed for %s: %s", filename, exc)
        similar: list[SimilarDocument] = []
        report("writing graph")
        if self.graph_store is not None:
            if graph_status == "complete" or not self.graph_store.has_source(filename):
                similar = self._similar_documents(doc_id, vectors)
                self.graph_store.write_document(doc_id, filename, doc_type, chunks, graphs, similar, metadata)
            else:
                # Don't replace a graph built earlier with an entity-less one just because the LLM is down
                graph_status += " (kept the existing graph for this file)"

        result = IngestResult(
            doc_id=doc_id,
            source=filename,
            doc_type=doc_type,
            pages=len(pages),
            chunks=len(chunks),
            languages=dict(Counter(c.language for c in chunks)),
            extraction_methods=dict(Counter(p.extraction_method for p in pages)),
            ocr_pages=sum(p.extraction_method in ("ocr_page", "ocr_legacy_font") for p in pages),
            ocr_images=sum(p.ocr_images for p in pages),
            replaced_points=removed,
            entities=len({e.id for g in graphs for e in g.entities}),
            relations=sum(len(g.relations) for g in graphs),
            dropped_relations=sum(g.dropped for g in graphs),
            similar_documents=[f"{d.source} ({d.score:.3f})" for d in similar],
            graph_status=graph_status,
            seconds=round(time.perf_counter() - started, 2),
        )
        logger.info(
            "Ingested %s: %d chunks, %d entities, %d relations (%d dropped), graph %s, %.1fs",
            filename, result.chunks, result.entities, result.relations, result.dropped_relations,
            graph_status, result.seconds,
        )
        return result

    def ingest_path(self, path: Path) -> IngestResult:
        return self.ingest(path.name, path.read_bytes())
