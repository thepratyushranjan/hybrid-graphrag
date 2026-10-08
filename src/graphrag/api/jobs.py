"""Background ingestion jobs. POST /ingest returns a job_id at once; GET /ingest/{job_id} reports progress.

One worker thread: documents are ingested one at a time, so a burst of uploads doesn't overload the
local GPU / API rate limits. Jobs live in memory (the last MAX_JOBS); re-ingesting is idempotent, so a
restart only loses the status history, never data consistency.
"""

import logging
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from graphrag.ingestion.pipeline import IngestionPipeline
from graphrag.ingestion.sql.pipeline import SqlIngestionPipeline
from graphrag.models import DocMetadata, IngestJob
from graphrag.models.social import SqlIngestJob

logger = logging.getLogger(__name__)

MAX_JOBS = 500


class JobStore:
    def __init__(self, pipeline: IngestionPipeline) -> None:
        self.pipeline = pipeline
        self.jobs: OrderedDict[str, IngestJob] = OrderedDict()
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")

    def submit(self, filename: str, data: bytes, metadata: DocMetadata) -> IngestJob:
        job = IngestJob(
            job_id=uuid.uuid4().hex,
            filename=filename,
            size_bytes=len(data),
            metadata=metadata,
            created_at=datetime.now(UTC),
        )
        with self.lock:
            self.jobs[job.job_id] = job
            while len(self.jobs) > MAX_JOBS:
                self.jobs.popitem(last=False)
        self.executor.submit(self._run, job.job_id, data)
        return job.model_copy()

    def get(self, job_id: str) -> IngestJob | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return job.model_copy() if job else None

    def list(self, limit: int = 50) -> list[IngestJob]:
        with self.lock:
            return [j.model_copy() for j in reversed(self.jobs.values())][:limit]

    def _update(self, job_id: str, **fields: object) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is not None:
                for key, value in fields.items():
                    setattr(job, key, value)

    def _run(self, job_id: str, data: bytes) -> None:
        job = self.get(job_id)
        if job is None:
            return
        self._update(job_id, status="running", stage="starting", started_at=datetime.now(UTC))
        try:
            result = self.pipeline.ingest(
                job.filename, data, job.metadata, progress=lambda stage: self._update(job_id, stage=stage)
            )
            errors = [] if result.graph_status == "complete" else [f"graph {result.graph_status}"]
            self._update(job_id, status="completed", stage="done", result=result, errors=errors)
        except Exception as exc:  # noqa: BLE001 - a failed job is reported, never crashes the worker
            logger.exception("Ingestion job %s (%s) failed", job_id, job.filename)
            self._update(job_id, status="failed", stage="failed", errors=[f"{exc.__class__.__name__}: {exc}"])
        finally:
            self._update(job_id, finished_at=datetime.now(UTC))

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


class SqlJobStore:
    """Background SQL-dump ingestion (parse + graph + ~10k embeddings takes minutes). One at a time."""

    def __init__(self, pipeline: SqlIngestionPipeline) -> None:
        self.pipeline = pipeline
        self.jobs: OrderedDict[str, SqlIngestJob] = OrderedDict()
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest-sql")

    def submit(self, name: str, source: str | bytes) -> SqlIngestJob:
        job = SqlIngestJob(job_id=uuid.uuid4().hex, source=name, created_at=datetime.now(UTC))
        with self.lock:
            self.jobs[job.job_id] = job
            while len(self.jobs) > MAX_JOBS:
                self.jobs.popitem(last=False)
        self.executor.submit(self._run, job.job_id, source)
        return job.model_copy()

    def get(self, job_id: str) -> SqlIngestJob | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return job.model_copy() if job else None

    def _update(self, job_id: str, **fields: object) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is not None:
                for key, value in fields.items():
                    setattr(job, key, value)

    def _run(self, job_id: str, source: str | bytes) -> None:
        job = self.get(job_id)
        if job is None:
            return
        self._update(job_id, status="running", stage="starting", started_at=datetime.now(UTC))
        try:
            result = self.pipeline.ingest(source, job.source, progress=lambda stage: self._update(job_id, stage=stage))
            self._update(job_id, status="completed", stage="done", result=result)
        except Exception as exc:  # noqa: BLE001 - a failed job is reported, never crashes the worker
            logger.exception("SQL ingestion job %s (%s) failed", job_id, job.source)
            self._update(job_id, status="failed", stage="failed", errors=[f"{exc.__class__.__name__}: {exc}"])
        finally:
            self._update(job_id, finished_at=datetime.now(UTC))

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
