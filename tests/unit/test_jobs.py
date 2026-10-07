"""Background ingestion jobs, with a fake pipeline."""

import time

from graphrag.api.jobs import JobStore
from graphrag.models import IngestJob, IngestResult


class FakePipeline:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def ingest(self, filename, data, metadata=None, progress=None):  # type: ignore[no-untyped-def]
        progress("extracting graph 1/1")
        if self.fail:
            raise RuntimeError("Ollama is down")
        return IngestResult(doc_id="doc:x", source=filename, doc_type="txt", pages=1, chunks=2, entities=3,
                            relations=4, ocr_images=1, graph_status="complete", seconds=0.1)


def _wait(store: JobStore, job_id: str) -> IngestJob:
    for _ in range(100):
        job = store.get(job_id)
        assert job is not None
        if job.status in ("completed", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_job_runs_in_background_and_reports_counts() -> None:
    store = JobStore(FakePipeline())  # type: ignore[arg-type]
    queued = store.submit("a.txt", b"hello", {"author": "me"})
    assert queued.metadata == {"author": "me"}  # submit returns at once (the fake may already be done)
    job = _wait(store, queued.job_id)
    assert job.status == "completed" and job.stage == "done" and job.errors == []
    assert job.result is not None and (job.result.chunks, job.result.relations, job.result.ocr_images) == (2, 4, 1)
    assert job.started_at and job.finished_at and job.finished_at >= job.started_at
    assert store.list()[0].job_id == job.job_id


def test_failed_job_is_reported_not_raised() -> None:
    store = JobStore(FakePipeline(fail=True))  # type: ignore[arg-type]
    job = _wait(store, store.submit("a.txt", b"hello", {}).job_id)
    assert job.status == "failed" and job.errors == ["RuntimeError: Ollama is down"]
