import json
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from pydantic import TypeAdapter, ValidationError

from graphrag.api.deps import get_jobs, get_pipeline
from graphrag.api.jobs import JobStore
from graphrag.ingestion.loaders import SUPPORTED_TYPES, UnsupportedFileType, doc_type_for
from graphrag.ingestion.pipeline import EmptyDocumentError, IngestionPipeline
from graphrag.models import DocMetadata, IngestJob, IngestResult

router = APIRouter(tags=["ingest"])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_METADATA = TypeAdapter(DocMetadata)


def _bad(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail)


def validate_upload(filename: str, data: bytes) -> None:
    """Size, extension and content checks (a '.pdf' must really be a PDF, text must be text)."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File is larger than 50 MB")
    if not data:
        raise _bad("File is empty")
    try:
        doc_type = doc_type_for(filename)
    except UnsupportedFileType as exc:
        raise _bad(str(exc)) from exc
    if doc_type == "pdf" and not data.lstrip()[:5].startswith(b"%PDF-"):
        raise _bad(f"{filename} is not a valid PDF")
    if doc_type in ("md", "txt") and b"\x00" in data[:4096]:
        raise _bad(f"{filename} looks like a binary file, not {doc_type} text")


def parse_metadata(raw: str | None) -> DocMetadata:
    if not raw:
        return {}
    try:
        return _METADATA.validate_python(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise _bad('metadata must be a JSON object of simple values, e.g. {"author": "A", "year": 2025}') from exc


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED, response_model=IngestJob | IngestResult)
def ingest(
    response: Response,
    file: Annotated[UploadFile, File(description=f"Any language. Types: {', '.join(sorted(SUPPORTED_TYPES))}")],
    jobs: Annotated[JobStore, Depends(get_jobs)],
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
    metadata: Annotated[str | None, Form(description='Optional JSON object, e.g. {"author": "A"}')] = None,
    wait: Annotated[bool, Query(description="Ingest synchronously and return the result (scripts/tests)")] = False,
) -> IngestJob | IngestResult:
    """Queue a document for ingestion and return its `job_id` at once (LLM extraction is slow).
    Poll `GET /ingest/{job_id}` for progress and counts."""
    filename = file.filename or "upload.txt"
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    validate_upload(filename, data)
    meta = parse_metadata(metadata)
    if wait:
        response.status_code = status.HTTP_200_OK
        try:
            return pipeline.ingest(filename, data, meta)
        except EmptyDocumentError as exc:
            raise _bad(str(exc)) from exc
    return jobs.submit(filename, data, meta)


@router.get("/ingest/{job_id}", response_model=IngestJob)
def ingest_status(job_id: str, jobs: Annotated[JobStore, Depends(get_jobs)]) -> IngestJob:
    """Status, current stage and counts (pages, chunks, entities, relations, OCR'd pages/images, errors)."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No ingestion job {job_id}")
    return job


@router.get("/ingest", response_model=list[IngestJob])
def ingest_jobs(jobs: Annotated[JobStore, Depends(get_jobs)], limit: int = 50) -> list[IngestJob]:
    """Most recent ingestion jobs first."""
    return jobs.list(limit)
