from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from graphrag.api.jobs import SqlJobStore
from graphrag.config import get_settings
from graphrag.models.social import SqlIngestJob

router = APIRouter(tags=["ingest"])

MAX_DUMP_BYTES = 300 * 1024 * 1024


def get_sql_jobs(request: Request) -> SqlJobStore:
    jobs: SqlJobStore | None = getattr(request.app.state, "sql_jobs", None)
    if jobs is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            f"SQL ingestion unavailable: {request.app.state.pipeline_error}")
    return jobs


@router.post("/ingest/sql", status_code=status.HTTP_202_ACCEPTED, response_model=SqlIngestJob)
def ingest_sql(
    request: Request,
    file: Annotated[UploadFile | None, File(description="MySQL dump (.sql). Omit to load SQL_DUMP_PATH")] = None,
) -> SqlIngestJob:
    """Load a MySQL dump (analyzed_data + lookup tables) into Neo4j and the `social_posts` Qdrant collection,
    in the background. Poll `GET /ingest/sql/{job_id}`. Re-running is idempotent."""
    jobs = get_sql_jobs(request)
    if file is None:
        path = Path(get_settings().sql_dump_path)
        if not path.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No dump at {path}: upload one or set SQL_DUMP_PATH")
        return jobs.submit(path.name, str(path))
    name = file.filename or "dump.sql"
    if not name.lower().endswith(".sql"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Upload a .sql dump")
    data = file.file.read(MAX_DUMP_BYTES + 1)
    if len(data) > MAX_DUMP_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Dump is larger than 300 MB")
    if b"INSERT INTO" not in data:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"{name} has no INSERT statements")
    return jobs.submit(name, data)


@router.get("/ingest/sql/{job_id}", response_model=SqlIngestJob)
def ingest_sql_status(job_id: str, request: Request) -> SqlIngestJob:
    job = get_sql_jobs(request).get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No SQL ingestion job {job_id}")
    return job
