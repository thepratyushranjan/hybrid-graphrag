from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from graphrag.api.deps import get_pipeline
from graphrag.ingestion.loaders import UnsupportedFileType
from graphrag.ingestion.pipeline import EmptyDocumentError, IngestionPipeline
from graphrag.models import IngestResult

router = APIRouter(tags=["ingest"])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@router.post("/ingest", response_model=IngestResult)
def ingest(
    file: Annotated[UploadFile, File(description="PDF, Markdown or text file, any language")],
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
) -> IngestResult:
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File is larger than 50 MB")
    if not data:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "File is empty")
    try:
        return pipeline.ingest(file.filename or "upload.txt", data)
    except UnsupportedFileType as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
