from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from graphrag.config import get_settings
from graphrag.ingestion.loaders import SUPPORTED_TYPES

router = APIRouter(tags=["documents"])

_MEDIA = {".pdf": "application/pdf", ".md": "text/markdown; charset=utf-8",
          ".markdown": "text/markdown; charset=utf-8", ".txt": "text/plain; charset=utf-8"}


@router.get("/documents/{name}", response_class=FileResponse)
def document(name: str) -> FileResponse:
    """The original file behind a citation (uploads first, then sample documents). PDFs open in the browser;
    append #page=N to jump to a page."""
    safe = Path(name).name  # no path traversal: only plain file names
    if safe != name or Path(safe).suffix.lower() not in SUPPORTED_TYPES:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown document")
    settings = get_settings()
    for folder in (settings.documents_dir, settings.samples_dir):
        path = Path(folder) / safe
        if path.is_file():
            return FileResponse(path, media_type=_MEDIA[path.suffix.lower()],
                                headers={"Content-Disposition": f'inline; filename="{safe}"'})
    raise HTTPException(status.HTTP_404_NOT_FOUND, f"{safe} is not stored (ingested before uploads were kept)")
