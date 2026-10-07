"""Pick a loader from the file extension. Every loader returns a list of LoadedPage."""

import hashlib
from pathlib import Path

from graphrag.config import Settings
from graphrag.ingestion.loaders.markdown_loader import load_markdown
from graphrag.ingestion.loaders.pdf_loader import ImageCaptioner, load_pdf
from graphrag.ingestion.normalizer import normalize_text
from graphrag.models import DocType, LoadedPage

SUPPORTED_TYPES: dict[str, DocType] = {".pdf": "pdf", ".md": "md", ".markdown": "md", ".txt": "txt"}


class UnsupportedFileType(ValueError):
    pass


def doc_type_for(filename: str) -> DocType:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_TYPES:
        allowed = ", ".join(sorted(SUPPORTED_TYPES))
        raise UnsupportedFileType(f"Unsupported file type '{suffix or filename}'. Allowed: {allowed}")
    return SUPPORTED_TYPES[suffix]


def make_doc_id(data: bytes) -> str:
    return "doc:" + hashlib.sha256(data).hexdigest()[:16]


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def load_document(
    filename: str, data: bytes, settings: Settings, captioner: ImageCaptioner | None = None
) -> list[LoadedPage]:
    doc_type = doc_type_for(filename)
    doc_id = make_doc_id(data)
    if doc_type == "pdf":
        return load_pdf(data, doc_id, filename, settings, captioner)
    if doc_type == "md":
        return load_markdown(_decode(data), doc_id, filename)
    text = normalize_text(_decode(data))
    return [LoadedPage(doc_id=doc_id, source=filename, doc_type="txt", text=text)] if text else []
