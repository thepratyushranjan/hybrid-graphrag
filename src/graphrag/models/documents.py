from typing import Literal

from pydantic import BaseModel, Field

DocType = Literal["pdf", "md", "txt"]
# text: native text layer | ocr_page: whole page was a scan | text+ocr_image: text layer plus OCR'd images
# ocr_legacy_font: page used a legacy Hindi font (Kruti Dev...), so its text layer was gibberish and it was OCR'd
ExtractionMethod = Literal["text", "ocr_page", "ocr_legacy_font", "text+ocr_image"]


class LoadedPage(BaseModel):
    """One unit of extracted text (a PDF page, a Markdown section or a whole text file)."""

    doc_id: str
    source: str
    doc_type: DocType
    text: str
    page: int | None = None  # 1-based, PDFs only
    section: str | None = None  # Markdown header path, e.g. "Report > Traffic"
    extraction_method: ExtractionMethod = "text"
    ocr_confidence: float | None = None  # mean Tesseract confidence (0-100) of the OCR'd text
    image_captions: int = 0  # images without text that a vision LLM described


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    doc_type: DocType
    chunk_index: int
    text: str
    token_count: int
    page: int | None = None
    section: str | None = None
    language: str = "unknown"
    extraction_method: ExtractionMethod = "text"
    ocr_confidence: float | None = None
    image_captions: int = 0


class IngestResult(BaseModel):
    doc_id: str
    source: str
    doc_type: DocType
    pages: int
    chunks: int
    languages: dict[str, int] = Field(default_factory=dict)
    extraction_methods: dict[str, int] = Field(default_factory=dict)
    replaced_points: int = 0  # old points of the same source removed before upsert
    seconds: float


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
