"""PDF loader: text layer per page, plus OCR for scanned pages, legacy-font Hindi pages and text inside
embedded images. Optionally, images with no text are described by a vision LLM."""

import hashlib
import io
import logging
from collections.abc import Callable

import pymupdf
from PIL import Image

from graphrag.config import Settings
from graphrag.ingestion.loaders.ocr import OcrResult, ocr_image
from graphrag.ingestion.normalizer import normalize_text
from graphrag.models import ExtractionMethod, LoadedPage

logger = logging.getLogger(__name__)

_SCAN_DPI = 300

# Pre-Unicode Hindi fonts map Devanagari glyphs onto Latin code points, so their "text" extracts as
# gibberish like "Hkkjr ljdkj" (= भारत सरकार). Pages using them must be OCR'd from the rendered image.
LEGACY_HINDI_FONTS = (
    "kruti", "devlys", "chanakya", "shusha", "susha", "aps-dv", "akruti", "shree-dev", "shree_dev",
    "agra", "walkman", "kundli", "marathi-saral", "shivaji", "amar ujala", "aman", "naidunia",
)

# Called with an image that has no readable text; returns a short description
ImageCaptioner = Callable[[Image.Image], str]


def _pixmap_to_image(pix: pymupdf.Pixmap) -> Image.Image:
    if pix.colorspace and pix.colorspace.n not in (1, 3):  # CMYK etc.
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def _mean_confidence(results: list[OcrResult]) -> float | None:
    scores = [r.confidence for r in results if r.confidence is not None]
    return round(sum(scores) / len(scores), 1) if scores else None


def legacy_hindi_font(page: pymupdf.Page) -> str | None:
    """Return the name of a legacy (non-Unicode) Hindi font used on the page, if any."""
    for _xref, _ext, _type, basefont, name, *_ in page.get_fonts(full=True):
        for font in (basefont, name):
            if any(marker in font.lower() for marker in LEGACY_HINDI_FONTS):
                return font
    return None


class _PdfPageReader:
    def __init__(
        self, doc: pymupdf.Document, doc_id: str, source: str, settings: Settings, captioner: ImageCaptioner | None
    ) -> None:
        self.doc = doc
        self.doc_id = doc_id
        self.source = source
        self.settings = settings
        self.captioner = captioner
        self.seen_images: set[str] = set()  # logos/headers repeat on every page; process each image once

    def ocr(self, image: Image.Image) -> OcrResult:
        s = self.settings
        return ocr_image(image, s.ocr_languages, s.ocr_min_confidence, s.ocr_deskew, s.ocr_max_skew_degrees)

    def read(self, page: pymupdf.Page) -> LoadedPage | None:
        page_no = page.number + 1
        text = normalize_text(page.get_text("text"))
        method: ExtractionMethod = "text"
        ocr_results: list[OcrResult] = []
        captions = 0

        legacy_font = legacy_hindi_font(page)
        if legacy_font and not self.settings.ocr_enabled:
            logger.warning("%s p%d: legacy Hindi font %s but OCR is off; text will be gibberish",
                           self.source, page_no, legacy_font)

        if self.settings.ocr_enabled and (legacy_font or len(text) < self.settings.scanned_page_min_chars):
            # Scanned page, or a legacy-font page whose text layer is unusable: OCR the rendered page
            result = self.ocr(_pixmap_to_image(page.get_pixmap(dpi=_SCAN_DPI)))
            if result.text:
                if legacy_font:
                    logger.info("%s p%d: legacy Hindi font %s, using OCR text", self.source, page_no, legacy_font)
                    text, method = normalize_text(result.text), "ocr_legacy_font"
                else:
                    text, method = normalize_text(f"{text}\n\n{result.text}"), "ocr_page"
                ocr_results.append(result)
        elif self.settings.ocr_enabled or self.captioner:
            # Normal page: OCR embedded images; describe the ones without text if captions are on
            for xref, *_ in page.get_images(full=True):
                image = self._embedded_image(xref, page_no)
                if image is None:
                    continue
                result = self.ocr(image) if self.settings.ocr_enabled else OcrResult(text="", confidence=None)
                if result.text:
                    text += f"\n\n[Image text, page {page_no}]\n{normalize_text(result.text)}"
                    method = "text+ocr_image"
                    ocr_results.append(result)
                elif self.captioner and (caption := self._caption(image, page_no)):
                    text += f"\n\n[Image description, page {page_no}]\n{caption}"
                    captions += 1

        text = text.strip()
        if not text:
            logger.info("%s p%d: no text found, skipping page", self.source, page_no)
            return None
        return LoadedPage(
            doc_id=self.doc_id,
            source=self.source,
            doc_type="pdf",
            text=text,
            page=page_no,
            extraction_method=method,
            ocr_confidence=_mean_confidence(ocr_results),
            image_captions=captions,
            ocr_images=len(ocr_results) if method == "text+ocr_image" else 0,
        )

    def _embedded_image(self, xref: int, page_no: int) -> Image.Image | None:
        try:
            pix = pymupdf.Pixmap(self.doc, xref)
        except RuntimeError as exc:
            logger.warning("%s p%d: cannot read image xref=%d: %s", self.source, page_no, xref, exc)
            return None
        if min(pix.width, pix.height) < self.settings.ocr_min_image_px:
            return None
        digest = hashlib.sha1(pix.samples).hexdigest()
        if digest in self.seen_images:
            return None
        self.seen_images.add(digest)
        return _pixmap_to_image(pix)

    def _caption(self, image: Image.Image, page_no: int) -> str:
        try:
            return normalize_text(self.captioner(image)) if self.captioner else ""
        except Exception as exc:  # noqa: BLE001 - a failed caption must not fail the whole document
            logger.warning("%s p%d: image description failed: %s", self.source, page_no, exc)
            return ""


def load_pdf(
    data: bytes, doc_id: str, source: str, settings: Settings, captioner: ImageCaptioner | None = None
) -> list[LoadedPage]:
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        reader = _PdfPageReader(doc, doc_id, source, settings, captioner)
        return [p for page in doc if (p := reader.read(page)) is not None]
