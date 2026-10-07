from pathlib import Path

import pytest

from graphrag.config import Settings
from graphrag.ingestion.loaders import UnsupportedFileType, load_document

PDF = "vendor_compliance_bulletin_q3_2025.pdf"
MD = "ganga_infra_annual_report_2025.md"


def test_markdown_sections_keep_header_path(samples: Path, settings: Settings) -> None:
    pages = load_document(MD, (samples / MD).read_bytes(), settings)
    sections = [p.section or "" for p in pages]
    assert any(s.endswith("2. Key Vendors and Suppliers") for s in sections)
    assert any("वार्षिक सुरक्षा" in s for s in sections)


def test_pdf_text_image_ocr_and_scanned_page(samples: Path, settings: Settings) -> None:
    pages = {p.page: p for p in load_document(PDF, (samples / PDF).read_bytes(), settings)}
    assert pages[1].extraction_method == "text+ocr_image"  # text layer + Hindi text inside an image
    assert "[Image text, page 1]" in pages[1].text
    assert "शक्ति स्टील वर्क्स" in pages[1].text
    assert pages[2].extraction_method == "ocr_page"  # scanned page with no text layer
    assert "Shakti Steel Works on probation" in pages[2].text
    assert "परिवीक्षा" in pages[2].text
    assert pages[2].ocr_confidence is not None and pages[2].ocr_confidence > 80
    assert pages[3].extraction_method == "text"


def test_pdf_without_ocr_skips_scanned_page(samples: Path, settings: Settings) -> None:
    no_ocr = settings.model_copy(update={"ocr_enabled": False})
    pages = load_document(PDF, (samples / PDF).read_bytes(), no_ocr)
    assert [p.page for p in pages] == [1, 3]


def test_unsupported_file_type(settings: Settings) -> None:
    with pytest.raises(UnsupportedFileType):
        load_document("report.docx", b"data", settings)
