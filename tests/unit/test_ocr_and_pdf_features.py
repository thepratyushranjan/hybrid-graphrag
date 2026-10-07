"""OpenCV deskew, legacy Hindi font detection and vision captions for images without text."""

import io

import numpy as np
import pymupdf
import pytest
from PIL import Image, ImageDraw, ImageFont

from graphrag.config import Settings
from graphrag.ingestion.loaders.ocr import ocr_image, preprocess
from graphrag.ingestion.loaders.pdf_loader import load_pdf

TEXT = [
    "Clause 9.1 requires a safety certificate for",
    "every consignment of guard rails supplied by",
    "Shakti Steel Works to Ganga Roadways sites.",
]


def _text_image(lines: list[str], angle: float = 0.0) -> Image.Image:
    font = ImageFont.load_default(size=36)
    img = Image.new("L", (1400, 420), 255)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((60, 80 + i * 90), line, fill=0, font=font)
    return img.rotate(angle, expand=True, fillcolor=255, resample=Image.Resampling.BICUBIC) if angle else img


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize("angle", [6.0, -4.0])
def test_deskew_estimates_and_corrects_rotation(angle: float) -> None:
    _, skew = preprocess(_text_image(TEXT, angle), deskew=True, max_skew_degrees=10)
    assert skew == pytest.approx(-angle, abs=0.6)


def test_deskew_improves_ocr_of_tilted_scan(settings: Settings) -> None:
    tilted = _text_image(TEXT, 6.0)
    straight = ocr_image(tilted, "eng", settings.ocr_min_confidence, deskew=True)
    crooked = ocr_image(tilted, "eng", settings.ocr_min_confidence, deskew=False)
    assert "Shakti Steel Works" in straight.text
    assert len(straight.text) > len(crooked.text)


def test_preprocess_outputs_binary_image() -> None:
    binary, _ = preprocess(_text_image(TEXT), deskew=False)
    assert set(np.unique(binary)) <= {0, 255}


def _pdf_with_legacy_font() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    # Kruti Dev stores Hindi as Latin code points: "Hkkjr ljdkj" renders as भारत सरकार in the real font
    page.insert_font(fontname="KrutiDev010", fontbuffer=pymupdf.Font("helv").buffer)
    page.insert_text((72, 100), "Hkkjr ljdkj dh vf/klwpuk " * 4, fontname="KrutiDev010", fontsize=14)
    return doc.tobytes()


def test_legacy_hindi_font_page_is_ocred(settings: Settings) -> None:
    pages = load_pdf(_pdf_with_legacy_font(), "doc:x", "legacy.pdf", settings)
    assert pages[0].extraction_method == "ocr_legacy_font"


def _pdf_with_text_and_chart() -> bytes:
    chart = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(chart)
    for i, h in enumerate([80, 150, 220]):
        draw.rectangle((60 + i * 110, 280 - h, 140 + i * 110, 280), fill=(40, 90, 200))
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Quarterly smart meter installations are shown in the chart below." * 2)
    page.insert_image(pymupdf.Rect(72, 120, 372, 345), stream=_png(chart))
    return doc.tobytes()


def test_image_without_text_gets_caption(settings: Settings) -> None:
    calls: list[tuple[int, int]] = []

    def fake_captioner(image: Image.Image) -> str:
        calls.append(image.size)
        return "A bar chart with three rising blue bars."

    page = load_pdf(_pdf_with_text_and_chart(), "doc:x", "chart.pdf", settings, fake_captioner)[0]
    assert len(calls) == 1
    assert page.image_captions == 1
    assert "[Image description, page 1]\nA bar chart with three rising blue bars." in page.text


def test_caption_failure_does_not_fail_document(settings: Settings) -> None:
    def broken_captioner(image: Image.Image) -> str:
        raise RuntimeError("model offline")

    page = load_pdf(_pdf_with_text_and_chart(), "doc:x", "chart.pdf", settings, broken_captioner)[0]
    assert page.image_captions == 0
    assert "Quarterly smart meter" in page.text


def test_no_captioner_means_no_captions(settings: Settings) -> None:
    page = load_pdf(_pdf_with_text_and_chart(), "doc:x", "chart.pdf", settings)[0]
    assert page.image_captions == 0 and "[Image description" not in page.text
