"""Tesseract OCR with OpenCV pre-processing (grayscale, denoise, threshold, deskew)
and per-word confidence filtering."""

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

_MIN_OCR_WIDTH = 1500  # Tesseract reads small text far better after upscaling
_SKEW_PROBE_WIDTH = 800  # skew is estimated on a downscaled copy for speed


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float | None  # mean confidence (0-100) of the kept words, None if nothing was kept
    skew_degrees: float = 0.0  # rotation applied to straighten the image


def _rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def _line_score(ink: np.ndarray, degrees: float) -> float:
    """Text lines are horizontal when row sums alternate sharply between ink rows and blank rows."""
    h, w = ink.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    rows = cv2.warpAffine(ink, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0).sum(axis=1, dtype=np.float64)
    return float(np.sum(np.diff(rows) ** 2))


def estimate_skew(binary: np.ndarray, max_degrees: float) -> float:
    """Projection-profile skew estimate (coarse 1° sweep, then 0.1° refinement). `binary`: black text on white."""
    if max_degrees <= 0:
        return 0.0
    scale = min(1.0, _SKEW_PROBE_WIDTH / binary.shape[1])
    small = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else binary
    ink = (small < 128).astype(np.uint8)
    if ink.sum() < 50:  # blank image
        return 0.0
    coarse = max(np.arange(-max_degrees, max_degrees + 0.5, 1.0), key=lambda a: _line_score(ink, a))
    fine = max(np.arange(coarse - 1, coarse + 1.05, 0.1), key=lambda a: _line_score(ink, a))
    return round(float(fine), 1)


def preprocess(image: Image.Image, deskew: bool = True, max_skew_degrees: float = 10) -> tuple[np.ndarray, float]:
    """Grayscale -> upscale -> denoise -> Otsu threshold -> deskew. Returns (binary image, applied rotation)."""
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    if gray.shape[1] < _MIN_OCR_WIDTH:
        scale = _MIN_OCR_WIDTH / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.medianBlur(gray, 3)  # removes scanner speckle without eating thin strokes (after upscaling)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    skew = estimate_skew(binary, max_skew_degrees) if deskew else 0.0
    if abs(skew) >= 0.2:
        binary = _rotate(binary, skew)
        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)  # re-binarise after interpolation
    return binary, skew


def ocr_image(
    image: Image.Image,
    languages: str,
    min_confidence: float,
    deskew: bool = True,
    max_skew_degrees: float = 10,
) -> OcrResult:
    """OCR an image and keep only words at or above `min_confidence`, preserving line breaks."""
    binary, skew = preprocess(image, deskew, max_skew_degrees)
    try:
        data = pytesseract.image_to_data(
            Image.fromarray(binary), lang=languages, config="--psm 3", output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractError as exc:
        logger.warning("OCR failed: %s", exc)
        return OcrResult(text="", confidence=None, skew_degrees=skew)

    lines: dict[tuple[int, int, int], list[str]] = {}
    confidences: list[float] = []
    for i, word in enumerate(data["text"]):
        conf = float(data["conf"][i])
        if not word.strip() or conf < min_confidence:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confidences.append(conf)

    text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
    confidence = round(sum(confidences) / len(confidences), 1) if confidences else None
    return OcrResult(text=text, confidence=confidence, skew_degrees=skew)
