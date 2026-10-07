import re
import unicodedata

# PDF fonts often emit typographic ligatures as single glyphs ("Oﬃce"); expand them for search
_LIGATURES = str.maketrans({"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st"})
_ZERO_WIDTH = re.compile("[​⁠﻿]")  # keep ZWJ/ZWNJ (‌/‍): Indic scripts need them
_SPACES = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """NFC-normalise, drop zero-width characters and collapse runs of whitespace."""
    text = unicodedata.normalize("NFC", text).translate(_LIGATURES)
    text = _ZERO_WIDTH.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", text).strip()
