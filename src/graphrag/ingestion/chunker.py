"""Recursive, token-measured chunking with deterministic chunk IDs."""

import hashlib
from collections.abc import Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import DetectorFactory, LangDetectException, detect

from graphrag.models import Chunk, LoadedPage

DetectorFactory.seed = 0  # langdetect is random by default; make it repeatable

# Regex separators, tried in order: paragraph, line, Devanagari danda, bullet markers, Latin sentence
# ends, clauses, words, characters. Separators are kept at the end of the left piece (so a sentence
# keeps its "।" or "."); the bullet pattern is a zero-width lookahead, so a bullet starts the next piece.
SEPARATORS = [r"\n\n", r"\n", r"।\s*", r"(?=[⏩•▪➤►])", r"\.\s", r"\?\s", r"!\s", r";\s", r",\s", r"\s", ""]


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def make_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    return "chunk:" + hashlib.sha256(f"{doc_id}|{chunk_index}|{text}".encode()).hexdigest()[:24]


class Chunker:
    def __init__(self, chunk_size: int, chunk_overlap: int, count_tokens: Callable[[str], int]) -> None:
        self.count_tokens = count_tokens
        self.splitter = RecursiveCharacterTextSplitter(
            separators=SEPARATORS,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=count_tokens,
            keep_separator="end",
            is_separator_regex=True,
            strip_whitespace=True,
        )

    def split(self, pages: list[LoadedPage]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in pages:
            for text in self.splitter.split_text(page.text):
                index = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(page.doc_id, index, text),
                        doc_id=page.doc_id,
                        source=page.source,
                        doc_type=page.doc_type,
                        chunk_index=index,
                        text=text,
                        token_count=self.count_tokens(text),
                        page=page.page,
                        section=page.section,
                        language=detect_language(text),
                        extraction_method=page.extraction_method,
                        ocr_confidence=page.ocr_confidence,
                        image_captions=page.image_captions,
                    )
                )
        return chunks
