"""Markdown loader: split on headers first so sections stay together, and keep the header path."""

from langchain_text_splitters import MarkdownHeaderTextSplitter

from graphrag.ingestion.normalizer import normalize_text
from graphrag.models import LoadedPage

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def load_markdown(text: str, doc_id: str, source: str) -> list[LoadedPage]:
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS, strip_headers=False)
    pages: list[LoadedPage] = []
    for section in splitter.split_text(text):
        body = normalize_text(section.page_content)
        if not body:
            continue
        path = " > ".join(section.metadata[key] for _, key in _HEADERS if key in section.metadata)
        pages.append(
            LoadedPage(doc_id=doc_id, source=source, doc_type="md", text=body, section=path or None)
        )
    return pages
