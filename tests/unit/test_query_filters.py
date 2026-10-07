"""Payload filters from the query analysis: applied only when the question really names the document/type."""

from typing import Any

import pytest

from graphrag.config import Settings
from graphrag.models import QueryAnalysisLLM
from graphrag.retrieval.query_analyzer import QueryAnalyzer, match_document

SOURCES = ["vendor_compliance_bulletin_q3_2025.pdf", "ganga_infra_annual_report_2025.md", "notes.txt"]


class FakeGraph:
    def document_sources(self, limit: int = 100) -> list[dict[str, Any]]:
        return [{"source": s, "doc_type": s.rsplit(".", 1)[1]} for s in SOURCES]


@pytest.fixture
def analyzer(settings: Settings) -> QueryAnalyzer:
    return QueryAnalyzer(settings, FakeGraph(), llm=None, cache=None)  # type: ignore[arg-type]


def test_match_document_by_words() -> None:
    assert match_document("compliance bulletin", SOURCES) == SOURCES[0]
    assert match_document("ganga_infra_annual_report_2025.md", SOURCES) == SOURCES[1]
    assert match_document("board minutes", SOURCES) is None


@pytest.mark.parametrize(
    ("question", "hint", "expected"),
    [
        ("In the compliance bulletin, what happened?", {"document": SOURCES[0]}, {"source": SOURCES[0]}),
        ("According to the annual report, who leads X?", {"document": "annual report"}, {"source": SOURCES[1]}),
        # the LLM invents a document the question never mentions -> ignored
        ("Who leads the subsidiary?", {"document": SOURCES[0]}, {}),
        ("Which vendors are mentioned in the PDFs?", {"doc_type": "pdf"}, {"doc_type": "pdf"}),
        # type hint without the word in the question -> ignored
        ("Which vendors are mentioned?", {"doc_type": "pdf"}, {}),
        # a document that doesn't exist -> ignored
        ("In the board minutes, what was decided?", {"document": "board minutes"}, {}),
    ],
)
def test_filters_need_the_question_to_name_them(
    analyzer: QueryAnalyzer, question: str, hint: dict[str, str], expected: dict[str, str]
) -> None:
    assert analyzer._filters(QueryAnalysisLLM(**hint), question) == expected  # type: ignore[arg-type]
