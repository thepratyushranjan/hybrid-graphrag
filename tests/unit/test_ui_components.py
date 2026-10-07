"""UI helpers that guard against injected markup (the UI runs in the api image, so it is importable)."""

import sys
from pathlib import Path
from unittest.mock import patch

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ui"))

from components.evidence import render_answer  # noqa: E402
from components.graph_view import _safe  # noqa: E402
from components.timing import stage_totals  # noqa: E402


def test_answer_html_is_escaped_but_citations_link() -> None:
    out: list[str] = []
    with patch.object(st, "markdown", side_effect=lambda body, **kw: out.append(body)):
        render_answer("X <img src=x onerror=alert(1)> **bold** [C1] [C9]", [{"cite_id": "C1", "source": "a.pdf",
                                                                             "text": "t"}], 3)
    assert "<img" not in out[0] and "&lt;img" in out[0]
    assert '<a href="#cite-3-C1"' in out[0] and "[C9]" in out[0] and "**bold**" in out[0]


def test_graph_labels_cannot_close_script() -> None:
    assert "<" not in _safe("</script><script>alert(1)</script>")


def test_timing_stages_group_raw_timings() -> None:
    rows = stage_totals({"query_analysis": 900, "embed_query": 20, "vector_search": 5, "graph_search": 7,
                         "rank_facts": 80, "rerank": 120, "generate": 6000, "total": 7200})
    assert rows == [
        {"Stage": "Query analysis", "ms": 900.0}, {"Stage": "Vector search", "ms": 25.0},
        {"Stage": "Graph search", "ms": 7.0}, {"Stage": "Fusion + rerank", "ms": 200.0},
        {"Stage": "LLM answer", "ms": 6000.0},
    ]
