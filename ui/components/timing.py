"""Per-stage timing bar. Stages partly run in parallel (vector search overlaps query analysis),
so this shows durations side by side, not a stacked timeline."""

from typing import Any

import altair as alt
import streamlit as st

from components.theme import ink, series_1

STAGES: list[tuple[str, tuple[str, ...]]] = [
    ("Query analysis", ("query_analysis", "check_filters")),
    ("Vector search", ("embed_query", "vector_search", "vector_search_filtered")),
    ("Graph search", ("vector_seeds", "graph_search", "bridge_fetch", "entity_mentions")),
    ("Fusion + rerank", ("rank_facts", "rerank")),
    ("LLM answer", ("generate",)),
]


def stage_totals(timings: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {"Stage": name, "ms": round(sum(timings.get(k, 0.0) for k in keys), 1)}
        for name, keys in STAGES
        if any(k in timings for k in keys)
    ]


def render_timing(timings: dict[str, float]) -> None:
    rows = stage_totals(timings)
    if not rows:
        return
    order = [r["Stage"] for r in rows]
    base = alt.Chart(alt.Data(values=rows)).encode(
        y=alt.Y("Stage:N", sort=order, title=None, axis=alt.Axis(labelColor=ink("secondary"), ticks=False,
                                                                 domain=False, labelPadding=8)),
        x=alt.X("ms:Q", title=None, axis=None),
        tooltip=[alt.Tooltip("Stage:N"), alt.Tooltip("ms:Q", title="milliseconds", format=",.0f")],
    )
    bars = base.mark_bar(color=series_1(), size=14, cornerRadiusEnd=4)
    labels = base.mark_text(align="left", dx=6, color=ink("secondary"), fontSize=12).encode(
        text=alt.Text("ms:Q", format=",.0f")
    )
    chart = (bars + labels).properties(height=34 * len(rows) + 10).configure_view(stroke=None)
    total = timings.get("total")
    st.caption(f"⏱️ Timing per stage, ms (total {total:,.0f} ms; vector search runs in parallel with "
               "query analysis, so stages overlap)" if total else "⏱️ Timing per stage, ms")
    st.altair_chart(chart, width="stretch")
    with st.popover("Table view"):
        st.dataframe(rows, hide_index=True)
