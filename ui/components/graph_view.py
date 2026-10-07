"""Interactive subgraph of the facts behind an answer (pyvis, self-contained HTML: works offline)."""

import base64
import html
import re
from typing import Any

import streamlit as st
from pyvis.network import Network

from components.theme import ENTITY_TYPES, SHAPES, ink, type_color


_UNSAFE = re.compile(r"[<>]")


def _safe(text: str) -> str:
    """Entity names come from uploaded documents and LLM output: strip anything that could close the
    script tag or inject markup into the graph page."""
    return _UNSAFE.sub("", text)


def _legend(present: list[str]) -> str:
    chips = []
    for t in ENTITY_TYPES:
        if t in present:
            shape = {"dot": "●", "square": "■", "triangle": "▲", "diamond": "◆", "star": "★"}[SHAPES[t]]
            chips.append(f'<span style="margin-right:14px"><span style="color:{type_color(t)}">{shape}</span> '
                         f'<span style="color:{ink("secondary")}">{t}</span></span>')
    return '<div style="font-size:0.85rem;margin:2px 0 6px">' + "".join(chips) + "</div>"


def render_subgraph(subgraph: dict[str, Any], used: set[str], key: str) -> None:
    nodes, edges = subgraph.get("nodes", []), subgraph.get("edges", [])
    if not edges:
        st.caption("No graph relationships to draw.")
        return
    net = Network(height="460px", width="100%", directed=True, cdn_resources="in_line",
                  bgcolor=ink("surface"), font_color=ink("primary"))
    # compact layout: the view zooms to fit, and vis.js hides text that ends up smaller than ~5px
    net.barnes_hut(gravity=-4500, spring_length=140, central_gravity=0.45, spring_strength=0.04, overlap=1)
    for n in nodes:
        name = _safe(n["name"])
        label = name if len(name) <= 28 else name[:26] + "…"
        net.add_node(_safe(n["id"]), label=label, shape=SHAPES.get(n.get("type") or "", "dot"), size=20,
                     color=type_color(n.get("type")), title=f'{_safe(n.get("type") or "Entity")}: {name}',
                     font={"color": ink("primary"), "size": 20})
    # one edge per direction between a pair: parallel facts (SUPPLIES, IMPACTS, ...) would otherwise pile up
    merged: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for e in edges:
        merged.setdefault((e["source"], e["target"]), []).append(e)
    for (source, target), group in merged.items():
        cited = any(e["cite_id"] in used for e in group)
        predicates = list(dict.fromkeys(_safe(e["predicate"]) for e in group))
        label = " · ".join(predicates[:2]) + (f" +{len(predicates) - 2}" if len(predicates) > 2 else "")
        title = "\n".join(
            f'[{e["cite_id"]}] {_safe(e["predicate"])} (confidence {e["confidence"]:.2f})'
            + (" · cited in the answer" if e["cite_id"] in used else "")
            for e in group
        )
        net.add_edge(_safe(source), _safe(target), label=label, width=3 if cited else 1.5,
                     color=ink("secondary") if cited else ink("edge"), title=title,
                     font={"color": ink("secondary"), "size": 15, "strokeWidth": 3,
                           "strokeColor": ink("surface"), "align": "middle"},
                     arrows={"to": {"enabled": True, "scaleFactor": 0.6}}, smooth={"type": "curvedCW", "roundness": 0.15})
    st.markdown(_legend(sorted({n.get("type") or "" for n in nodes})), unsafe_allow_html=True)
    st.caption("Drag nodes, scroll to zoom, hover for details. Thick edges are cited in the answer.")
    # A data: URL gets an opaque origin, so the graph page can't reach the Streamlit app (defence in depth on
    # top of _safe); the vis.js library is inlined, so it also works offline.
    page = base64.b64encode(net.generate_html(notebook=False).encode()).decode()
    st.iframe(f"data:text/html;base64,{page}", height=480)


def escape(text: str) -> str:
    return html.escape(text, quote=True)
