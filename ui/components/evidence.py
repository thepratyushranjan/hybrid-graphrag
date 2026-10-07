"""Answer with clickable citation markers, cited sources, chunk cards and the triples table."""

import re
from typing import Any

import streamlit as st

from components.graph_view import escape

_CITE = re.compile(r"\[([CG]\d+)\]")


def _anchor(msg: int, cite_id: str) -> str:
    return f"cite-{msg}-{cite_id}"


def render_answer(answer: str, citations: list[dict[str, Any]], msg: int) -> None:
    """[C1] / [G2] become links to the cited source below; hovering shows a preview."""
    preview = {c["cite_id"]: (c.get("source", "") + " — " + c["text"])[:240] for c in citations}

    def link(m: re.Match[str]) -> str:
        cid = m.group(1)
        if cid not in preview:
            return m.group(0)
        return (f'<a href="#{_anchor(msg, cid)}" title="{escape(preview[cid])}" '
                f'style="text-decoration:none;font-weight:600">[{cid}]</a>')

    # the answer is LLM output (which may echo document text): escape any HTML in it, then add our own links
    safe_answer = escape(answer).replace("&quot;", '"').replace("&#x27;", "'")
    st.markdown(_CITE.sub(link, safe_answer), unsafe_allow_html=True)


def doc_link(public_api: str, source: str, page: int | None) -> str:
    return f"{public_api}/documents/{source}" + (f"#page={page}" if page else "")


def render_citations(citations: list[dict[str, Any]], msg: int, public_api: str) -> None:
    with st.expander(f"🔗 Sources cited ({len(citations)})", expanded=True):
        for c in citations:
            anchor = f'<a id="{_anchor(msg, c["cite_id"])}"></a>'
            if c["kind"] == "chunk":
                where = f"page {c['page']}" if c.get("page") else (c.get("section") or "").split(" > ")[-1]
                url = doc_link(public_api, c["source"], c.get("page"))
                st.markdown(f'{anchor}**[{c["cite_id"]}]** <a href="{escape(url)}" target="_blank">'
                            f'{escape(c["source"])}</a> · {escape(where)}<br>'
                            f'<span style="opacity:0.8">{escape(c["text"][:260])}…</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'{anchor}**[{c["cite_id"]}]** {escape(c["text"])}<br>'
                            f'<span style="opacity:0.8">“{escape(c.get("evidence") or "")}”</span>',
                            unsafe_allow_html=True)


def render_rerank_effect(chunks: list[dict[str, Any]]) -> None:
    """Bonus: how the cross-encoder reordered the fused candidates (position before -> after)."""
    rows = [
        {
            "": c["cite_id"],
            "Source": c["item"]["chunk"]["source"],
            "Before rerank": c["item"].get("fused_rank"),
            "After rerank": i,
            "Move": ("▲ " if c["item"]["fused_rank"] > i else "▼ " if c["item"]["fused_rank"] < i else "= ")
            + str(abs(c["item"]["fused_rank"] - i)),
            "Rerank score": round(c["item"]["rerank_score"], 2),
        }
        for i, c in enumerate(chunks, start=1)
        if c["item"].get("rerank_score") is not None and c["item"].get("fused_rank")
    ]
    if rows:
        st.markdown("**Rerank effect** (position after vector + graph fusion → after the cross-encoder)")
        st.dataframe(rows, hide_index=True, width="stretch")


def render_chunk_cards(chunks: list[dict[str, Any]], used: set[str], public_api: str) -> None:
    with st.expander(f"📄 Text excerpts ({len(chunks)})"):
        if not chunks:
            st.caption("No text excerpts.")
        render_rerank_effect(chunks)
        for c in chunks:
            item, ch = c["item"], c["item"]["chunk"]
            where = f"page {ch['page']}" if ch.get("page") else (ch.get("section") or "").split(" > ")[-1]
            found = " + ".join(item.get("found_by", [])) or "-"
            scores = [f"found by **{found}**"]
            if item.get("vector_score") is not None:
                scores.append(f"vector {item['vector_score']:.3f}")
            if item.get("rerank_score") is not None:
                scores.append(f"rerank {item['rerank_score']:.2f}")
            if item.get("fused_rank"):
                scores.append(f"fused rank {item['fused_rank']}")
            meta = [x for x in (ch.get("language"), ", ".join(ch.get("dates", [])[:3]),
                                ch["extraction_method"].replace("_", " ") if ch.get("extraction_method") != "text" else "")
                    if x]
            url = doc_link(public_api, ch["source"], ch.get("page"))
            with st.container(border=True):
                st.markdown(
                    f"**[{c['cite_id']}]** {'✅ cited · ' if c['cite_id'] in used else ''}"
                    f'<a href="{escape(url)}" target="_blank">{escape(ch["source"])}</a> · {escape(where)}  \n'
                    + " · ".join(scores)
                    + (f"  \n<small>{escape(' · '.join(meta))}</small>" if meta else ""),
                    unsafe_allow_html=True,
                )
                st.write(ch["text"])


def render_triples(facts: list[dict[str, Any]], aggregates: list[dict[str, Any]], used: set[str]) -> None:
    with st.expander(f"🕸️ Graph relationships ({len(facts)})"):
        if aggregates:
            st.markdown("**Counts**")
            st.dataframe([{"Entity": a["name"], "Count": a["count"]} for a in aggregates], hide_index=True)
        if not facts:
            st.caption("No graph relationships.")
            return
        st.dataframe(
            [
                {
                    "": f["cite_id"] + (" ✅" if f["cite_id"] in used else ""),
                    "Subject": f["fact"]["subject"],
                    "Relation": f["fact"]["predicate"],
                    "Object": f["fact"]["object"],
                    "Source": ", ".join(f["support"] or f["fact"].get("sources", [])) or "-",
                    "Confidence": round(f["fact"]["confidence"], 2),
                    "Hops": f["fact"]["hops"],
                    "Evidence": f["fact"]["evidence"],
                }
                for f in facts
            ],
            width="stretch",
            hide_index=True,
        )
