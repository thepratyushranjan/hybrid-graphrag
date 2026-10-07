"""Hybrid GraphRAG chat UI: grounded answers with [C#] (text) / [G#] (graph) citations, the evidence behind
them, and document ingestion. Talks to the FastAPI service only (POST /query, POST /ingest, GET /health)."""

import os
from typing import Any

import streamlit as st

from api_client import ApiClient

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Hybrid GraphRAG", page_icon="🕸️", layout="wide")
client = ApiClient(API_URL)

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------- rendering helpers ----------

def render_chunks(chunks: list[dict[str, Any]], used: set[str]) -> None:
    with st.expander(f"📄 Text excerpts ({len(chunks)})"):
        if not chunks:
            st.caption("No text excerpts.")
        for c in chunks:
            item, ch = c["item"], c["item"]["chunk"]
            where = f"page {ch['page']}" if ch.get("page") else (ch.get("section") or "")
            badges = " · ".join(
                x for x in (
                    "✅ cited" if c["cite_id"] in used else "",
                    "+".join(item.get("found_by", [])),
                    f"rerank {item['rerank_score']:.2f}" if item.get("rerank_score") is not None else "",
                    ch.get("language", ""),
                    ch["extraction_method"].replace("_", " ") if ch.get("extraction_method") != "text" else "",
                ) if x
            )
            with st.container(border=True):
                st.markdown(f"**[{c['cite_id']}]** `{ch['source']}` · {where}  \n{badges}")
                st.write(ch["text"])


def render_facts(facts: list[dict[str, Any]], aggregates: list[dict[str, Any]], used: set[str]) -> None:
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
                    "Hops": f["fact"]["hops"],
                    "From": ", ".join(f["support"]) or "-",
                    "Evidence": f["fact"]["evidence"],
                }
                for f in facts
            ],
            width="stretch",
            hide_index=True,
        )


def render_analysis(data: dict[str, Any]) -> None:
    a = data["analysis"]
    parts = [f"mode **{data['mode']}**", f"query type **{a['query_type']}**", f"language **{a['language']}**"]
    entities = [e.get("entity_name") or e["text"] for e in a.get("entities", []) if e.get("origin") == "question"]
    seeds = [e.get("entity_name") for e in a.get("entities", []) if e.get("origin") == "vector"]
    if entities:
        parts.append("entities: " + ", ".join(entities))
    if seeds:
        parts.append("graph seeded from top chunks: " + ", ".join(seeds))
    if a.get("filters"):
        parts.append("filter: " + ", ".join(f"{k}={v}" for k, v in a["filters"].items()))
    if a.get("time_range"):
        t = a["time_range"]
        parts.append(f"time: {t['expression']} ({t.get('start') or '…'} → {t.get('end') or '…'})")
    total = data.get("timings_ms", {}).get("total")
    if total:
        parts.append(f"{total / 1000:.1f}s")
    st.caption(" · ".join(parts))
    for dropped in a.get("dropped_filters", []):
        st.caption(f"⚠️ Filter ignored: {dropped}")


def render_ingest_result(data: dict[str, Any]) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Chunks", data.get("chunks", 0))
    c2.metric("Entities", data.get("entities", 0))
    c3.metric("Relations", data.get("relations", 0))
    status = data.get("graph_status", "")
    if status == "complete":
        st.caption(f"🕸️ Graph complete · {data.get('dropped_relations', 0)} relations rejected by checks")
    else:
        st.warning(f"Graph {status}", icon="⚠️")
    if data.get("similar_documents"):
        st.caption("🔗 Similar documents: " + ", ".join(data["similar_documents"]))
    st.caption(
        f"Pages: {data.get('pages')} · Languages: {data.get('languages')} · "
        f"Extraction: {data.get('extraction_methods')}"
    )
    with st.expander("Raw response"):
        st.json(data)


def render_assistant(payload: dict[str, Any]) -> None:
    if payload.get("error"):
        st.warning(payload["error"])
        return
    st.markdown(payload["answer"])
    if not payload.get("grounded"):
        st.caption("ℹ️ No LLM answer: the knowledge base had no relevant evidence, or the LLM is unavailable.")
    used = {c["cite_id"] for c in payload.get("citations", [])}
    if payload.get("citations"):
        with st.expander(f"🔗 Sources cited ({len(used)})", expanded=True):
            for c in payload["citations"]:
                if c["kind"] == "chunk":
                    where = f"page {c['page']}" if c.get("page") else (c.get("section") or "")
                    st.markdown(f"**[{c['cite_id']}]** `{c['source']}` · {where} — {c['text'][:220]}…")
                else:
                    st.markdown(f"**[{c['cite_id']}]** {c['text']}  \n<small>“{c.get('evidence', '')}”</small>",
                                unsafe_allow_html=True)
    if payload.get("invalid_citations"):
        st.caption("⚠️ Removed citations that weren't in the evidence: " + ", ".join(payload["invalid_citations"]))
    if payload.get("uncited_sentences"):
        st.caption(f"⚠️ {len(payload['uncited_sentences'])} sentence(s) without a citation")
    render_analysis(payload)
    render_chunks(payload.get("chunks", []), used)
    render_facts(payload.get("facts", []), payload.get("aggregates", []), used)


# ---------- sidebar ----------

with st.sidebar:
    st.header("⚙️ System")
    health = client.health()
    if health.data:
        checks = health.data.get("checks", {})
        for name, state in checks.items():
            st.markdown(f"{'🟢' if state == 'ok' else '🔴'} **{name}** — {state}")
        llm, emb = health.data.get("llm", {}), health.data.get("embedding", {})
        st.caption(f"LLM: {llm.get('provider')} / {llm.get('model')}")
        st.caption(f"Embeddings: {emb.get('provider')} / {emb.get('model')}")
        for warning in health.data.get("warnings", []):
            st.warning(warning, icon="⚠️")
    else:
        st.error(health.error or "API unavailable")
    if st.button("Refresh status", width="stretch"):
        st.rerun()

    st.divider()
    st.header("📥 Ingest documents")
    uploads = st.file_uploader(
        "PDF, Markdown or text", type=["pdf", "md", "txt"], accept_multiple_files=True
    )
    if st.button("Ingest", disabled=not uploads, type="primary", width="stretch"):
        for upload in uploads or []:
            with st.spinner(f"Ingesting {upload.name}…"):
                result = client.ingest(upload.name, upload.getvalue(), upload.type or "application/octet-stream")
            if result.ok:
                st.success(f"{upload.name} ingested in {result.data['seconds']}s")
                render_ingest_result(result.data)
            elif result.not_implemented:
                st.info(f"{result.error}.")
            else:
                st.error(f"{upload.name}: {result.error}")

    st.divider()
    st.header("🔎 Retrieval")
    mode = st.radio(
        "Mode", ["hybrid", "vector", "graph"], horizontal=True,
        help="Compare vector-only, graph-only and hybrid retrieval on the same question",
    )
    top_k = st.slider("Text excerpts sent to the LLM", 1, 10, 4)
    hops = st.radio("Graph hops", [1, 2], index=1, horizontal=True)
    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()


# ---------- chat ----------

st.title("🕸️ Hybrid GraphRAG")
st.caption("Answers grounded in Qdrant vector search + Neo4j graph traversal, with citations.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant(msg["content"])

if question := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving from Qdrant + Neo4j and writing the answer…"):
            result = client.query(question, top_k=top_k, hops=hops, mode=mode)
        if result.ok and result.data is not None:
            payload: dict[str, Any] = result.data
        elif result.not_implemented:
            payload = {"error": f"{result.error}."}
        else:
            payload = {"error": result.error or "Query failed"}
        render_assistant(payload)
    st.session_state.messages.append({"role": "assistant", "content": payload})
