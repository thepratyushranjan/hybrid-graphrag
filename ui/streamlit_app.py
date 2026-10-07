"""Hybrid GraphRAG chat UI: grounded answers with clickable [C#] (text) / [G#] (graph) citations, the evidence
behind them, an interactive subgraph and per-stage timings. Talks to the FastAPI service only, never to the
databases directly."""

import os
import time
from typing import Any

import streamlit as st

from api_client import ApiClient
from components.evidence import render_answer, render_chunk_cards, render_citations, render_triples
from components.graph_view import render_subgraph
from components.timing import render_timing

API_URL = os.getenv("API_URL", "http://localhost:8000")  # container-to-container
PUBLIC_API_URL = os.getenv("PUBLIC_API_URL", "http://localhost:8000").rstrip("/")  # links opened by the browser

st.set_page_config(page_title="Hybrid GraphRAG", page_icon="🕸️", layout="wide")
client = ApiClient(API_URL)

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------- rendering ----------

def render_ingest_result(data: dict[str, Any]) -> None:
    if data.get("ocr_pages") or data.get("ocr_images"):
        st.caption(f"🔍 OCR: {data.get('ocr_pages', 0)} scanned page(s), {data.get('ocr_images', 0)} image(s)")
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
    st.caption(f"Pages: {data.get('pages')} · Languages: {data.get('languages')} · "
               f"Extraction: {data.get('extraction_methods')}")


def render_analysis(data: dict[str, Any]) -> None:
    a = data["analysis"]
    parts = [f"mode **{data['mode']}**", f"query type **{a['query_type']}**", f"language **{a['language']}**"]
    if data.get("llm"):
        parts.insert(0, f"🤖 **{data['llm']['provider']} · {data['llm']['model']}**")
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
    st.caption(" · ".join(parts))
    for dropped in a.get("dropped_filters", []):
        st.caption(f"⚠️ Filter ignored: {dropped}")


def render_assistant(payload: dict[str, Any], msg: int) -> None:
    if payload.get("error"):
        st.warning(payload["error"])
        return
    citations = payload.get("citations", [])
    used = {c["cite_id"] for c in citations}
    render_answer(payload["answer"], citations, msg)
    if payload.get("intent") == "smalltalk":
        return  # greetings / thanks / help: nothing was searched

    if not citations:
        # nothing from the knowledge base was used: don't flood the chat with evidence the answer ignored
        searched = len(payload.get("chunks", [])) + len(payload.get("graph_facts", []))
        reason = "no relevant evidence was found" if not payload.get("grounded") else "the answer cites no sources"
        st.caption(f"ℹ️ Nothing from the knowledge base was used ({reason}; {searched} items searched).")
        return

    if payload.get("invalid_citations"):
        st.caption("⚠️ Removed citations that weren't in the evidence: " + ", ".join(payload["invalid_citations"]))
    if payload.get("uncited_sentences"):
        st.caption(f"⚠️ {len(payload['uncited_sentences'])} sentence(s) without a citation")
    render_citations(citations, msg, PUBLIC_API_URL)
    render_analysis(payload)
    render_chunk_cards(payload.get("chunks", []), used, PUBLIC_API_URL)
    render_triples(payload.get("graph_facts", []), payload.get("aggregates", []), used)
    if payload.get("subgraph", {}).get("edges"):
        with st.expander("🧭 Interactive subgraph"):
            render_subgraph(payload["subgraph"], used, key=f"graph-{msg}")
    render_timing(payload.get("timings_ms", {}))


# ---------- sidebar ----------

with st.sidebar:
    st.header("⚙️ System")
    health = client.health()
    if health.data:
        for name, state in health.data.get("checks", {}).items():
            st.markdown(f"{'🟢' if state == 'ok' else '🔴'} **{name}** — {state}")
        llm, emb = health.data.get("llm", {}), health.data.get("embedding", {})
        st.caption(f"LLM: {llm.get('provider')} / {llm.get('model')}  \n"
                   f"Embeddings: {emb.get('provider')} / {emb.get('model')}")
        for warning in health.data.get("warnings", []):
            st.warning(warning, icon="⚠️")
    else:
        st.error(health.error or "API unavailable")

    stats = client.stats()
    if stats.ok and stats.data:
        nodes, rels = stats.data["neo4j"]["nodes"], stats.data["neo4j"]["relationships"]
        s1, s2, s3 = st.columns(3)
        s1.metric("Documents", nodes.get("Document", 0))
        s2.metric("Vectors", stats.data["qdrant"]["points"])
        s3.metric("Entities", nodes.get("Entity", 0))
        with st.expander("Database stats"):
            st.markdown("**Nodes**")
            st.dataframe([{"Label": k, "Count": v} for k, v in nodes.items()], hide_index=True)
            st.markdown("**Relationships**")
            st.dataframe([{"Type": k, "Count": v} for k, v in rels.items()], hide_index=True)
            st.markdown("**Documents**")
            for d in stats.data.get("documents", []):
                st.markdown(f"- [{d['source']}]({PUBLIC_API_URL}/documents/{d['source']})")
    if st.button("Refresh", width="stretch"):
        st.rerun()

    st.divider()
    st.header("📥 Ingest documents")
    uploads = st.file_uploader("PDF, Markdown or text · any language", type=["pdf", "md", "txt"],
                               accept_multiple_files=True)
    if st.button("Ingest", disabled=not uploads, type="primary", width="stretch"):
        for upload in uploads or []:
            submitted = client.ingest(upload.name, upload.getvalue(), upload.type or "application/octet-stream")
            if not submitted.ok or not submitted.data:
                st.error(f"{upload.name}: {submitted.error}")
                continue
            job_id = submitted.data["job_id"]
            with st.status(f"{upload.name}: queued", expanded=False) as box:
                job = submitted.data
                while job["status"] in ("queued", "running"):  # poll the background job
                    time.sleep(1)
                    polled = client.job(job_id)
                    if not polled.ok or not polled.data:
                        break
                    job = polled.data
                    box.update(label=f"{upload.name}: {job['stage']}")
                if job["status"] == "completed":
                    box.update(label=f"{upload.name} ingested in {job['result']['seconds']}s", state="complete")
                    render_ingest_result(job["result"])
                else:
                    box.update(label=f"{upload.name}: {job['status']}", state="error")
                    for error in job.get("errors", []):
                        st.error(error)

    st.divider()
    st.header("🤖 Answer model")
    providers = client.providers()
    usable = [p for p in (providers.data or {}).get("items", []) if p["available"]] if providers.ok else []
    blocked = [p for p in (providers.data or {}).get("items", []) if not p["available"]] if providers.ok else []
    llm_provider: str | None = None
    if usable:
        # default: the local model if it's running, else the server's default provider
        default = next((i for i, p in enumerate(usable) if p["local"]),
                       next((i for i, p in enumerate(usable) if p["default"]), 0))
        labels = {p["provider"]: f"{p['provider'].capitalize()} · {p['model']}" + (" (local)" if p["local"] else "")
                  for p in usable}
        llm_provider = st.selectbox("Used for query analysis and the answer", list(labels), index=default,
                                    format_func=labels.get)
        st.caption("The knowledge graph is built at upload time by the server's default model, so every model "
                   "answers from the same evidence.")
    else:
        st.error(providers.error if not providers.ok else "No LLM provider is available")
    for p in blocked:
        st.caption(f"❌ {p['provider'].capitalize()} · {p['model']} — {p['reason']}")

    st.divider()
    st.header("🔎 Retrieval")
    mode = st.radio("Mode", ["hybrid", "vector", "graph"], horizontal=True,
                    help="Compare vector-only, graph-only and hybrid retrieval on the same question")
    top_k = st.slider("Text excerpts sent to the LLM", 1, 10, 4)
    hops = st.radio("Graph hops", [1, 2], index=1, horizontal=True)
    rerank = st.toggle("Cross-encoder reranker", value=True,
                       help="Rerank excerpts and graph facts with the multilingual cross-encoder")
    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()


# ---------- chat ----------

st.title("🕸️ Hybrid GraphRAG")
st.caption("Answers grounded in Qdrant vector search + Neo4j graph traversal, with citations.")

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant(msg["content"], i)

if question := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving from Qdrant + Neo4j and writing the answer…"):
            result = client.query(question, top_k=top_k, hops=hops, mode=mode, rerank=rerank,
                                  llm_provider=llm_provider)
        if result.ok and result.data is not None:
            payload: dict[str, Any] = result.data
        else:
            payload = {"error": result.error or "Query failed"}
        render_assistant(payload, len(st.session_state.messages))
    st.session_state.messages.append({"role": "assistant", "content": payload})
