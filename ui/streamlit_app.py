"""Hybrid GraphRAG chat UI.

Expected /query response:
    {"answer": str,
     "chunks": [{"chunk_id", "doc_id", "page", "score", "text"}],
     "graph_facts": [{"subject", "predicate", "object", "chunk_id"}]}
"""

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

def render_chunks(chunks: list[dict[str, Any]]) -> None:
    with st.expander(f"📄 Retrieved chunks ({len(chunks)})"):
        if not chunks:
            st.caption("No chunks returned.")
        for i, chunk in enumerate(chunks, start=1):
            with st.container(border=True):
                meta = f"**[C{i}]** `{chunk.get('doc_id', '?')}` · page {chunk.get('page', '-')}"
                if chunk.get("score") is not None:
                    meta += f" · score {chunk['score']:.3f}"
                st.markdown(meta)
                st.write(chunk.get("text", ""))


def render_graph_facts(facts: list[dict[str, Any]]) -> None:
    with st.expander(f"🕸️ Graph relationships ({len(facts)})"):
        if not facts:
            st.caption("No graph facts returned.")
            return
        st.dataframe(
            [
                {
                    "Subject": f.get("subject"),
                    "Predicate": f.get("predicate"),
                    "Object": f.get("object"),
                    "Source chunk": f.get("chunk_id"),
                }
                for f in facts
            ],
            width="stretch",
            hide_index=True,
        )


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
    st.markdown(payload.get("answer", ""))
    render_chunks(payload.get("chunks", []))
    render_graph_facts(payload.get("graph_facts", []))


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
                st.info(f"{result.error} (Milestone 5).")
            else:
                st.error(f"{upload.name}: {result.error}")

    st.divider()
    st.header("🔎 Retrieval")
    top_k = st.slider("Top-K chunks (Qdrant)", 1, 10, 4)
    hops = st.radio("Graph hops (Neo4j)", [1, 2], horizontal=True)
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
        with st.spinner("Retrieving from Qdrant + Neo4j…"):
            result = client.query(question, top_k=top_k, hops=hops)
        if result.ok and result.data is not None:
            payload: dict[str, Any] = result.data
        elif result.not_implemented:
            payload = {"error": f"{result.error} — it arrives in Milestone 4/5."}
        else:
            payload = {"error": result.error or "Query failed"}
        render_assistant(payload)
    st.session_state.messages.append({"role": "assistant", "content": payload})
