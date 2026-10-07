"""Compare vector-only, graph-only and hybrid retrieval on the same questions (run with `make compare`).

    make compare                        # the built-in demo questions
    make compare q="your question"      # one question
"""

import sys

from graphrag.config import get_settings
from graphrag.embeddings.embedder import build_embedder
from graphrag.graph.neo4j_store import Neo4jStore
from graphrag.llm.client import LLMClient, LLMError
from graphrag.models import RetrievalResult
from graphrag.retrieval.factory import build_retriever
from graphrag.retrieval.hybrid import run_sync
from graphrag.vector_store.qdrant_store import QdrantStore

QUESTIONS = [
    "Who leads the subsidiary whose steel supplier was put on probation?",
    "Which vendors connected to Ganga Smart Power are impacted by Clause 7.2?",
    "गंगा स्मार्ट पावर के विक्रेताओं पर कौन सी धारा लागू होती है?",
    "Which organization supplies the most subsidiaries?",
    "What was the group revenue in 2025?",
]


def show(result: RetrievalResult) -> None:
    """C = chunk, A = aggregate, F = graph fact; entities marked * were seeded from vector hits."""
    a = result.analysis
    print(f"  [{result.mode.upper():6}] type={a.query_type} entities={[(e.entity_name or e.text) + ('*' if e.origin == 'vector' else '') for e in a.entities]} "
          f"templates={result.templates} filters={a.filters or '-'} time={a.time_range.expression if a.time_range else '-'} total={result.timings_ms.get('total')}ms")
    for i, c in enumerate(result.chunks, start=1):
        where = f"p{c.chunk.page}" if c.chunk.page else (c.chunk.section or "").split(" > ")[-1][:30]
        print(f"     C{i} {c.chunk.source[:28]:28} {where:30} via={'+'.join(c.found_by):12} "
              f"rerank={c.rerank_score} | {c.chunk.text[:70].replace(chr(10), ' ')}")
    for agg in result.aggregates[:5]:
        print(f"     A  {agg.name}: {agg.count}")
    for f in result.facts[:8]:
        print(f"     F  {f.as_text()}  (hops={f.hops}, rel={f.relevance}, {f.template})")


def main() -> None:
    settings = get_settings()
    embedder = build_embedder(settings)
    vectors, graph = QdrantStore(settings), Neo4jStore(settings)
    try:
        llm: LLMClient | None = LLMClient(settings)
    except LLMError as exc:
        print(f"(LLM unavailable, query analysis falls back to full-text: {exc})")
        llm = None
    retriever = build_retriever(settings, embedder, vectors, graph, llm)
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else QUESTIONS
    for q in questions:
        print(f"\n=== {q}")
        for mode in ("vector", "graph", "hybrid"):
            show(run_sync(retriever.retrieve(q, mode)))  # type: ignore[arg-type]
    vectors.close()
    graph.close()


if __name__ == "__main__":
    main()
