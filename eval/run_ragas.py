"""Ragas evaluation of vector-only vs graph-only vs hybrid answers (run with `make eval`).

For every question in eval/testset.json and every mode, the full /query pipeline answers it, then Ragas scores:
  - Faithfulness: are the answer's claims supported by the retrieved context?
  - Answer Relevancy: does the answer address the question? (needs embeddings: our multilingual e5)
  - Context Precision (with reference): are the useful contexts ranked first?
plus two reference-based metrics, because the three above don't check that the answer is *right* (an answer saying
"the evidence doesn't state it" can be perfectly faithful and relevant):
  - Factual Correctness (F1): do the answer's claims match the reference answer's?
  - Context Recall: does the retrieved context contain what the reference answer needs?
The judge is the configured LLM (LLM_PROVIDER / LLM_MODEL) through its OpenAI-compatible endpoint.

    make eval                             # all questions, all three modes
    make eval args="--limit 2 --modes hybrid"
"""

import argparse
import asyncio
import json
import logging
import math
import re
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)

from graphrag.config import get_settings
from graphrag.embeddings.embedder import build_embedder
from graphrag.generation.synthesizer import AnswerGenerator
from graphrag.graph.neo4j_store import Neo4jStore
from graphrag.llm.client import LLMClient
from graphrag.models import QueryResponse
from graphrag.retrieval.factory import build_retriever
from graphrag.vector_store.qdrant_store import QdrantStore

EVAL_DIR = Path(__file__).resolve().parent
MODES = ("vector", "graph", "hybrid")
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "factual_correctness", "context_recall")
METRIC_LABELS = {
    "faithfulness": "Faithfulness", "answer_relevancy": "Answer relevancy", "context_precision": "Context precision",
    "factual_correctness": "Factual correctness", "context_recall": "Context recall",
}
MAX_FACT_CONTEXTS = 8  # graph facts given to the judge as contexts (each costs one context-precision call)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval")


_CITATION = re.compile(r"\s*\[[CG]\d+\]")


def judged_answer(answer: str) -> str:
    """Citation markers aren't claims; the judge would count "[C1]" as unsupported text."""
    return _CITATION.sub("", answer)


def contexts_of(response: QueryResponse) -> list[str]:
    """What the answer was generated from: text excerpts, then graph facts (with their evidence) and counts."""
    contexts = [c.item.chunk.text for c in response.chunks]
    contexts += [f"{f.fact.as_text()}. Evidence: {f.fact.evidence}" for f in response.graph_facts[:MAX_FACT_CONTEXTS]]
    if response.aggregates:
        contexts.append("Counts: " + "; ".join(f"{a.name}: {a.count}" for a in response.aggregates))
    return contexts


async def score(metric: Any, **kwargs: Any) -> float | None:
    try:
        value = (await metric.ascore(**kwargs)).value
        return None if value is None or (isinstance(value, float) and math.isnan(value)) else round(float(value), 3)
    except Exception as exc:  # noqa: BLE001 - a judge failure scores None, the run continues
        logger.warning("%s failed: %s", type(metric).__name__, str(exc)[:200])
        return None


def summarise(rows: list[dict[str, Any]], modes: list[str]) -> str:
    def mean(mode: str, metric: str) -> str:
        values = [r[metric] for r in rows if r["mode"] == mode and r.get(metric) is not None]
        return f"{statistics.mean(values):.3f}" if values else "—"

    lines = [
        "| Mode | " + " | ".join(METRIC_LABELS[m] for m in METRICS) + " | Answered |",
        "|---|" + "---|" * (len(METRICS) + 1),
    ]
    for mode in modes:
        answered = sum(r["grounded"] for r in rows if r["mode"] == mode)
        total = sum(r["mode"] == mode for r in rows)
        lines.append(f"| {mode} | " + " | ".join(mean(mode, m) for m in METRICS) + f" | {answered}/{total} |")

    def cell(r: dict[str, Any] | None, metric: str) -> str:
        return "—" if r is None or r.get(metric) is None else f"{r[metric]:.2f}"

    lines += ["", "Per question — factual correctness / context recall:", "",
              "| Question | Lang | Type | " + " | ".join(modes) + " |", "|---|---|---|" + "---|" * len(modes)]
    for qid in dict.fromkeys(r["id"] for r in rows):
        q_rows = {r["mode"]: r for r in rows if r["id"] == qid}
        first = next(iter(q_rows.values()))
        cells = [f"{cell(q_rows.get(m), 'factual_correctness')} / {cell(q_rows.get(m), 'context_recall')}" for m in modes]
        lines.append(f"| {qid}: {first['question']} | {first['language']} | {first['type']} | {' | '.join(cells)} |")
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default=str(EVAL_DIR / "testset.json"))
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    modes = [m for m in args.modes.split(",") if m in MODES]
    questions = json.loads(Path(args.testset).read_text(encoding="utf-8"))[: args.limit]

    settings = get_settings()
    embedder = build_embedder(settings)
    llm = LLMClient(settings)
    vectors, graph = QdrantStore(settings), Neo4jStore(settings)
    generator = AnswerGenerator(settings, build_retriever(settings, embedder, vectors, graph, llm), llm)

    # Judge = the configured provider, through the same OpenAI-compatible endpoint the app uses
    judge_client = AsyncOpenAI(api_key=llm.client.api_key, base_url=str(llm.client.base_url), timeout=300)
    judge_kwargs: dict[str, Any] = {"temperature": 0}
    if settings.llm_reasoning_effort:
        judge_kwargs["reasoning_effort"] = settings.llm_reasoning_effort
    judge = llm_factory(settings.llm_model, provider="openai", client=judge_client, **judge_kwargs)
    ragas_embeddings = HuggingFaceEmbeddings(model=settings.embedding_model, device="cpu")
    faithfulness = Faithfulness(llm=judge)
    relevancy = AnswerRelevancy(llm=judge, embeddings=ragas_embeddings)
    precision = ContextPrecisionWithReference(llm=judge)
    correctness = FactualCorrectness(llm=judge)
    recall = ContextRecall(llm=judge)

    print(f"Ragas: {len(questions)} questions x {len(modes)} modes, judge {settings.llm_provider}/{settings.llm_model}")
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for q in questions:
        for mode in modes:
            t0 = time.perf_counter()
            response = await generator.answer(q["question"], mode)  # type: ignore[arg-type]
            contexts = contexts_of(response)
            row: dict[str, Any] = {
                "id": q["id"], "language": q["language"], "type": q["type"], "mode": mode,
                "question": q["question"], "reference": q["reference"], "answer": response.answer,
                "grounded": response.grounded, "contexts": len(contexts),
            }
            # Sequential on purpose: one judge call at a time keeps a local GPU model (and RAM) comfortable
            answer = judged_answer(response.answer)
            row["faithfulness"] = (
                await score(faithfulness, user_input=q["question"], response=answer, retrieved_contexts=contexts)
                if contexts else None
            )
            row["answer_relevancy"] = await score(relevancy, user_input=q["question"], response=answer)
            row["context_precision"] = (
                await score(precision, user_input=q["question"], reference=q["reference"], retrieved_contexts=contexts)
                if contexts else 0.0
            )
            row["factual_correctness"] = await score(correctness, response=answer, reference=q["reference"])
            row["context_recall"] = (
                await score(recall, user_input=q["question"], retrieved_contexts=contexts, reference=q["reference"])
                if contexts else 0.0
            )
            rows.append(row)
            print(f"  {q['id']} {mode:6} F={row['faithfulness']} AR={row['answer_relevancy']} "
                  f"CP={row['context_precision']} FC={row['factual_correctness']} CR={row['context_recall']} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = EVAL_DIR / "results"
    out.mkdir(exist_ok=True)
    table = summarise(rows, modes)
    meta = (f"Judge: {settings.llm_provider}/{settings.llm_model} · embeddings: {settings.embedding_model} · "
            f"{len(questions)} questions · {time.perf_counter() - started:.0f}s · {stamp} UTC")
    (out / f"ragas_{stamp}.json").write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2))
    (out / "latest.md").write_text(f"# Ragas evaluation\n\n{meta}\n\n{table}\n", encoding="utf-8")
    print(f"\n{meta}\n\n{table}\n\nSaved eval/results/latest.md and eval/results/ragas_{stamp}.json")
    vectors.close()
    graph.close()


if __name__ == "__main__":
    asyncio.run(main())
