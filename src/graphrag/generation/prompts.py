"""Dual-context prompt: [Direct Text Excerpts] (C#) + [Graph Relationships] (G#)."""

from graphrag.models import Aggregate, CitedChunk, CitedFact, RetrievalResult

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "mr": "Marathi", "ne": "Nepali", "bn": "Bengali", "gu": "Gujarati",
    "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam", "ur": "Urdu",
    "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese", "it": "Italian", "ru": "Russian",
}

SYSTEM_PROMPT = """You answer questions using ONLY the evidence provided below. Never use outside knowledge.

Citation rules:
- Cite every claim: [C#] for text excerpts, [G#] for graph relationships, right after the claim. \
Several sources: [C1][G2].
- Only use ids that appear in the evidence. Every bullet point and every sentence that states a fact needs one.
- Never infer, assume or guess ("implied by", "likely", "probably"): state only what a [C#] or [G#] says.
- If the evidence does not contain the answer, say so plainly instead of guessing; you may say what related \
information the evidence does contain.

Evidence notes:
- Graph relationships were extracted automatically from the excerpts by an AI model and may be imprecise. \
When a relationship and an excerpt disagree, trust the excerpt.
- Questions often need several steps. If the question describes something instead of naming it \
("the subsidiary whose supplier was put on probation"), first find which entity that is, then follow its \
relationships (e.g. "Z LEADS Y") to the answer. Cite every step.

Style: answer directly in the first sentence, then add the supporting details. Be concise. State the facts \
themselves and put the citation after them; never write about the sources ("[C1] states that...", \
"the graph confirms..."). Answer in {language}."""


MAX_DATES = 3  # date mentions shown in a [C#] header
EVIDENCE_CHARS = 160  # the quote a fact was extracted from often holds the key word ("probation")


def number_evidence(result: RetrievalResult) -> tuple[list[CitedChunk], list[CitedFact]]:
    chunks = [CitedChunk(cite_id=f"C{i}", item=item) for i, item in enumerate(result.chunks, start=1)]
    by_chunk = {c.item.chunk.chunk_id: c.cite_id for c in chunks}
    facts = [
        CitedFact(cite_id=f"G{i}", fact=f, support=[by_chunk[c] for c in f.chunk_ids if c in by_chunk])
        for i, f in enumerate(result.facts, start=1)
    ]
    return chunks, facts


def _chunk_header(c: CitedChunk) -> str:
    ch = c.item.chunk
    parts = [f"doc: {ch.source}"]
    if ch.page:
        parts.append(f"page {ch.page}")
    if ch.section:
        parts.append(f"section: {ch.section}")
    if ch.dates:
        parts.append("dates: " + ", ".join(ch.dates[:MAX_DATES]) + (" …" if len(ch.dates) > MAX_DATES else ""))
    if ch.extraction_method != "text":
        parts.append(f"via {ch.extraction_method.replace('_', ' ')}")
    return f"[{c.cite_id}] ({' | '.join(parts)})"


def build_user_prompt(
    question: str, chunks: list[CitedChunk], facts: list[CitedFact], aggregates: list[Aggregate]
) -> str:
    lines = ["[Direct Text Excerpts]"]
    for c in chunks:
        lines += [_chunk_header(c), c.item.chunk.text.strip(), ""]
    if not chunks:
        lines += ["(none)", ""]

    lines.append("[Graph Relationships]")
    for f in facts:
        # a fact whose excerpt isn't in the prompt still names its document ("an excerpt not shown" made
        # models distrust the fact and skip multi-hop steps)
        src = ", ".join(f.support) if f.support else ", ".join(f.fact.sources) or "another document"
        quote = f.fact.evidence if len(f.fact.evidence) <= EVIDENCE_CHARS else f.fact.evidence[:EVIDENCE_CHARS] + "…"
        lines.append(
            f'[{f.cite_id}] {f.fact.as_typed_text()}  src: {src}, conf {f.fact.confidence:.2f} — "{quote}"'
        )
    if aggregates:
        lines.append("Counts computed over the whole graph (cite them with the related [G#] facts):")
        lines += [f"- {a.name}: {a.count}" for a in aggregates]
    if not facts and not aggregates:
        lines.append("(none)")

    lines += ["", f"QUESTION: {question}"]
    return "\n".join(lines)


def system_prompt(language: str) -> str:
    return SYSTEM_PROMPT.format(language=LANGUAGE_NAMES.get(language, "the same language as the question"))


CITATION_REMINDER = """Your previous answer cited no sources, so it can't be shown:
<<<
{answer}
>>>
Rewrite it using ONLY the evidence above. End every claim with its [C#] or [G#] citation and drop anything \
no excerpt or relationship states. If the evidence doesn't answer the question, say so in one sentence."""
