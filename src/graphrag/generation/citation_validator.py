"""Check the LLM's citations against the prompt: drop invented ids, flag uncited sentences,
and return structured citations the UI can link."""

import re
from dataclasses import dataclass, field

from graphrag.models import Citation, CitedChunk, CitedFact

_CITE = re.compile(r"\[\s*([CG]\d+(?:\s*[,;]\s*[CG]\d+)*)\s*\]")  # [C1] or [C1, G2]
_ID = re.compile(r"[CG]\d+")
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+|\n+")
_MIN_SENTENCE = 25  # shorter fragments (headings, "Yes.") aren't flagged
_MARKER = re.compile(r"\[[CG]\d+\]")
_CITE_AFTER_STOP = re.compile(r"([.!?।])\s*((?:\[[CG]\d+\])+)")  # "... India. [C1]" -> "... India [C1]."


def _is_heading(sentence: str) -> bool:
    """Markdown headings / labels like "**Details:**" or "## Summary" aren't claims."""
    bare = sentence.strip().strip("*_#>- ").strip()
    return bare.endswith(":") or sentence.lstrip().startswith("#")


@dataclass
class ValidatedAnswer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)


def _citation(cite_id: str, chunks: dict[str, CitedChunk], facts: dict[str, CitedFact]) -> Citation:
    if cite_id in chunks:
        ch = chunks[cite_id].item.chunk
        return Citation(cite_id=cite_id, kind="chunk", source=ch.source, page=ch.page, section=ch.section,
                        text=ch.text)
    f = facts[cite_id]
    return Citation(cite_id=cite_id, kind="fact", source="knowledge graph", text=f.fact.as_text(),
                    evidence=f.fact.evidence)


def validate(answer: str, chunks: list[CitedChunk], facts: list[CitedFact]) -> ValidatedAnswer:
    by_chunk = {c.cite_id: c for c in chunks}
    by_fact = {f.cite_id: f for f in facts}
    valid_ids = by_chunk.keys() | by_fact.keys()
    used: list[str] = []
    invalid: list[str] = []

    def keep_valid(m: re.Match[str]) -> str:
        ids = _ID.findall(m.group(1))
        kept = [i for i in ids if i in valid_ids]
        invalid.extend(i for i in ids if i not in valid_ids)
        used.extend(i for i in kept if i not in used)
        return "".join(f"[{i}]" for i in kept)

    cleaned = _CITE.sub(keep_valid, answer)
    cleaned = re.sub(r"[ \t]+([.,;:!?।])", r"\1", cleaned)  # tidy spaces left by removed citations

    # a citation written after the full stop still belongs to that sentence
    splittable = _CITE_AFTER_STOP.sub(r" \2\1", cleaned)
    uncited = [
        s.strip()
        for s in _SENTENCE_END.split(splittable)
        if len(s.strip()) >= _MIN_SENTENCE and not _MARKER.search(s) and not _is_heading(s)
    ]
    return ValidatedAnswer(
        answer=cleaned.strip(),
        citations=[_citation(i, by_chunk, by_fact) for i in used],
        invalid=list(dict.fromkeys(invalid)),
        uncited_sentences=uncited,
    )
