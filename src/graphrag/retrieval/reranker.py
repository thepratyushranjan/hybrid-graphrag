"""Bonus: multilingual cross-encoder reranking of the fused candidates (and relevance scores for graph facts)."""

import numpy as np

from graphrag.config import Settings
from graphrag.models import RankedChunk


class Reranker:
    def __init__(self, settings: Settings) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(settings.reranker_model, device="cpu")

    def rerank(self, question: str, items: list[RankedChunk]) -> list[RankedChunk]:
        if not items:
            return items
        scores = self.model.predict([(question, item.chunk.text) for item in items], show_progress_bar=False)
        for item, score in zip(items, scores, strict=True):
            item.rerank_score = round(float(score), 4)
            item.score = item.rerank_score
        return sorted(items, key=lambda c: -(c.rerank_score or 0.0))

    def relevance(self, question: str, texts: list[str]) -> list[float]:
        """0-1 relevance of each text to the question (sigmoid of the cross-encoder logit)."""
        if not texts:
            return []
        logits = np.asarray(self.model.predict([(question, t) for t in texts], show_progress_bar=False))
        return (1 / (1 + np.exp(-logits))).tolist()
