from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from graphrag.api.deps import get_retriever
from graphrag.models import RetrievalMode, RetrievalResult
from graphrag.retrieval.hybrid import HybridRetriever

router = APIRouter(tags=["retrieve"])


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    mode: RetrievalMode = "hybrid"
    top_k: int | None = Field(default=None, ge=1, le=20)
    hops: int | None = Field(default=None, ge=1, le=2)


@router.post("/retrieve", response_model=RetrievalResult)
async def retrieve(
    req: RetrieveRequest, retriever: Annotated[HybridRetriever, Depends(get_retriever)]
) -> RetrievalResult:
    """Hybrid retrieval only (no answer generation). `mode` = vector | graph | hybrid, for comparing branches."""
    return await retriever.retrieve(req.question, req.mode, req.top_k, req.hops)
