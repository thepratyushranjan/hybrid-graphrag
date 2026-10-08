from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from graphrag.api.deps import get_retriever
from graphrag.models import QueryFilters, RetrievalMode, RetrievalResult
from graphrag.retrieval.hybrid import HybridRetriever

router = APIRouter(tags=["retrieve"])


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    mode: RetrievalMode = "hybrid"
    top_k: int | None = Field(default=None, ge=1, le=20)
    hops: int | None = Field(default=None, ge=1, le=2)
    filters: QueryFilters | None = Field(default=None, description="Explicit filters (never dropped)")
    rerank: bool | None = Field(default=None, description="Cross-encoder reranking; null = server default")
    llm_provider: Literal["ollama", "openai", "gemini"] | None = Field(
        default=None, description="LLM for query analysis + answer (see GET /llm/providers); null = server default"
    )
    corpus: Literal["docs", "sql"] = Field(
        default="docs", description="docs = uploaded documents; sql = social-media posts loaded by POST /ingest/sql"
    )


@router.post("/retrieve", response_model=RetrievalResult)
async def retrieve(
    req: RetrieveRequest, request: Request, retriever: Annotated[HybridRetriever, Depends(get_retriever)]
) -> RetrievalResult:
    """Hybrid retrieval only (no answer generation). `mode` = vector | graph | hybrid, for comparing branches."""
    if req.corpus == "sql":
        sql_retriever = getattr(request.app.state, "sql_retriever", None)
        if sql_retriever is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "The SQL corpus is not available")
        return await sql_retriever.retrieve(req.question, req.mode, req.top_k, req.hops, req.filters, req.rerank)
    return await retriever.retrieve(req.question, req.mode, req.top_k, req.hops, req.filters, req.rerank)
