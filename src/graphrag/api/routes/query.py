from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from graphrag.api.deps import get_generator
from graphrag.api.routes.retrieve import RetrieveRequest
from graphrag.generation.synthesizer import AnswerGenerator, ProviderUnavailable
from graphrag.models import QueryResponse

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(
    req: RetrieveRequest, generator: Annotated[AnswerGenerator, Depends(get_generator)]
) -> QueryResponse:
    """Grounded answer with [C#] (text) and [G#] (graph) citations, validated against the evidence."""
    try:
        return await generator.answer(
            req.question, req.mode, req.top_k, req.hops, req.filters, req.rerank, req.llm_provider
        )
    except ProviderUnavailable as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
