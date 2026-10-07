from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from graphrag.llm.registry import LLMRegistry, ProviderStatus

router = APIRouter(tags=["llm"])


def get_registry(request: Request) -> LLMRegistry:
    registry: LLMRegistry | None = getattr(request.app.state, "llm_registry", None)
    if registry is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "LLM registry not initialised")
    return registry


@router.get("/llm/providers", response_model=list[ProviderStatus])
def providers(registry: Annotated[LLMRegistry, Depends(get_registry)]) -> list[ProviderStatus]:
    """Every LLM provider, its model, and whether it can be used right now (status cached ~30 s)."""
    return registry.status()
