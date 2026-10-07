"""All LLM providers the server can use, their status, and the one selected per question."""

from pydantic import BaseModel

from graphrag.config import Settings
from graphrag.llm.client import LLMClient, LLMError

PROVIDERS = ("ollama", "openai", "gemini")
LOCAL = {"ollama"}


class ProviderStatus(BaseModel):
    provider: str
    model: str
    local: bool
    default: bool  # LLM_PROVIDER: builds the graph at ingest, used when a question names no provider
    available: bool
    reason: str | None = None  # why it can't be used


class LLMRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.default_provider = settings.llm_provider
        self.clients: dict[str, LLMClient] = {}
        self.errors: dict[str, str] = {}
        for provider in PROVIDERS:
            try:
                self.clients[provider] = LLMClient(settings, provider)
            except LLMError as exc:
                self.errors[provider] = str(exc)

    @property
    def default(self) -> LLMClient | None:
        return self.clients.get(self.default_provider)

    def get(self, provider: str | None) -> LLMClient | None:
        """The client for `provider` (None = default). Raises LLMError if it isn't usable."""
        name = provider or self.default_provider
        client = self.clients.get(name)
        if client is None:
            raise LLMError(f"{name} is not available: {self.errors.get(name, 'unknown provider')}")
        state = client.ping()
        if state != "ok":
            raise LLMError(f"{name} is not available: {state.removeprefix('error: ')}")
        return client

    def status(self) -> list[ProviderStatus]:
        statuses = []
        for provider in PROVIDERS:
            client = self.clients.get(provider)
            state = client.ping() if client else f"error: {self.errors[provider]}"
            statuses.append(
                ProviderStatus(
                    provider=provider,
                    model=self.settings.model_for(provider),
                    local=provider in LOCAL,
                    default=provider == self.default_provider,
                    available=state == "ok",
                    reason=None if state == "ok" else state.removeprefix("error: "),
                )
            )
        return statuses
