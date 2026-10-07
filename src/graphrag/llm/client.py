"""One LLM client for OpenAI, Gemini and Ollama.

All three speak the OpenAI chat-completions protocol (Gemini and Ollama through their
OpenAI-compatible endpoints), so a single SDK covers every provider.
"""

import base64
import logging
import re
import time
from typing import TypeVar

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

from graphrag.config import Settings

logger = logging.getLogger(__name__)

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

CAPTION_PROMPT = (
    "Describe this image from a document in 1-3 factual sentences, for search indexing. "
    "Name what it shows (photo, chart, diagram, map, logo...), the key objects, numbers, trends or labels. "
    "Do not speculate. Reply in English."
)


T = TypeVar("T", bound=BaseModel)

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings, provider: str | None = None) -> None:
        self.provider = provider or settings.llm_provider
        self.model = settings.model_for(self.provider)
        is_default = self.provider == settings.llm_provider
        self.vision_model = (settings.vision_model if is_default else None) or self.model
        # e.g. "none" turns off "thinking" on local models like gemma4 (much faster); unset for gpt-4o-mini
        # "thinking" settings are tuned for the default (local) model; other providers get their own defaults
        # (gpt-4o-mini rejects the parameter)
        llm_effort = settings.llm_reasoning_effort if is_default else None
        answer_effort = settings.answer_reasoning_effort if is_default else None
        self.extraction_effort = settings.extraction_reasoning_effort if is_default else None
        self.extra: dict[str, str] = {"reasoning_effort": llm_effort} if llm_effort else {}
        self.answer_extra: dict[str, str] = {"reasoning_effort": answer_effort} if answer_effort else self.extra
        if self.provider == "openai":
            api_key, base_url = settings.openai_api_key, None
        elif self.provider == "gemini":
            api_key, base_url = settings.gemini_api_key, GEMINI_OPENAI_BASE_URL
        else:  # ollama ignores the key, but the SDK requires one
            api_key, base_url = "ollama", f"{settings.ollama_base_url.rstrip('/')}/v1"
        if not api_key:
            key = "OPENAI_API_KEY" if self.provider == "openai" else "GEMINI_API_KEY"
            raise LLMError(f"no {key} in .env")
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=180, max_retries=2)
        self._ping_cache: tuple[float, str] | None = None

    def ping(self, ttl: float = 30.0) -> str:
        """'ok' if the provider answers and serves the model; cached for `ttl` seconds (health is polled)."""
        if self._ping_cache and time.monotonic() - self._ping_cache[0] < ttl:
            return self._ping_cache[1]
        try:
            fast = self.client.with_options(timeout=5, max_retries=0)
            ids = {m.id.removeprefix("models/") for m in fast.models.list()}
            state = "ok" if not ids or self.model in ids else f"error: model '{self.model}' not available"
        except OpenAIError as exc:
            state = f"error: {exc.__class__.__name__}: {exc}"
        self._ping_cache = (time.monotonic(), state)
        return state

    def describe_image(self, png: bytes, prompt: str = CAPTION_PROMPT) -> str:
        image_url = "data:image/png;base64," + base64.b64encode(png).decode()
        try:
            resp = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                temperature=0.2,
                max_tokens=1024,  # generous: "thinking" models spend tokens before answering
            )
        except OpenAIError as exc:
            raise LLMError(f"{self.provider}/{self.vision_model} image description failed: {exc}") from exc
        return (resp.choices[0].message.content or "").strip()

    def complete_json(
        self, system: str, user: str, schema: type[T], max_tokens: int = 4096, reasoning_effort: str | None = None
    ) -> T:
        """Chat completion constrained to `schema` (JSON schema response format), validated with Pydantic.

        On invalid JSON the model gets one retry with the validation error appended.
        """
        messages: list[dict[str, str]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        }
        last_error: Exception | None = None
        for _attempt in range(2):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=0,
                    max_tokens=max_tokens,
                    response_format=response_format,  # type: ignore[arg-type]
                    **({"reasoning_effort": reasoning_effort} if reasoning_effort else self.extra),  # type: ignore[arg-type]
                )
            except OpenAIError as exc:
                raise LLMError(f"{self.provider}/{self.model} request failed: {exc}") from exc
            content = _JSON_FENCE.sub("", (resp.choices[0].message.content or "").strip())
            try:
                return schema.model_validate_json(content)
            except ValidationError as exc:
                last_error = exc
                messages += [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"That JSON was invalid: {exc.errors()[:3]}. Reply with corrected JSON only."},
                ]
        raise LLMError(f"{self.provider}/{self.model} returned invalid JSON twice: {last_error}")

    def complete(self, system: str, user: str, max_tokens: int = 1500, temperature: float = 0.0) -> str:
        """Plain-text chat completion (used for the grounded answer)."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens,
                **self.answer_extra,  # type: ignore[arg-type]
            )
        except OpenAIError as exc:
            raise LLMError(f"{self.provider}/{self.model} request failed: {exc}") from exc
        return (resp.choices[0].message.content or "").strip()
