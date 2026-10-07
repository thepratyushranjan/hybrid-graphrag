"""One LLM client for OpenAI, Gemini and Ollama.

All three speak the OpenAI chat-completions protocol (Gemini and Ollama through their
OpenAI-compatible endpoints), so a single SDK covers every provider.
"""

import base64
import logging

from openai import OpenAI, OpenAIError

from graphrag.config import Settings

logger = logging.getLogger(__name__)

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

CAPTION_PROMPT = (
    "Describe this image from a document in 1-3 factual sentences, for search indexing. "
    "Name what it shows (photo, chart, diagram, map, logo...), the key objects, numbers, trends or labels. "
    "Do not speculate. Reply in English."
)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        self.vision_model = settings.vision_model or settings.llm_model
        if self.provider == "openai":
            api_key, base_url = settings.openai_api_key, None
        elif self.provider == "gemini":
            api_key, base_url = settings.gemini_api_key, GEMINI_OPENAI_BASE_URL
        else:  # ollama ignores the key, but the SDK requires one
            api_key, base_url = "ollama", f"{settings.ollama_base_url.rstrip('/')}/v1"
        if not api_key:
            raise LLMError(f"LLM_PROVIDER={self.provider} needs an API key in .env")
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=180, max_retries=2)

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
