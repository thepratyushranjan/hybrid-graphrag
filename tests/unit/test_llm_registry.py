"""Provider selection: per-provider models, availability reasons, thinking settings only on the default."""

from graphrag.config import Settings
from graphrag.llm.client import LLMClient
from graphrag.llm.registry import LLMRegistry


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "llm_provider": "ollama", "llm_model": "gemma4:latest", "openai_api_key": None, "gemini_api_key": None,
        "llm_reasoning_effort": "none", "answer_reasoning_effort": "low", "extraction_reasoning_effort": "low",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def test_model_per_provider() -> None:
    s = _settings(openai_model="gpt-4o-mini", gemini_model="gemini-2.5-flash")
    assert s.model_for("ollama") == "gemma4:latest"
    assert s.model_for("openai") == "gpt-4o-mini" and s.model_for("gemini") == "gemini-2.5-flash"


def test_providers_without_keys_report_why() -> None:
    registry = LLMRegistry(_settings())
    assert set(registry.clients) == {"ollama"}
    assert registry.errors == {"openai": "no OPENAI_API_KEY in .env", "gemini": "no GEMINI_API_KEY in .env"}


def test_thinking_settings_only_apply_to_the_default_provider() -> None:
    s = _settings(openai_api_key="sk-test")
    local, cloud = LLMClient(s, "ollama"), LLMClient(s, "openai")
    assert local.extra == {"reasoning_effort": "none"} and local.answer_extra == {"reasoning_effort": "low"}
    assert cloud.extra == {} and cloud.answer_extra == {} and cloud.extraction_effort is None
    assert cloud.model == "gpt-4o-mini"


def test_status_lists_every_provider(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    registry = LLMRegistry(_settings(gemini_api_key="g-test"))
    monkeypatch.setattr(LLMClient, "ping", lambda self, ttl=30.0: "ok" if self.provider == "ollama" else "error: 401")
    by_name = {p.provider: p for p in registry.status()}
    assert by_name["ollama"].available and by_name["ollama"].local and by_name["ollama"].default
    assert not by_name["openai"].available and by_name["openai"].reason == "no OPENAI_API_KEY in .env"
    assert not by_name["gemini"].available and by_name["gemini"].reason == "401"
