import pytest
from pydantic import ValidationError

from graphrag.config import Settings


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "neo4j_password": "password123",
        "llm_provider": "ollama",
        "llm_model": "llama3.1:8b",
        "openai_api_key": None,
        "gemini_api_key": None,
        "embedding_provider": "huggingface",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dim": 384,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def test_valid_settings_have_no_warnings() -> None:
    assert make_settings().provider_warnings() == []


def test_missing_openai_key_is_a_warning_not_an_error() -> None:
    warnings = make_settings(llm_provider="openai", llm_model="gpt-4o-mini").provider_warnings()
    assert any("OPENAI_API_KEY" in w for w in warnings)


def test_missing_gemini_key_is_a_warning() -> None:
    warnings = make_settings(llm_provider="gemini", llm_model="gemini-2.5-flash").provider_warnings()
    assert any("GEMINI_API_KEY" in w for w in warnings)


def test_empty_llm_model_is_a_warning() -> None:
    assert any("LLM_MODEL" in w for w in make_settings(llm_model="").provider_warnings())


def test_embedding_dim_mismatch_is_a_warning() -> None:
    warnings = make_settings(
        embedding_provider="openai", openai_api_key="sk-test", embedding_model="text-embedding-3-small"
    ).provider_warnings()
    assert any("EMBEDDING_DIM=384" in w for w in warnings)


def test_unsupported_llm_provider_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(llm_provider="groq")


def test_short_neo4j_password_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(neo4j_password="short")
