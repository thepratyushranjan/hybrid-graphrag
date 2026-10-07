"""Embedding providers behind one interface. Token counting uses the same model's tokenizer,
so the chunker's 500-token limit matches what the embedder actually sees."""

import logging
from typing import Protocol

from graphrag.config import Settings

logger = logging.getLogger(__name__)

# (query prefix, passage prefix) the model was trained with
_HF_PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/": ("query: ", "passage: "),  # all e5 models
    "BAAI/bge-small-en": ("Represent this sentence for searching relevant passages: ", ""),
}


class EmbedderError(RuntimeError):
    pass


class Embedder(Protocol):
    dim: int

    def count_tokens(self, text: str) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HuggingFaceEmbedder:
    def __init__(self, settings: Settings) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(settings.embedding_model, device="cpu")
        self.batch_size = settings.embedding_batch_size
        self.dim = int(self.model.get_embedding_dimension() or 0)
        self.query_prefix, self.passage_prefix = next(
            (p for key, p in _HF_PREFIXES.items() if settings.embedding_model.startswith(key)), ("", "")
        )
        budget = settings.chunk_size + self.count_tokens(self.passage_prefix) + 2  # +2: [CLS]/[SEP]
        if budget > self.model.max_seq_length:
            logger.warning(
                "CHUNK_SIZE=%d plus prefix exceeds %s max length %d; chunk tails will be truncated",
                settings.chunk_size, settings.embedding_model, self.model.max_seq_length,
            )

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        import tiktoken
        from openai import OpenAI

        if not settings.openai_api_key:
            raise EmbedderError("EMBEDDING_PROVIDER=openai needs OPENAI_API_KEY")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        self.dim = settings.embedding_dim
        self.batch_size = max(settings.embedding_batch_size, 64)
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            resp = self.client.embeddings.create(model=self.model, input=texts[start : start + self.batch_size])
            vectors.extend(item.embedding for item in resp.data)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embedder(settings: Settings) -> Embedder:
    embedder: Embedder
    if settings.embedding_provider == "openai":
        embedder = OpenAIEmbedder(settings)
    else:
        embedder = HuggingFaceEmbedder(settings)
    if embedder.dim != settings.embedding_dim:
        raise EmbedderError(
            f"{settings.embedding_model} produces {embedder.dim}-dim vectors but EMBEDDING_DIM={settings.embedding_dim}"
        )
    return embedder
