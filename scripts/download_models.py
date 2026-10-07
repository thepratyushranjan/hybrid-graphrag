"""Download the HuggingFace models at image build time so the first query doesn't stall."""

import os

from sentence_transformers import CrossEncoder, SentenceTransformer


def main() -> None:
    if os.getenv("EMBEDDING_PROVIDER", "huggingface") == "huggingface":
        embedding_model = os.environ["EMBEDDING_MODEL"]
        print(f"Downloading embedding model {embedding_model}")
        SentenceTransformer(embedding_model)
    else:
        print("EMBEDDING_PROVIDER is not huggingface, skipping embedding model download")

    reranker_model = os.environ["RERANKER_MODEL"]
    print(f"Downloading reranker model {reranker_model}")
    CrossEncoder(reranker_model)


if __name__ == "__main__":
    main()
