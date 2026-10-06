from qdrant_client import QdrantClient

from graphrag.config import Settings


class QdrantStore:
    def __init__(self, settings: Settings) -> None:
        self.collection_name = settings.collection_name
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    def ping(self) -> None:
        self.client.get_collections()

    def close(self) -> None:
        self.client.close()
