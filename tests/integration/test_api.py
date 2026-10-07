"""API contract: validation, stats, health, error handling (no LLM calls)."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from neo4j.exceptions import ServiceUnavailable

from graphrag.api.deps import get_generator
from graphrag.api.main import app


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize(
    ("name", "content", "metadata", "message"),
    [
        ("report.docx", b"x", None, "Unsupported file type"),
        ("fake.pdf", b"not a pdf", None, "not a valid PDF"),
        ("empty.txt", b"", None, "File is empty"),
        ("binary.txt", b"\x00\x01\x02", None, "binary file"),
        ("ok.txt", b"hello", "[1, 2]", "metadata must be a JSON object"),
    ],
)
def test_upload_validation_returns_422(
    client: TestClient, name: str, content: bytes, metadata: str | None, message: str
) -> None:
    data = {"metadata": metadata} if metadata else None
    resp = client.post("/ingest", files={"file": (name, content)}, data=data)
    assert resp.status_code == 422 and message in resp.json()["detail"]


def test_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/ingest/does-not-exist").status_code == 404


def test_stats_and_health(client: TestClient) -> None:
    stats = client.get("/stats").json()
    assert stats["qdrant"]["points"] >= 0 and "nodes" in stats["neo4j"] and "relationships" in stats["neo4j"]
    health = client.get("/health").json()
    assert {"qdrant", "neo4j", "llm"} <= health["checks"].keys()


def test_database_down_is_503(client: TestClient) -> None:
    class Down:
        async def answer(self, *args: object, **kwargs: object) -> None:
            raise ServiceUnavailable("connection refused")

    app.dependency_overrides[get_generator] = lambda: Down()
    try:
        resp = client.post("/query", json={"question": "Who leads Ganga Smart Power?"})
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 503 and "Neo4j is unavailable" in resp.json()["detail"]


def test_query_request_validation(client: TestClient) -> None:
    assert client.post("/query", json={"question": "x"}).status_code == 422  # too short
    assert client.post("/query", json={"question": "Who?", "mode": "magic"}).status_code == 422


def test_documents_are_served_safely(client: TestClient) -> None:
    pdf = client.get("/documents/vendor_compliance_bulletin_q3_2025.pdf")
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF")
    assert client.get("/documents/..%2F..%2F.env").status_code == 404  # no path traversal
    assert client.get("/documents/secrets.env").status_code == 404  # only document types
    assert client.get("/documents/never_uploaded.pdf").status_code == 404
