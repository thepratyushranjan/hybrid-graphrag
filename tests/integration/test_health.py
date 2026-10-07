"""Runs against the live Qdrant + Neo4j containers (`make up`, then `make test`)."""

from fastapi.testclient import TestClient

from graphrag.api.main import app


def test_health_reports_both_databases_ok() -> None:
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200, resp.text
    assert resp.json()["checks"] == {"qdrant": "ok", "neo4j": "ok"}
