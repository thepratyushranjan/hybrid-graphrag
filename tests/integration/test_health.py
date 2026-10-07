"""Runs against the live Qdrant + Neo4j containers (`make up`, then `make test`)."""

from fastapi.testclient import TestClient

from graphrag.api.main import app


def test_health_reports_both_databases_ok() -> None:
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200, resp.text
    checks = resp.json()["checks"]
    assert checks["qdrant"] == "ok" and checks["neo4j"] == "ok"
    assert "llm" in checks  # reported, but a missing LLM doesn't make the service unhealthy
