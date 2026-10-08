"""Thin HTTP client the Streamlit UI uses to talk to the FastAPI service."""

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class ApiResult:
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    not_implemented: bool = False


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 300.0) -> None:  # /query may wait for a local LLM
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> ApiResult:
        """GET/POST and wrap the outcome; list responses are wrapped as {"items": [...]}."""
        try:
            resp = httpx.request(method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs)
        except httpx.HTTPError as exc:
            return ApiResult(ok=False, error=f"Cannot reach API at {self.base_url}: {exc}")

        if resp.status_code == 405 or (resp.status_code == 404 and not path.startswith(("/ingest", "/documents/"))):
            return ApiResult(ok=False, not_implemented=True, error=f"{method} {path} is not implemented yet")
        try:
            body = resp.json()
        except ValueError:
            body = {"detail": resp.text}
        if isinstance(body, list):
            body = {"items": body}
        if resp.is_success:
            return ApiResult(ok=True, data=body if isinstance(body, dict) else {"items": body})
        # /health returns a useful body with 503, so keep it
        return ApiResult(ok=False, data=body, error=str(body.get("detail", f"HTTP {resp.status_code}")))

    def health(self) -> ApiResult:
        return self._request("GET", "/health")

    def ingest(self, filename: str, content: bytes, content_type: str) -> ApiResult:
        """Queues the document; the response is a job (poll `job`)."""
        return self._request("POST", "/ingest", files={"file": (filename, content, content_type)})

    def job(self, job_id: str) -> ApiResult:
        return self._request("GET", f"/ingest/{job_id}")

    def query(
        self, question: str, top_k: int, hops: int, mode: str = "hybrid", rerank: bool = True,
        llm_provider: str | None = None, corpus: str = "docs",
    ) -> ApiResult:
        return self._request(
            "POST",
            "/query",
            json={"question": question, "top_k": top_k, "hops": hops, "mode": mode, "rerank": rerank,
                  "llm_provider": llm_provider, "corpus": corpus},
        )

    def ingest_sql(self, filename: str | None = None, content: bytes | None = None) -> ApiResult:
        """Queues a SQL-dump ingest; without a file the server loads its SQL_DUMP_PATH."""
        files = {"file": (filename, content, "application/sql")} if filename and content else None
        return self._request("POST", "/ingest/sql", files=files)

    def sql_job(self, job_id: str) -> ApiResult:
        return self._request("GET", f"/ingest/sql/{job_id}")

    def providers(self) -> ApiResult:
        return self._request("GET", "/llm/providers")

    def stats(self) -> ApiResult:
        return self._request("GET", "/stats")
